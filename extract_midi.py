#!/usr/bin/env python3
"""
Extract MIDI from Synthesia-style piano tutorial videos.

Detects key illumination (green for left hand, blue for right hand) and
converts to a two-track MIDI file with accurate timing.
"""

import argparse
import cv2
import numpy as np
from midiutil import MIDIFile


class SynthesiaExtractor:
    def __init__(self, video_path, key_y=500, c_positions=None):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Video: {self.width}x{self.height}, {self.fps} fps")

        self.key_sample_y = key_y

        # Sample region size (pixels)
        self.sample_width = 7
        self.sample_height = 7

        # What fraction of pixels in the region must be lit to count as "key pressed"
        self.lit_threshold = 0.3

        # Debounce: consecutive unlit frames before releasing
        self.release_debounce_frames = 2

        # C note positions (LEFT EDGE of key) - calibrated for 1276x720 video
        self.c_left_edge = c_positions or {
            24: 51, 36: 224, 48: 396, 60: 567, 72: 742, 84: 914, 96: 1086,
        }

        c_notes = sorted(self.c_left_edge.keys())
        octave_widths = [self.c_left_edge[c_notes[i+1]] - self.c_left_edge[c_notes[i]]
                        for i in range(len(c_notes) - 1)]
        self.octave_width = np.mean(octave_widths)
        self.white_key_width = self.octave_width / 7

        self.note_positions = {}
        self._build_note_positions()

    def _build_note_positions(self):
        """Build x-positions for all 88 piano keys based on C note calibration."""
        for c_midi, c_left in self.c_left_edge.items():
            white_positions = [0, 2, 4, 5, 7, 9, 11]
            for i, semitone in enumerate(white_positions):
                midi = c_midi + semitone
                x = c_left + (i + 0.5) * self.white_key_width
                self.note_positions[midi] = {'x': int(x), 'is_black': False}

            black_info = [(1, 0, 1), (3, 1, 2), (6, 3, 4), (8, 4, 5), (10, 5, 6)]
            for semitone, left_white, right_white in black_info:
                midi = c_midi + semitone
                x = c_left + (left_white + 1) * self.white_key_width
                self.note_positions[midi] = {'x': int(x), 'is_black': True}

        c7_left = self.c_left_edge[96]
        for i, semitone in enumerate([0, 2, 4, 5, 7, 9, 11]):
            midi = 96 + semitone
            if midi <= 108:
                x = c7_left + (i + 0.5) * self.white_key_width
                self.note_positions[midi] = {'x': int(x), 'is_black': False}

        print(f"Mapped {len(self.note_positions)} keys")

    def is_key_lit(self, frame, midi):
        """Check if a key is lit by sampling a region of pixels."""
        if midi not in self.note_positions:
            return False, None

        info = self.note_positions[midi]
        cx = info['x']

        # Position adjustments for accurate detection
        if info['is_black']:
            cy = self.key_sample_y + 15
        else:
            cx = cx - 3
            cy = self.key_sample_y

        half_w = self.sample_width // 2
        half_h = self.sample_height // 2

        x1 = max(0, cx - half_w)
        x2 = min(self.width, cx + half_w + 1)
        y1 = max(0, cy - half_h)
        y2 = min(self.height, cy + half_h + 1)

        if x2 <= x1 or y2 <= y1:
            return False, None

        region = frame[y1:y2, x1:x2]
        hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

        h = hsv_region[:, :, 0]
        s = hsv_region[:, :, 1]
        v = hsv_region[:, :, 2]

        green_mask = (h >= 35) & (h <= 85) & (s > 50) & (v > 50)
        blue_mask = (h >= 85) & (h <= 135) & (s > 50) & (v > 50)

        total_pixels = region.shape[0] * region.shape[1]
        green_ratio = np.sum(green_mask) / total_pixels
        blue_ratio = np.sum(blue_mask) / total_pixels

        if green_ratio >= self.lit_threshold:
            return True, 'left'
        if blue_ratio >= self.lit_threshold:
            return True, 'right'

        return False, None

    def extract(self, output_path, skip_seconds=5.0, min_duration=0.03, debug=False):
        """Extract notes from video and save as MIDI file."""
        if debug:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(14 * self.fps))
            ret, debug_frame = self.cap.read()
            if ret:
                self._save_debug(debug_frame, "debug_calibration.jpg")

        start_frame = int(skip_seconds * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        note_state = {}
        all_notes = []
        frame_num = start_frame

        print(f"Processing from {skip_seconds}s...")

        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            time_sec = frame_num / self.fps

            if frame_num % int(self.fps * 20) == 0:
                print(f"  {time_sec:.0f}s / {self.total_frames/self.fps:.0f}s")

            for midi in self.note_positions:
                is_lit, hand = self.is_key_lit(frame, midi)

                if midi not in note_state:
                    note_state[midi] = {
                        'active': False,
                        'start_time': None,
                        'hand': None,
                        'unlit_count': 999
                    }

                state = note_state[midi]

                if is_lit:
                    state['unlit_count'] = 0

                    if not state['active']:
                        state['active'] = True
                        state['start_time'] = time_sec
                        state['hand'] = hand
                else:
                    state['unlit_count'] += 1

                    if state['active'] and state['unlit_count'] >= self.release_debounce_frames:
                        end_time = time_sec - (self.release_debounce_frames - 1) / self.fps
                        dur = end_time - state['start_time']

                        if dur >= min_duration:
                            all_notes.append({
                                'midi': midi,
                                'start': state['start_time'],
                                'duration': dur,
                                'hand': state['hand']
                            })

                        state['active'] = False
                        state['start_time'] = None

            frame_num += 1

        # Close remaining notes
        final_time = self.total_frames / self.fps
        for midi, state in note_state.items():
            if state['active'] and state['start_time'] is not None:
                dur = final_time - state['start_time']
                if dur >= min_duration:
                    all_notes.append({
                        'midi': midi,
                        'start': state['start_time'],
                        'duration': dur,
                        'hand': state['hand']
                    })

        self.cap.release()

        print(f"\nExtracted {len(all_notes)} notes")
        self._save_midi(all_notes, output_path)

        return all_notes

    def _save_midi(self, notes, path, tempo=120):
        """Save extracted notes to a MIDI file with two tracks (left/right hand)."""
        midi = MIDIFile(2)
        midi.addTrackName(0, 0, "Left Hand")
        midi.addTrackName(1, 0, "Right Hand")
        midi.addTempo(0, 0, tempo)

        sec_per_beat = 60.0 / tempo

        for note in notes:
            track = 0 if note['hand'] == 'left' else 1
            midi.addNote(track, 0, note['midi'],
                        note['start'] / sec_per_beat,
                        note['duration'] / sec_per_beat, 100)

        with open(path, 'wb') as f:
            midi.writeFile(f)
        print(f"Saved: {path}")

    def _save_debug(self, frame, path):
        """Save a debug image showing detected key positions."""
        vis = frame.copy()

        cv2.line(vis, (0, self.key_sample_y), (self.width, self.key_sample_y), (0, 255, 255), 1)
        cv2.line(vis, (0, self.key_sample_y + 15), (self.width, self.key_sample_y + 15), (255, 255, 0), 1)

        for midi, info in self.note_positions.items():
            if info['is_black']:
                x = info['x']
                y = self.key_sample_y + 15
            else:
                x = info['x'] - 3
                y = self.key_sample_y

            half_w = self.sample_width // 2
            half_h = self.sample_height // 2

            is_lit, hand = self.is_key_lit(frame, midi)

            if is_lit:
                color = (0, 255, 0) if hand == 'left' else (255, 0, 0)
                cv2.rectangle(vis, (x - half_w, y - half_h), (x + half_w, y + half_h), color, 2)
            else:
                cv2.rectangle(vis, (x - half_w, y - half_h), (x + half_w, y + half_h), (128, 128, 128), 1)

            if midi % 12 == 0:
                cv2.putText(vis, f"C{midi//12-1}", (x-10, y-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        cv2.imwrite(path, vis)
        print(f"Debug image: {path}")


def analyze(notes):
    """Print analysis of extracted notes."""
    if not notes:
        return

    left = [n for n in notes if n['hand'] == 'left']
    right = [n for n in notes if n['hand'] == 'right']
    pitches = [n['midi'] for n in notes]
    durations = [n['duration'] for n in notes]

    print(f"\nAnalysis:")
    print(f"  Total: {len(notes)}, Left: {len(left)}, Right: {len(right)}")
    print(f"  MIDI range: {min(pitches)} - {max(pitches)}")
    print(f"  Duration range: {min(durations):.3f}s - {max(durations):.3f}s")
    print(f"  Avg duration: {np.mean(durations):.3f}s")

    print(f"\nFirst 25 notes:")
    for n in sorted(notes, key=lambda x: x['start'])[:25]:
        h = "L" if n['hand'] == 'left' else "R"
        name = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][n['midi'] % 12]
        print(f"  {n['start']:.3f}s: {h} {name}{n['midi']//12-1} (MIDI {n['midi']}) dur={n['duration']:.3f}s")


def main():
    parser = argparse.ArgumentParser(
        description="Extract MIDI from Synthesia-style piano tutorial videos"
    )
    parser.add_argument("video", help="Path to the video file")
    parser.add_argument("-o", "--output", help="Output MIDI file path (default: output.mid)",
                       default="output.mid")
    parser.add_argument("-s", "--skip", type=float, default=5.0,
                       help="Seconds to skip at start (default: 5.0)")
    parser.add_argument("-y", "--key-y", type=int, default=500,
                       help="Y coordinate for key detection (default: 500)")
    parser.add_argument("--debug", action="store_true",
                       help="Save debug calibration image")
    parser.add_argument("--analyze", action="store_true",
                       help="Print note analysis after extraction")

    args = parser.parse_args()

    extractor = SynthesiaExtractor(args.video, key_y=args.key_y)
    notes = extractor.extract(args.output, skip_seconds=args.skip, debug=args.debug)

    if args.analyze and notes:
        analyze(notes)


if __name__ == "__main__":
    main()
