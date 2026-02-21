# CLAUDE.md

## Project Overview

This project extracts MIDI files from Synthesia-style piano tutorial videos by detecting key illumination using computer vision.

## Key Files

- `extract_midi.py` - Main extraction script with `SynthesiaExtractor` class

## Architecture

The extraction process:

1. **Key Position Mapping**: Uses calibrated C note positions to calculate x-coordinates for all 88 piano keys. White keys are positioned at the center of their visual width; black keys are positioned at the boundary between adjacent white keys.

2. **Color Detection**: Samples a 7x7 pixel region at each key position and checks for green (left hand) or blue (right hand) illumination in HSV color space:
   - Green: H 35-85, S > 50, V > 50
   - Blue: H 85-135, S > 50, V > 50

3. **Debounce Logic**: Requires 2 consecutive unlit frames before releasing a note to prevent false note-offs from video compression artifacts.

4. **MIDI Output**: Creates a two-track MIDI file with left hand on track 0 and right hand on track 1.

## Calibration Values

Default calibration for 1276x720 video:
- Key sample Y: 500 (white keys), 515 (black keys)
- C note left edges (MIDI note: x-pixel):
  - C1 (24): 51
  - C2 (36): 224
  - C3 (48): 396
  - C4 (60): 567
  - C5 (72): 742
  - C6 (84): 914
  - C7 (96): 1086

## Development Notes

- The `--debug` flag saves a calibration image at frame 14s showing detection regions
- White key detection is shifted 3 pixels left from calculated center
- Black key detection is 15 pixels below the white key sample line
- `lit_threshold` of 0.3 means 30% of sampled pixels must match the color
