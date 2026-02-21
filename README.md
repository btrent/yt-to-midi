# yt-to-midi

Extract MIDI files from Synthesia-style piano tutorial videos by detecting key illumination.

## How It Works

This tool analyzes piano tutorial videos where keys light up when pressed (typically green for left hand, blue for right hand). It:

1. Samples a region of pixels at each key position on the keyboard
2. Detects green/blue illumination using HSV color space
3. Tracks note on/off timing with debounce to handle video compression artifacts
4. Outputs a two-track MIDI file (left hand / right hand)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/yt-to-midi.git
cd yt-to-midi

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install opencv-python numpy midiutil
```

## Usage

```bash
# Basic usage
python extract_midi.py video.mp4 -o output.mid

# With options
python extract_midi.py video.mp4 -o output.mid --skip 5.0 --debug --analyze
```

### Options

- `-o, --output`: Output MIDI file path (default: `output.mid`)
- `-s, --skip`: Seconds to skip at start of video (default: `5.0`)
- `-y, --key-y`: Y coordinate for key detection line (default: `500`)
- `--debug`: Save a debug image showing detected key positions
- `--analyze`: Print note analysis after extraction

## Calibration

The default calibration is for 1276x720 resolution videos. For different video resolutions or keyboard positions, you may need to:

1. Run with `--debug` to generate a calibration image
2. Adjust the `--key-y` parameter to match where keys illuminate
3. For significantly different resolutions, modify the `c_left_edge` dictionary in the code

## Limitations

- Optimized for standard Synthesia-style videos with green/blue key colors
- Requires manual calibration for different video resolutions
- Does not detect velocity (all notes output at velocity 100)
- Tempo is fixed at 120 BPM (timing is preserved in absolute terms)

## License

MIT
