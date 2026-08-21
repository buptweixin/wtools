# wtools

A personal toolbox of utility functions for computer vision tasks, including file I/O helpers, image processing, visualization, memory monitoring, and head-pose estimation from facial landmarks.

## Installation

### From source (recommended)

```bash
git clone https://github.com/buptweixin/wtools.git
cd wtools
pip install -r requirements.txt
python setup.py install
```

### Dependencies

The following packages are required (see `requirements.txt`):

| Package | Purpose |
|---|---|
| numpy | Array operations throughout |
| opencv-python | Image decoding / encoding / drawing |
| lmdb | LMDB key-value store wrapper |
| PyYAML | YAML config loading |
| matplotlib | Keypoint visualization |
| psutil | Process memory monitoring |
| tabulate | Memory monitor table formatting |
| click | CLI for `gen_pose.py` |
| tqdm | Progress bars in `gen_pose.py` |
| ipython | Notebook environment detection |
| av | Video decoding via PyAV (optional, required for video_to_jpeg_bin) |

## Features

- **File I/O** (`wtools.utils.io`) -- Load/dump helpers for pickle, JSON, JSON Lines, YAML, and points files; plus a dict-like LMDB wrapper.
- **Image Processing** (`wtools.utils.imgproc`) -- Fast image size extraction (no PIL needed), safe cropping with zero-padding, and byte-string <-> NumPy array conversion.
- **Visualization** (`wtools.utils.visualization`) -- Draw bounding boxes and keypoints on images, and arrange multiple images into a grid canvas.
- **Utilities** (`wtools.utils.utils`) -- Multi-process memory monitoring and Jupyter notebook detection.
- **Head Pose** (`wtools.landmark.calculate_pose`) -- Estimate pitch / yaw / roll Euler angles from 2-D facial landmarks via PnP.
- **Video** (`wtools.utils.video`) -- JPEGBIN1 binary container for pre-extracted JPEG-compressed video frames; includes read, write, and video-to-bin conversion with hardware acceleration support.
- **CLI Tools** (`tools/`) -- Batch generate head-pose annotations from landmark files.

## Usage Examples

### File I/O (`wtools.utils.io`)

```python
from wtools.utils.io import load_json, dump_json, load_pickle, load_yaml, LMDB

# Load and dump JSON
config = load_json("config.json")
dump_json({"epochs": 100}, "output.json")

# Load YAML
params = load_yaml("params.yaml")

# Dict-like LMDB key-value store
with LMDB("/tmp/mydb", flag="c") as db:
    db["key1"] = b"value1"
    print(db["key1"])        # b'value1'
    print(len(db))           # 1
    for k, v in db.items():
        print(k, v)
```

### Image Processing (`wtools.utils.imgproc`)

```python
import numpy as np
from wtools.utils.imgproc import get_image_size, safe_crop, str2img, img2str

# Get image dimensions without decoding the full image
width, height = get_image_size("photo.jpg")

# Safe crop with automatic zero-padding for out-of-bounds regions
img = np.ones((100, 100, 3), dtype=np.uint8) * 128
cropped = safe_crop(img, [80, 80, 50, 50])  # shape: (50, 50, 3)

# Convert between byte strings and NumPy arrays
with open("photo.jpg", "rb") as f:
    raw = f.read()
img_array = str2img(raw)
raw_bytes = img2str(img_array)
```

### Visualization (`wtools.utils.visualization`)

```python
import cv2
from wtools.utils.visualization import draw_bbox, draw_keypoints, display_image_grid

# Draw a bounding box with a label
img = cv2.imread("photo.jpg")
draw_bbox(img, [10, 20, 200, 300], (0, 255, 0), text="face")
cv2.imwrite("annotated.jpg", img)

# Draw keypoints with indices
kpts = [(30, 40), (60, 40), (45, 70)]
annotated = draw_keypoints(img, kpts, use_index=True)

# Arrange multiple images into a grid canvas
canvas = display_image_grid(["img1.jpg", "img2.jpg", "img3.jpg"], cols=3)
cv2.imwrite("grid.jpg", canvas)
```

### Utilities (`wtools.utils.utils`)

```python
from wtools.utils.utils import MemoryMonitor, isnotebook

# Monitor memory of the current process (and optionally child processes)
monitor = MemoryMonitor()
print(monitor.table())   # formatted table
print(monitor.str())     # one-line summary

# Detect Jupyter notebook environment
if isnotebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm
```

### Head Pose Estimation (`wtools.landmark.calculate_pose`)

```python
import numpy as np
from wtools.landmark.calculate_pose import calculate_pitch_yaw_roll

# 14 facial landmarks in the canonical order:
# [left-eyebrow-left, left-eyebrow-right,
#  right-eyebrow-left, right-eyebrow-right,
#  left-eye-left, left-eye-right,
#  right-eye-left, right-eye-right,
#  nose-left, nose-right,
#  mouth-left, mouth-right,
#  lower-lip, chin]
landmarks_2D = np.array([...], dtype=np.float32)  # shape: (14, 2)
pitch, yaw, roll = calculate_pitch_yaw_roll(landmarks_2D)
print(f"pitch={pitch:.1f}, yaw={yaw:.1f}, roll={roll:.1f}")
```

For dlib 68-point landmarks, select the 14 points with these indices:
```python
TRACKED_POINTS = [17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8]
landmarks_2D = landmarks_68[TRACKED_POINTS]
```

For WFLW 98-point landmarks:
```python
TRACKED_POINTS = [33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16]
landmarks_2D = landmarks_98[TRACKED_POINTS]
```

### Video (`wtools.utils.video`)

```python
from wtools.utils.video import write_jpeg_bin, read_jpeg_bin, video_to_jpeg_bin, read_jpeg_bin_metadata

# Convert a video to JPEGBIN1 format
meta = video_to_jpeg_bin("input.mp4", "output.bin", sample_fps=2.0, max_size=448)

# Read metadata without decoding frames
meta = read_jpeg_bin_metadata("output.bin")
print(f"{meta['nframes']} frames, {meta['width']}x{meta['height']}")

# Read all frames as numpy array
video, meta, fps = read_jpeg_bin("output.bin")

# Uniformly sample 16 frames
video, meta, fps = read_jpeg_bin("output.bin", num_frames=16)
```

### CLI Tool: `gen_pose.py`

Batch-generate head poses from a list of landmark files and produce a
pose-balanced resampled list:

```bash
# Default (SenseTime 106-point format)
python tools/gen_pose.py /path/to/img_pts_list.txt

# Specify root directory and landmark format
python tools/gen_pose.py /path/to/img_pts_list.txt --root_dir /data --pts_format mmc
```

The input list file should contain one entry per line, each with an image
path and a `.pts` file path separated by whitespace:

```
img/0001.jpg pts/0001.pts
img/0002.jpg pts/0002.pts
```

Outputs (written to `root_dir`):
- `<img_pts_list_path>.json` -- pose dict for every entry.
- `img_pts_list_half_large_pose.txt` -- resampled, pose-balanced list.

### CLI Tool: `video_to_bin.py`

Convert video files to JPEGBIN1 (.bin) format:

```bash
video-to-bin input.mp4
video-to-bin input.mp4 -o output.bin --sample-fps 4.0 --max-size 448
video-to-bin input.mp4 --hwaccel cuda
```

## Project Structure

```
wtools/
├── wtools/                        # Main package
│   ├── __init__.py
│   ├── utils/                     # Utility modules
│   │   ├── __init__.py
│   │   ├── io.py                  # File I/O helpers and LMDB wrapper
│   │   ├── imgproc.py             # Image processing utilities
│   │   ├── utils.py               # MemoryMonitor and isnotebook
│   │   ├── video.py               # JPEGBIN1 binary video container
│   │   └── visualization.py       # Drawing and grid display helpers
│   └── landmark/                  # Landmark-related modules
│       ├── __init__.py
│       └── calculate_pose.py      # Head pose estimation from 2-D landmarks
├── tools/                         # CLI tools
│   ├── __init__.py
│   ├── gen_pose.py                # Batch pose generation CLI
│   └── video_to_bin.py            # Video-to-JPEGBIN1 conversion CLI
├── setup.py                       # Package setup script
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── .gitignore
```

## License

MIT
