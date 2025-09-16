# Posture Detection with MediaPipe

This project uses [MediaPipe](https://github.com/google/mediapipe), [OpenCV](https://opencv.org/), and [NumPy](https://numpy.org/) to perform posture detection and analysis from a webcam feed.

## Features
- Real-time webcam capture
- Pose landmark detection using MediaPipe
- Posture analysis (shoulder/hip alignment, tilt, etc.)
- Visualization with OpenCV drawing utilities

## Requirements

- **Python 3.10** (recommended; MediaPipe wheels for Windows are most stable on this version)
- Conda or venv to manage dependencies
- A working webcam

## Setup

1. Clone this repository or download the project files.

2. Create and activate a Python 3.10 environment (example with conda):

   ```bash
   conda create -n posture310 python=3.10 -y
   conda activate posture310
   ```

3. Install the required dependencies:

   ```bash
   pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```

4. Verify installation:

   ```bash
   python - <<'PY'
   import numpy, mediapipe, cv2
   print("numpy:", numpy.__version__)
   print("mediapipe:", mediapipe.__version__)
   print("opencv:", cv2.__version__)
   PY
   ```

   You should see version numbers printed without errors.

## Usage

Run the main script:

```bash
python posture.py
```

The script will open your webcam and start posture detection.  
Press **`q`** (or the quit key defined in the script) to exit.

## Notes

- If you see a **DLL load failed** error on Windows, ensure you have the [Microsoft Visual C++ 2015–2022 Redistributable (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe) installed.
- Avoid upgrading NumPy to version `>= 2.0` — it will break MediaPipe compatibility. Stick with the pinned versions in `requirements.txt`.
- Tested on: Windows 10 / Python 3.10 / Anaconda.

## License

This project is licensed under the [Apache 2.0 License](LICENSE) since MediaPipe is Apache 2.0.
