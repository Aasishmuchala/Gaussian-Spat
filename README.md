# 4D Gaussian Splatting Web App

A Next.js web application for transforming multi-view video into dynamic 4D Gaussian splatting reconstructions using FreeTimeGS.

## Features

- **Drag & Drop Upload** - Easy multi-view video upload interface
- **Automatic Pipeline** - Frame extraction, 3D reconstruction, and 4DGS training
- **Interactive 3D Viewer** - Real-time visualization with Three.js
- **Temporal Animation** - Scrub through time to see dynamic reconstruction

## Requirements

### System Requirements
- **GPU**: NVIDIA GPU with CUDA 11.8+ (RTX 3070 or better recommended)
- **VRAM**: 8GB minimum, 16GB+ recommended
- **RAM**: 16GB minimum
- **Storage**: 50GB+ SSD
- **OS**: Linux or Windows with Python environment

### Required Software
- **Python 3.10+** with virtual environment
- **Node.js 18+** for Next.js

### Python Dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install gsplat pycolmap opencv-python numpy scipy
pip install viser imageio[ffmpeg] tqdm tensorboard
```

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Aasishmuchala/Gaussian-Spat.git
cd Gaussian-Spat
```

### 2. Install Node.js Dependencies
```bash
npm install
```

### 3. Set Up Python Environment
```bash
# Create virtual environment
python -m venv .venv

# Activate (Linux/macOS)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Development Mode
```bash
npm run dev
# The app will be available at http://localhost:3000
```

### Production Build
```bash
npm run build
npm start
```

## How It Works

### Pipeline Overview

1. **Upload** - Drag multi-view video files (3-8 cameras recommended)
2. **Frame Extraction** - Extract synchronized frames from all video inputs
3. **3D Reconstruction** - Using OpenCV/SfM to estimate camera poses and 3D points
4. **Keyframe Processing** - Select keyframes and compute velocity estimates
5. **4DGS Training** - Train temporal Gaussian primitives with motion
6. **Viewer** - Interactive 3D/4D visualization

### Input Requirements
- Multi-view video capture from 3-8 synchronized cameras
- Videos should be at same resolution and frame rate
- Recommended: 30 FPS, 1080p+ resolution

## Project Structure

```
Gaussian-Spat/
├── app/                    # Next.js App Router pages
│   ├── api/               # API routes
│   ├── page.tsx          # Home/upload page
│   ├── processing/       # Processing status page
│   └── viewer/           # 3D viewer page
├── components/           # React components
│   ├── DropZone.tsx      # Video upload component
│   ├── FileList.tsx      # Uploaded files list
│   ├── ProcessingProgress.tsx  # Pipeline progress
│   └── Viewer3D.tsx      # Three.js 3D viewer
├── lib/
│   └── pipeline/         # Python processing scripts
│       ├── preprocessing.py    # Multi-view 3D reconstruction
│       ├── keyframe_processor.py    # Keyframe + velocity
│       ├── train_4dgs.py   # 4DGS training
│       └── main_pipeline.py       # Pipeline orchestrator
└── requirements.txt      # Python dependencies
```

## API Reference

### POST /api/upload
Upload video files.
```bash
curl -X POST -F "files=@video1.mp4" -F "files=@video2.mp4" http://localhost:3000/api/upload
```

### POST /api/process
Start the processing pipeline.
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"projectId": "uuid", "videos": [...]}' \
  http://localhost:3000/api/process
```

## License

This project is available for research and educational purposes.

## References

- [FreeTimeGsVanilla](https://github.com/OpsiClear-4DGS/FreeTimeGsVanilla) - CVPR 2025
- [gsplat](https://github.com/nerfstudio-project/gsplat) - Gaussian Splatting library
- [OpenCV](https://opencv.org/) - Computer Vision library