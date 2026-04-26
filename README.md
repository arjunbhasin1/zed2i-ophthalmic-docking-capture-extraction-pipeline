# Ophthalmic Trocar Docking Perception Pipeline

This repository contains the source code for the data acquisition, preprocessing, and integrated perception-to-control pipeline used in the BEng Research Project: *External-camera-guided trocar docking system for ophthalmic robotic surgery*.

## Folder Structure
- `capture/`: Scripts for hardware interaction, including ZED 2i camera initialization, preview, and recording.
- `preprocessing/`: Scripts for dataset preparation, including frame extraction, stereo-splitting, and metadata generation.
- `integrated_pipeline_and_training/`: The modular deployment pipeline.
    - `vision/`: Acquisition (`camera.py`) and YOLOv8n-based detection modules (`detector.py`).
    - `robot/`: Meca500 control interface (`meca500.py`) and 2D image-space alignment logic (`control.py`).
    - `models/`: Contains baseline weights (`yolov8n.pt`) and the best-performing trained model weights (`best.pt`).
    - `utils/`: Central configuration (`config.py`) for system parameters and proportional gains $K_x$ and $K_y$.

## Project Attribution
This study was a collaborative effort conducted at King's College London.
- **Arjun Bhasin**: Responsible for the primary perception pipeline, including dataset acquisition, frame extraction, cleaning, manual annotation, and the development of the 2D alignment control logic.
- **Haiyang Bian**: Responsible for model training, optimization, and software-side evaluation.

## Requirements
- OpenCV
- Numpy
- Ultralytics (YOLOv8 framework)
- ZED 2i Camera (Stereolabs)
- Meca500 Robotic System
