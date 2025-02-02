# Camera Processing Application

A simple Python application that captures and processes video feed from a camera using OpenCV.

## Setup

1. Create a new directory for your project and download `HelloWorld.py` into it:
```bash
mkdir hello-world
cd hello-world
# Place HelloWorld.py in this directory
```

## Installation

First, create and activate a conda environment:
```bash
# Create a new conda environment with Python 3.11
conda create -n DT084A python=3.11

# Activate the environment
conda activate DT084A
```

Then install the required packages:
```bash
pip install opencv-python numpy matplotlib
```

## Usage

Run the application:
```bash
python HelloWorld.py
```

Press 'q' to quit the application.

## Extend the Code

The script includes a `process_frame()` function where you can add OpenCV operations:
- Image processing (filters, transformations, color conversions)
- GUI elements (trackbars, mouse callbacks)
- Core operations (matrix operations, drawing functions)

Check the [OpenCV Python documentation](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html) for more functionality.