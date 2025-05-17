<div align="center">

# CV Exercises

##### Computer vision exercises

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![opencv](https://img.shields.io/badge/opencv-3670A0?style=for-the-badge&logo=opencv)
</div>



This repository contains four independent computer-vision and robotics projects:

- **cannyedge.py**  
  Applies a real-time Canny edge detector to webcam video, displaying both the original and edge-filtered streams side by side.

- **correspondenceAndLocalization.py**  
  Detects 2D corner points on an image of a cube, establishes 2D↔3D correspondences with a known cube model, runs a robust ICP-style pose optimization, and overlays a synthetic 3D cube onto the original image.

- **Teleoperation/**  
  - **Python scripts** that use MediaPipe to track hand landmarks and estimate hand pose via webcam.  
  - **Processing sketch** that reads the estimated pose and teleoperates a virtual or physical robot arm in real time.

- **3DReconstructionFromVideo/**  
  Analyzes a video of walking chickens to track both camera motion and individual chicken trajectories, computes relative camera poses, and triangulates the 3D path of a selected chicken.
