# Efficient Human Motion Reconstruction from Monocular Videos with Physical Consistency Loss

This is the code for Efficient Human Motion Reconstruction from Monocular Videos with Physical Consistency Loss

 * [Project Page](https://hitlyn.github.io/EHMR/)
 * [Paper](https://dl.acm.org/doi/10.1145/3610548.3618169)
 * [Video](https://www.youtube.com/watch?v=XWgKF8hXung)


We have refined the parameters in this repository to accommodate an alternative keypoint layout building upon our submission to Siggraph Asia 2023

## Installation
- Create a ROS (Robot Operating System) workspace
- Put all the subfolders in this repository in the src of your workspace
- Build workspace

## Usage
`rosrun tams_human_motion tams_human_motion <config file(s)> <data file>`
- You can specify multiple YAML files for configuration
- You should specify one TXT file with keypoint detections

### Example
- Load human model `roslaunch tams_human_motion loadmodel.launch`
- Start visualizer `roslaunch tams_human_motion tamsviz.launch`
- Start reconstruction and wait until it's finished \
e.g. ```rosrun tams_human_motion tams_human_motion `rospack find tams_human_motion`/config/config.yaml `rospack find tams_human_motion`/data/mediapipe/cartwheel.yaml `rospack find tams_human_motion`/data/mediapipe/cartwheel.txt``` \
- We also support the SMPL keypoints for reconstruction \
e.g. ```rosrun tams_human_motion tams_human_motion `rospack find tams_human_motion`/config/config_metrabs.yaml `rospack find tams_human_motion`/data/SMPL/dance1.yaml `rospack find tams_human_motion`/data/SMPL/dance1.txt``` \
- You can change the parameters by editing the YAML files
- Set camera height_uncertainty to 0 to accelerate the optimization
- Remember to set the frame rate from the original video
- While the optimization is running, you can watch the progress via TAMSVIZ. In TAMSVIZ, you can select different views from the Display Tree, or by clicking on them in the 3D viewport, and configure them under Properties, e.g. to show or hide force vectors, play/stop the trajectory, show the entire trajectory, etc.

## Run on your own video
Our method accommodates custom keypoint layouts, allowing you to effortlessly define the objective function within the configuration file. To extract keypoint positions from a monocular video, you can etheir:
- Utilize [Mediapipe](https://developers.google.com/mediapipe) for an efficient solution that doesn't require a GPU.
- Install [Metrabs](https://github.com/isarandi/metrabs) if you have access to a GPU machine for enhanced estimation accuracy.
To properly format the keypoint data into a text file, please adhere to the guidelines provided [here](https://github.com/TAMS-Group/efficient-motion-reconstruction/blob/master/tams_human_motion/data/readme.txt). By following these instructions, you'll ensure that your data is compatible with our motion reconstruction framework.

## Citing
If you find our work useful, please consider citing:
```BibTeX
@inproceedings{cong2023efficient,
  title={Efficient Human Motion Reconstruction from Monocular Videos with Physical Consistency Loss},
  author={Cong, Lin and Ruppel, Philipp and Wang, Yizhou and Pan, Xiang and Hendrich, Norman and Zhang, Jianwei},
  journal={SIGGRAPH Asia 2023},
  year={2023}
}
```
