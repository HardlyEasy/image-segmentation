# Demo
Input image:
![input_crosswalk_1280](input/crossing_1024.jpg?raw=true "Employee Data 
title")
Output segmented image:
![output_crosswalk_1280](output/crossing_1024.jpg?raw=true "Employee Data 
title")

There are more images in `input` and `output` folder.

# Purpose
Image segmentation partitions an image into multiple image regions for
easier analysis. Image segmentation occur in many fields, such as object 
identification for autonomous vehicles or medical imaging.

# Program Guide
This program will read all images in the `input` folder and output
segmented images into the `output` folder. The segmented images will
have its different image regions colored in a random color.

## Dependencies
Requires python libraries `cv2` and `numpy`

Use commands `pip install opencv-python` and `pip install numpy` to install.

## Usage
Run `src/Main.py` to start the program.

Edit `settings.json` to adjust constraints.

## Structure
MVC pattern used. Different controllers handle different responsibilities. 
Also used disjoint-set forest with union-by-rank heuristic.

# Acknowledgements
Implemented algorithm in paper titled "Efficient Graph-Based Segmentation" by 
Pedro Felzenszwalb & Daniel Huttenlocher.

Photos in `input` folder downloaded from pexels.com and pixabay.com. 

Sources below:
1. [crossing_1024](https://pixabay.com/photos/girl-street-crossing-road-7006644/)
2. flip_640 - [Photo by Filipe de Azevedo from Pexels](https://www.pexels.com/photo/photo-of-man-riding-skateboard-on-pedestrian-2083866/)
3. peacock_640 - [Photo by Pixabay from Pexels](https://www.pexels.com/photo/nature-bird-animal-beautiful-50683/)
