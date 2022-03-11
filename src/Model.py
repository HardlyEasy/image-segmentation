from typing import *
import os
import cv2
import json


class Model:
    """Each instance holds data related to a run of the program
    #TODO:
    This way doesn't make much sense now, but allows scaling/flexibiliity
    Possible later features, multiple images, multiple cores, assignment
    Different constraint settings for different images
    """
    def __init__(self):
        input_path = os.path.join(os.getcwd(), '..', 'input')
        self.input_path = input_path
        self.settings = json.load(open(os.path.join(os.getcwd(), '..',
                                                    'settings.json')))
        images = []
        for filename in os.listdir(self.input_path):
            images.append(Image(input_path, filename))
        self.images = images


class Image:
    """Each instance holds data related to a single input image
    """
    # Instance variables
    def __init__(self, folder_path, filename):
        self.filename = filename
        self.img_matrix = cv2.imread(os.path.join(folder_path, filename))
        self.edge_list = []
        self.forest = []
        self.segmented_matrix = []

