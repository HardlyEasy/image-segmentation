from typing import *
import os
import cv2
import json


class Model:
    """An instance holds all data, there will not be a separate instance
    per image file
    """
    INPUT_PATH = os.path.join(os.getcwd(), '..', 'input')
    SETTINGS = json.load(open(os.path.join(os.getcwd(), '..',
                                           'settings.json')))

    def __init__(self):
        self.sigma = Model.SETTINGS['sigma']
        self.k = Model.SETTINGS['k']
        self.min_comp_size = Model.SETTINGS['min_component_size']

        self.filenames = []
        self.img_matrices = []
        self.edges = []
        self.forests = []
        self.segmented_matrices = []

    def create_matrices(self):
        """Populates model with img matrices
        Color order is BGR
        """
        img_matrices = []
        for input_filename in os.listdir(self.INPUT_PATH):
            img_matrices.append(self.find_matrix(input_filename))
            self.filenames.append(input_filename)
        self.img_matrices = img_matrices

    def find_matrix(self, input_filename: str) -> List:
        """Returns image matrix representation after reading image file
        """
        img_path = os.path.join(self.INPUT_PATH, input_filename)
        # pixels as BGR int, type being 'numpy.uint8'
        return cv2.imread(img_path)
