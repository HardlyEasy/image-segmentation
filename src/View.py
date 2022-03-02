from typing import *
import os
import cv2


class View:
    OUTPUT_PATH = os.path.join(os.getcwd(), '..', 'output')

    def prompt(self) -> Tuple[float, float, float]:
        """
        """
        sigma = float(input('Enter sigma: '))
        k = float(input('Enter k: '))
        min_component_size = float(input('Enter minimum component size: '))
        return sigma, k, min_component_size

    def write_image(self, output_filename: str, segmented_img: List):
        """Given image matrix, writes an img file
        """
        img_path = os.path.join(self.OUTPUT_PATH, output_filename)
        cv2.imwrite(img_path, segmented_img)
