from typing import *
import os
import cv2


class View:
    OUTPUT_PATH = os.path.join(os.getcwd(), '..', 'output')

    def write_image(self, output_filename: str, segmented_matrix: List):
        """Given image matrix, writes an img file
        """
        img_path = os.path.join(self.OUTPUT_PATH, output_filename)
        cv2.imwrite(img_path, segmented_matrix)
        width = len(segmented_matrix[0])
        height = len(segmented_matrix)
        print(output_filename, ' : ', width, 'x', height, ', ' , width *
              height, ' pixels ', sep='')
