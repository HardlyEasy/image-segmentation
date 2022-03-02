import operator
from typing import *
import cv2
import numpy as np


class RGBController:
    """Responsible for preparing img matrices for use by EdgeController
    """
    def __init__(self, model, view):
        self.model = model
        self.view = view

    def run(self):
        self.model.create_matrices()
        self.prepare_matrices()

    def prepare_matrices(self):
        """Revises model img matrices to prepared versions
        """
        for i in range(0, len(self.model.img_matrices)):
            img_matrix = self.model.img_matrices[i]
            prepped_matrix = self.prepare_matrix(img_matrix)
            self.model.img_matrices[i] = prepped_matrix

    def prepare_matrix(self, img_matrix: List) -> List:
        """Prepare img matrix by:
        1) Converting BGR to RGB
        2) Convert to float, type='numpy.float64'
        3) Applying gaussian blur (removes image noise)
        """
        img_matrix = cv2.cvtColor(img_matrix, cv2.COLOR_BGR2RGB)
        img_matrix = np.asarray(img_matrix, dtype=float)
        img_matrix = cv2.GaussianBlur(img_matrix, (5, 5), self.model.sigma)
        return img_matrix


class EdgeController:
    def __init__(self, model, view):
        self.model = model
        self.view = view

    def run(self):
        self.create_edges()

    def create_edges(self):
        """
        """
        for img_matrix in self.model.img_matrices:
            edge_list = self.find_edge_list(img_matrix)
            # Sort by weight
            edge_list = sorted(edge_list, key=operator.itemgetter(2))
            self.model.edges.append(edge_list)

    def find_weight(self, img_matrix, x1, y1, x2, y2):
        """Weight is total RGB difference between 2 pixels
        """
        r_diff = abs(img_matrix[y1][x1][0] - img_matrix[y2][x2][0])
        g_diff = abs(img_matrix[y1][x1][1] - img_matrix[y2][x2][1])
        b_diff = abs(img_matrix[y1][x1][2] - img_matrix[y2][x2][2])
        weight = r_diff + g_diff + b_diff
        return weight

    def find_vertex_id(self, width, x, y):
        """Returns a unique vertex number
        eg, width 400, height 225
        Row0: 0, 1, ... 399
        Row1: 400, 401, ... 799
        ...
        Row224: 89600, 89601, ... 89999
        """
        vertex_id = (y * width) + x
        return vertex_id


    def find_edge(self, img_matrix, x1, y1, x2, y2):
        """ edge defined as (vertex_id1, vertex_id2, weight)
        """
        width = len(img_matrix[0])
        vertex_id1 = self.find_vertex_id(width, x1, y1)
        vertex_id2 = self.find_vertex_id(width, x2, y2)
        weight = self.find_weight(img_matrix, x1, y1, x2, y2)
        edge = (vertex_id1, vertex_id2, weight)
        return edge

    def find_edge_list(self, img_matrix):
        """
        """
        edge_list = []
        height = len(img_matrix)
        width = len(img_matrix[0])
        for y1 in range(0, height):
            for x1 in range(0, width):
                # excludes the last column
                if x1 < (width - 1):
                    x2 = x1 + 1
                    y2 = y1
                    # east
                    edge_list.append(self.find_edge(img_matrix,
                                                    x1, y1, x2, y2))
                # excludes the last row
                if y1 < (height - 1):
                    x2 = x1
                    y2 = y1 + 1
                    # south
                    edge_list.append(self.find_edge(img_matrix,
                                                    x1, y1, x2, y2))
                # exclude last column and last row
                if x1 < (width - 1) and y1 < (height - 1):
                    x2 = x1 + 1
                    y2 = y1 + 1
                    # southeast
                    edge_list.append(self.find_edge(img_matrix,
                                                    x1, y1, x2, y2))
                # exclude last column and first row
                if x1 < (width - 1) and y1 != 0:
                    x2 = x1 + 1
                    y2 = y1 - 1
                    # northeast
                    edge_list.append(self.find_edge(img_matrix,
                                                    x1, y1, x2, y2))
        return edge_list
