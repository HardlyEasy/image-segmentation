from src.Forest import Forest
import operator
from typing import *
import cv2
import numpy as np
import random as rng


class MatrixController:
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

    def find_edge(self, img_matrix, x1, y1, x2, y2):
        """ edge defined as (vertex_id1, vertex_id2, weight)
        """
        width = len(img_matrix[0])
        vertex_id1 = self.find_vertex_id(width, x1, y1)
        vertex_id2 = self.find_vertex_id(width, x2, y2)
        weight = self.find_weight(img_matrix, x1, y1, x2, y2)
        edge = (vertex_id1, vertex_id2, weight)
        return edge

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

    def find_weight(self, img_matrix, x1, y1, x2, y2):
        """Weight is total RGB difference between 2 pixels
        """
        r_diff = abs(img_matrix[y1][x1][0] - img_matrix[y2][x2][0])
        g_diff = abs(img_matrix[y1][x1][1] - img_matrix[y2][x2][1])
        b_diff = abs(img_matrix[y1][x1][2] - img_matrix[y2][x2][2])
        weight = r_diff + g_diff + b_diff
        return weight


class SegmentController:
    def __init__(self, model, view):
        self.model = model
        self.view = view

    def run(self):
        self.make_forests()

    def make_forests(self):
        """Adds forests to model:
        1) creates forest
        2) merges smaller components in forest
        """
        for i in range(0, len(self.model.edges)):
            sorted_edge_list = self.model.edges[i]
            img_matrix = self.model.img_matrices[i]
            width = len(img_matrix[0])
            height = len(img_matrix)
            node_num = width * height
            forest = self.find_forest(sorted_edge_list, node_num)
            forest = self.merge_components(forest, sorted_edge_list)
            self.model.forests.append(forest)

    def find_threshold(self, k, size):
        threshold = k / size
        return threshold

    def find_forest(self, sorted_edge_list: List, node_num):
        # every pixel is its own disjoint set
        forest = Forest(node_num)
        # 2d threshold array
        threshold = np.zeros(shape=node_num, dtype=float)
        for i in range(0, node_num):
            threshold[i] = self.find_threshold(self.model.k, 1)
        for edge in sorted_edge_list:  # smallest weights to biggest weights
            x = forest.find_set(edge[0])
            y = forest.find_set(edge[1])
            w = edge[2]
            if x != y:  # component x != component y
                if w <= threshold[x] and w <= threshold[y]:  # merge condition
                    forest.merge(x, y)  # merge the two components
                    parent = forest.find_set(x)
                    threshold[parent] = w + self.find_threshold(
                        self.model.k, forest.get_size(parent))
        return forest

    def merge_components(self, forest, sorted_edge_list):
        """Merges smaller components together, which makes the segmented
        image clearer
        """
        for edge in sorted_edge_list:
            x = forest.find_set(edge[0])  # vertexid1
            y = forest.find_set(edge[1])  # vertexid2
            # Merge two components if one of them is small
            if x != y:
                x_size = forest.get_size(x)
                y_size = forest.get_size(y)
                if x_size < self.model.min_comp_size or \
                        y_size < self.model.min_comp_size:
                    forest.merge(x, y)
        return forest

class OutputController:
    def __init__(self, model, view):
        self.model = model
        self.view = view

    def run(self):
        self.make_segment_matrices()
        for i in range(0, len(self.model.segmented_matrices)):
            self.view.write_image(self.model.filenames[i],
                                  self.model.segmented_matrices[i])

    def make_segment_matrices(self):
        for i in range(0, len(self.model.forests)):
            forest = self.model.forests[i]
            width = len(self.model.img_matrices[i][0])
            height = len(self.model.img_matrices[i])
            matrix = self.find_segmented_matrix(forest, width, height)
            self.model.segmented_matrices.append(matrix)

    def find_segmented_matrix(self, forest, width, height):
        """Creates and fills matrix with random RGB colors
        """
        colors = []
        for i in range(0, (width * height) ):
            colors.append(self.get_rgb_random())
        # Make 2d img array, all elements initialized to 0
        shape = (height, width, 3)
        matrix = np.zeros(shape, dtype=np.uint8)
        #
        for y in range(0, height):
            for x in range(0, width):
                i = forest.find_set((width * y) + x)
                matrix[y, x] = colors[i]
        return matrix

    def get_rgb_random(self):
        rgb_random = list()
        rgb_random.append(rng.randint(0, 255))  # random red
        rgb_random.append(rng.randint(0, 255))  # random green
        rgb_random.append(rng.randint(0, 255))  # random blue
        return rgb_random
