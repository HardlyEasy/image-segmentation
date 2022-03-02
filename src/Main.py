# Make sure that image files are in same directory as Project3.py
# *** EXAMPLE RUN ***
# python Main.py 0.5 500 25 sunflower.jpg sunflower_segmented.jpg
# *** COMMAND LINE FORMAT ***
# python Main.py sigma k min_comp_size input_img_fpath output_img_fpath

# Description:
# This program implements the algorithm described in paper titled
# "Efficient Graph-Based Image Segmentation" by Pedro Felzenszwalb
import os

import cv2 # pip install opencv-python
import numpy as np # pip install numpy
import random as rng
import operator

from src.Model import Model
from src.View import View
from src.Controller import RGBController, EdgeController
	
# Weight is defined as the total RGB difference between 2 pixels
def calcWeight(img_arr, x1, y1, x2, y2):
	r_diff = abs( (img_arr[y1][x1][0] - img_arr[y2][x2][0]) )
	g_diff = abs ( (img_arr[y1][x1][1]- img_arr[y2][x2][1]) )
	b_diff = abs ( (img_arr[y1][x1][2] - img_arr[y2][x2][2]) )
	weight = r_diff + g_diff + b_diff
	return weight

# Example: with an image of width 400, height 225
#	Row 0:  0, 1, ... 399         
#   Row 1:  400, 401, ... 799
#   ... 
#   Row 224: 89600, 89601, ... 89999
def createVertexNum(width, x, y):
	vertex_id = (y * width) + x
	return vertex_id

# Example:
#	( (0,0), (1,0), 2.4 )
def createEdge(img_arr, x1, y1, x2, y2):
	width = len(img_arr[0])
	vertex_num1 = createVertexNum(width, x1, y1)
	vertex_num2 = createVertexNum(width, x2, y2)
	weight = calcWeight(img_arr, x1, y1, x2, y2)
	edge = (vertex_num1, vertex_num2, weight)
	return edge

def createEdgeList(img_arr):
	edge_lst = []
	height = len(img_arr)
	width = len(img_arr[0])
	for y1 in range(0, height):
		for x1 in range(0, width):
			if x1 < (width - 1): # excludes the last column
				x2 = x1 + 1
				y2 = y1
				edge_lst.append(createEdge(img_arr, x1, y1, x2, y2)) # east
			if y1 < (height - 1): # excludes the last row
				x2 = x1
				y2 = y1 + 1
				edge_lst.append(createEdge(img_arr, x1, y1, x2, y2)) # south
			if x1 < (width - 1) and y1 < (height - 1): # exclude last column and last row
				x2 = x1 + 1
				y2 = y1 + 1
				edge_lst.append(createEdge(img_arr, x1, y1, x2, y2)) # southeast
			if x1 < (width - 1) and y1 != 0: # exclude last column and first row
				x2 = x1 + 1
				y2 = y1 - 1
				edge_lst.append(createEdge(img_arr, x1, y1, x2, y2)) # northeast
	return edge_lst

def sortEdgeList(edge_lst):
	sorted_edge_lst = sorted(edge_lst,key=operator.itemgetter(2))
	return sorted_edge_lst

def get_threshold(k, size):
	threshold = k / size
	return threshold

# p.570-572 of Introduction to Algorithms 3rd edition
# Psuedocode for disjoint-set forest with the union-by-rank heuristic
"""
make_set(x): 
	x.p = x # designate parent node of x by x.p
	x.rank = 0
union(x,y):
	link( find_set(x), find_set(y) )
link(x,y):
	if(x.rank > y.rank):
		y.p = x
	else:
		x.p = y
		if(x.rank == y.rank):
			y.rank = y.rank + 1
find_set(x):
	if(x != x.p):
		x.p = find_set(x.p)
	return x.p
"""
class Forest:
	def __init__(self, node_num):
		self.num_set = node_num
		self.parent = []
		self.rank = []
		self.size = []
		for x in range(0, node_num):
			self.parent.append(x)
		self.rank.append(0)
		self.size.append(1)
		self.rank = self.rank * node_num
		self.size = self.size * node_num
	def getSize(self, x):
		return self.size[x]
	# Same as find_set(x) in book's psuedocode
	def findSet(self, x):
		if(x != self.parent[x]):
			self.parent[x] = self.findSet(self.parent[x])
		return self.parent[x]
	def merge(self, x, y):
		# Same as union(x, y) in book's psuedocode
		x = self.findSet(x)
		y = self.findSet(y)
		# TODO: Not sure about this ERROR check
		# if x == y:
		#     print("ERROR: x == y is true, this should not happen, please debug me.")
		# Same as link(x, y) in book's psuedocode
		if x != y:
			if self.rank[x] > self.rank[y]:
				self.parent[y] = x
				self.size[x] += self.size[y]
			else:
				self.parent[x] = y
				self.size[y] += self.size[x]
				if self.rank[x] == self.rank[y]:
					self.rank[y] += 1
		self.num_set -= 1

def getRgbRandom():
	rgb_random = []
	rgb_random.append(rng.randint(0,255)) # random red
	rgb_random.append(rng.randint(0,255)) # random green
	rgb_random.append(rng.randint(0,255)) # random blue
	return rgb_random

def makeImgArr(forest, width, height):
	# Create and fill array with random RGB colors
	colors = []
	for i in range(0, (width * height) ):
		colors.append(getRgbRandom())
	# Make 2d img array, all elements intialized to 0
	shape = (height, width, 3)
	img = np.zeros(shape, dtype=np.uint8)
	#
	for y in range(0, height):
		for x in range (0, width):
			i = forest.findSet( (width * y) + x )
			img[y, x] = colors[i]
	return img

# The segmentation algorithm described in pdf
def segmentation(sorted_edge_lst, node_num, k):
	# every pixel is its own disjoint set
	forest = Forest(node_num)
	# 2d threshold array
	threshold = np.zeros(shape=node_num, dtype=float)
	for i in range (0, node_num):
		threshold[i] = get_threshold(k, 1)
	for edge in sorted_edge_lst: # smallest weights to biggest weights
		x = forest.findSet(edge[0])
		y = forest.findSet(edge[1])
		w = edge[2]
		if x != y: # component x != component y
			if w <= threshold[x] and w <= threshold[y]: # merge condition
				forest.merge(x, y) # merge the two components
				parent = forest.findSet(x)
				threshold[parent] = w + get_threshold(k, forest.getSize(parent))
	return forest

# Makes the output image cleaner
def mergeSmallComponents(forest, sorted_edge_lst, min_comp_size):
	# edge format is (vertexid1, vertexid2, weight)
	for edge in sorted_edge_lst:
		x = forest.findSet(edge[0]) # vertexid1
		y = forest.findSet(edge[1]) # vertexid2
		# Merge two components if one of them is small
		if (x != y):
			x_size = forest.getSize(x)
			y_size = forest.getSize(y)
			if (x_size < min_comp_size) or (y_size < min_comp_size):
				forest.merge(x, y)
	return forest

def main():
	# pixels as BGR int, inttype='numpy.uint8'
	# img_arr = cv2.imread(sys.argv[4])
	# img_output_fpath = sys.argv[5]

	model = Model()
	view = View()
	rgb_controller = RGBController(model, view)
	edge_controller = EdgeController(model, view)

	rgb_controller.run()
	edge_controller.run()

	print(model.edges[0])


	"""
	# Setup the RGB img array
	width = len(img_arr[0])
	height = len(img_arr)
	RGB_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB) # BGR to RGB
	RGB_arr_float = np.asarray(RGB_arr,dtype=float) # RGB float, type='numpy.float64'
	img_guassian = cv2.GaussianBlur(RGB_arr_float, (5,5), sigma)


	# Get edge list and sorted it
	edge_lst = createEdgeList(img_guassian)
	sorted_edge_lst = sortEdgeList(edge_lst)
	
	# Segmentation algorithm as described in paper
	node_num = width * height
	forest = segmentation(sorted_edge_lst, node_num, k)
	# After segmentation, merge smaller components now
	forest = mergeSmallComponents(forest, sorted_edge_lst, min_comp_size)

	segmented_img = makeImgArr(forest, width, height)
	# cv2.imwrite(img_output_fpath, segmented_img)
	print('width x height = ', width,'x',height, sep='')
	print('Total pixels in image = ',node_num,sep='')
	"""

if __name__ == "__main__":
	main()