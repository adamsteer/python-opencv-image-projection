# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 09:55:49 2015

@author: adam
"""

from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

from skimage import graph, data, io, segmentation, color, draw

from skimage.feature import local_binary_pattern
from skimage.segmentation import mark_boundaries, slic
from skimage.util import img_as_float
from skimage.measure import regionprops
from skimage.future.graph import rag_mean_color, cut_threshold, cut_normalized


#from here: https://vcansimplify.wordpress.com/2014/07/06/scikit-image-rag-introduction/
def show_img(img):
    width = 10.0
    height = img.shape[0]*width/img.shape[1]
    f = plt.figure(figsize=(width, height))
    plt.imshow(img)
    
def display_edges(image, g, threshold):
    """Draw edges of a RAG on its image
 
    Returns a modified image with the edges drawn.Edges are drawn in green
    and nodes are drawn in yellow.
 
    Parameters
    ----------
    image : ndarray
        The image to be drawn on.
    g : RAG
        The Region Adjacency Graph.
    threshold : float
        Only edges in `g` below `threshold` are drawn.
 
    Returns:
    out: ndarray
        Image with the edges drawn.
    """
    image = image.copy()
    for edge in g.edges_iter():
        n1, n2 = edge
 
        r1, c1 = map(int, rag.node[n1]['centroid'])
        r2, c2 = map(int, rag.node[n2]['centroid'])
 
        line  = draw.line(r1, c1, r2, c2)
        circle = draw.circle(r1,c1,2)
 
        if g[n1][n2]['weight'] < threshold :
            image[line] = 0,1,0
        image[circle] = 1,1,0

    return image


#img = img_as_float(astronaut()[::2, ::2])

#img = io.imread('test_im.jpg')
img = io.imread('../test_ims/test_2_0126_subset.jpg')

redband = img[:,:,0]

radius = 9
n_points = 12 * radius

lbp = local_binary_pattern(redband, n_points, radius, 'uniform')

img[:,:,0] = lbp

#more from: https://vcansimplify.wordpress.com/2014/07/06/scikit-image-rag-introduction/
#slic segmentation
labels1 = slic(img, n_segments=100, compactness=5)
labels1 = labels1 + 1
regions1 = regionprops(labels1)

labels1_rgb = color.label2rgb(labels1, img, kind='avg')
show_img(labels1_rgb)

label1_rgb = segmentation.mark_boundaries(labels1_rgb, labels1, (0, 0, 0))
show_img(label1_rgb)

# RAG graph for the first segmentation
rag = rag_mean_color(img, labels1)
for region in regions1:
    rag.node[region['label']]['centroid'] = region['centroid']

#labels1 = cut_threshold(segments_slic, g)

edges_drawn_all = display_edges(label1_rgb, rag, 20 )

show_img(edges_drawn_all)

labels2 = cut_normalized(labels1, rag)
labels2 = labels2 + 1
#regions2 = regionprops(labels2)

labels2_rgb = color.label2rgb(labels2, img, kind='avg')
show_img(labels2_rgb)

#for region in regions2:
#    rag.node[region['label']]['centroid'] = region['centroid']
    
#edges_drawn_l2 = display_edges(labels2, rag, 50 )

#show_img(edges_drawn_l2)

#print("Felzenszwalb's number of segments: %d" % len(np.unique(segments_fz)))
print("Slic number of segments: %d" % len(np.unique(segments_slic)))
#print("Quickshift number of segments: %d" % len(np.unique(segments_quick)))



imshow(lbp)

