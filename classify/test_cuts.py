# -*- coding: utf-8 -*-
"""
Created on Sat Apr  4 20:21:26 2015

@author: adam


"""

from skimage import graph, data, io, segmentation, color

from skimage.future.graph import rag_mean_color, cut_threshold, cut_normalized, draw_rag
from matplotlib import pyplot as plt
import os
 
#img = data.coffee()
os.system('rm *.png')
img = data.coffee()
#img = color.gray2rgb(img)
 
labels1 = segmentation.slic(img, compactness=30, n_segments=400)
out1 = color.label2rgb(labels1, img, kind='avg')
 
g = rag_mean_color(img, labels1, mode='similarity')
labels2 = cut_normalized(labels1, g)
 
offset = 1000
count = 1
tmp_labels = labels1.copy()
for g1,g2 in graph.graph_cut.sub_graph_list:
    for n,d in g1.nodes_iter(data=True):
        for l in d['labels']:
            tmp_labels[labels1 == l] = offset
    offset += 1
    for n,d in g2.nodes_iter(data=True):
        for l in d['labels']:
            tmp_labels[labels1 == l] = offset
    offset += 1       
    tmp_img = color.label2rgb(tmp_labels, img, kind='avg')
    io.imsave(str(count) + '.png',tmp_img)
    count += 1
