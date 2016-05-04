# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 09:55:49 2015

@author: adam

Adding basic classification to segment/merge
"""

from __future__ import print_function, division

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy import signal

from skimage import graph, data, io, segmentation, color, draw

from skimage.feature import local_binary_pattern
from skimage.segmentation import mark_boundaries, slic, felzenszwalb
from skimage.util import img_as_float,img_as_ubyte
from skimage.measure import regionprops
#from skimage.future.graph import rag_mean_color, cut_threshold, cut_normalized, draw_rag, merge_hierarchical
from skimage import future
from skimage.filters.rank import entropy
from skimage.morphology import disk


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
    image : ndarraypeakind = signal.find_peaks_cwt(red_hist, np.arange(1,40))

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

def _weight_mean_color(graph, src, dst, n):
    """
    Callback to handle merging nodes by recomputing mean color.

    The method expects that the mean color of `dst` is already computed.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    weight : float
        The absolute difference of the mean color between node `dst` and `n`.
    """

    diff = graph.node[dst]['mean color'] - graph.node[n]['mean color']
    diff = np.linalg.norm(diff)
    return diff


def merge_mean_color(graph, src, dst):
    """Callback called before merging two nodes of a mean color distance graph.

    This method computes the mean color of `dst`.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    """
    graph.node[dst]['total color'] += graph.node[src]['total color']
    graph.node[dst]['pixel count'] += graph.node[src]['pixel count']
    graph.node[dst]['mean color'] = (graph.node[dst]['total color'] /
                                     graph.node[dst]['pixel count'])
        
#########end of function declarations

#img = img_as_float(astronaut()[::2, ::2])

#img = io.imread('test_im.jpg')
#raw_image = io.imread('20121023_f13_0123.jpg')
raw_image = io.imread('../test_ims/test_im.jpg')



#extract bands for classification...
red = raw_image[:,:,0]
blue = raw_image[:,:,2]

blue_red = blue/red

##make a texture measure, LBP - at two scales
#radius = 3
#n_points = 8 * radius
#lbp_3 = local_binary_pattern(red, n_points, radius, 'uniform')
#radius = 7
#n_points = 8 * radius
#lbp_7 = local_binary_pattern(red, n_points, radius, 'uniform')
#
#multi_lbp = (lbp_3+lbp_7) /2
#
#img = np.zeros_like(raw_image)
#
#img[:,:,0] = raw_image[:,:,0]
#img[:,:,1] = multi_lbp
#img[:,:,2] = blue_red

#apriori segmentation
#segments = felzenszwalb(raw_image, scale=400, sigma=0.8, min_size=10)
segments = slic(raw_image, n_segments=100, compactness=5)

label1_rgb = mark_boundaries(raw_image, segments, (255, 255, 255))


segment_props = regionprops(segments)

#create a region adjancency graph
rag = future.graph.rag_mean_color(raw_image, segments )


out = future.graph.draw_rag(segments, rag, label1_rgb, thresh = 30)
 
imshow(out)
#merged = future.graph.ncut(segments,rag,num_cuts=10, in_place=True, max_edge=1.0)
merged = future.graph.merge_hierarchical(segments, rag, thresh=30, rag_copy=False,
                                   in_place_merge=True,
                                   merge_func=merge_mean_color,
                                   weight_func=_weight_mean_color)

out_2 = mark_boundaries(raw_image, merged, (255,255,255))

imshow(out_2)

##ok, now investigate segments! and build a classifier
class_im = np.zeros_like(raw_image, dtype='float32')

red_hist, red_bins = np.histogram(raw_image[:,:,0], bins= 255)
peakind = signal.find_peaks_cwt(red_hist, np.arange(1,40))

#some simplerules 
#lbp3_cut = np.mean(lbp_3)
br_cut = np.median(blue_red) + 2* np.std(blue_red)

#adding 10 to the brightness values is arbirtary.. 
#also need to remove hard coding for 3 peaks, since
# there are often only two...
brightness_lowthresh = peakind[0]+10;
brightness_midthresh = peakind[1]+10;
brightness_highthresh = peakind[2]-10;


for (i, segVal) in enumerate(np.unique(segments)):
    #[segments == segVal] = 255
    #med_lbps[segments==segVal] = np.median(lbp_3[segments == segVal])
    #med_rb[segments==segVal] = np.median(blue_red[segments == segVal])
    seg_lbp = np.median(lbp_3[segments == segVal])
    seg_br = np.median(blue_red[segments == segVal])
    seg_intens = np.median(red[segments == segVal])
    #this is the class decision tree...
    #open water
    if (seg_intens < brightness_lowthresh):
        class_im[segments == segVal] = 1;
    #elif (seg_intens > brightness_lowthresh and seg_intens <= brightness_highthresh 
    #and seg_lbp > lbp3_cut ):
    ##thin ice next...
    #    class_im[segments == segVal] = 2;
    #elif (seg_intens > brightness_lowthresh and seg_intens <= brightness_highthresh 
    #and seg_lbp < lbp3_cut ):
    ##brash next...
    #    class_im[segments == segVal] = 3;  
    #elif (seg_intens > brightness_highthresh):
    ##snow covered ice last
    #    class_im[segments == segVal] = 4;
    else:
       class_im[segments == segVal] = 5;

# make a color map of fixed colors
cmap1 = mpl.colors.ListedColormap(['k','b','r', 'g', 'w'])

bounds=[0,1,2,3,4,5]
norm = mpl.colors.BoundaryNorm(bounds, cmap1.N)

imshow(class_im, cmap = cmap1, norm = norm)

#plt.colorbar(img,cmap=cmap1,norm=norm,boundaries=bounds,ticks=[0,2,5])


#
##more from: https://vcansimplify.wordpress.com/2014/07/06/scikit-image-rag-introduction/
##slic segmentation
#labels1 = slic(img, n_segments=100, compactness=5)
##labels1_1 = labels1 + 1
#regions1 = regionprops(labels1)
#
#for region in regions1:
#   rag.node[region['label']]['centroid'] = region['centroid']
#
#labels1_rgb = color.label2rgb(labels1, raw_image, kind='avg')
#show_img(labels1_rgb)
#
#label1_rgb = segmentation.mark_boundaries(labels1_rgb, labels1, (255, 255, 255))
#show_img(label1_rgb)
#
## RAG graph for the first seg
#
#ent = entropy(red, disk(7))mentation
#rag = rag_mean_color(image, labels1, mode='similarity')
#
##labels1 = cut_threshold(segments_slic, g)
#edges_drawn_all = display_edges(label1_rgb, rag, 20)
#
#show_img(edges_drawn_all)
#
#labels2 = cut_normalized(labels1, rag)
#
#regions2 = regionprops(labels2)
#
#for region in regions2:
#    rag.node[region['label']]['centroid'] = region['centroid']
#
#labels2_rgb = color.label2rgb(labels2, raw_image, kind='avg')
#show_img(labels2_rgb)
#
#
#    
##edges_drawn_l2 = display_edges(labels2, rag, 50 )
#
##show_img(edges_drawn_l2)
#
##print("Felzenszwalb's number of segments: %d" % len(np.unique(segments_fz)))
#print("Slic number of segments: %d" % len(np.unique(labels1)))
##print("Quickshift number of segments: %d" % len(np.unique(segments_quick)))
#
#
#
##imshow(lbp)
#
