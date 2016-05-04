# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 21:05:48 2015

@author: Adam Steer

Exploring local binary pattern texture measures
"""

from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage.color import label2rgb
from skimage import io


#a function to put labels on LBP image plots 
def overlay_labels(image, lbp, labels):
    mask = np.logical_or.reduce([lbp == each for each in labels])
    return label2rgb(mask, image=image, bg_label=0, alpha=0.5)

# a function to paint some histogram bars red
def highlight_bars(bars, indexes):
    for i in indexes:
        bars[i].set_facecolor('r')

#a little histogram function
def hist(ax, lbp):
    n_bins = lbp.max() + 1
    return ax.hist(lbp.ravel(), normed=True, bins=n_bins, range=(0, n_bins),
                   facecolor='0.5')

#plot font size
plt.rcParams['font.size'] = 9


# settings for LBP
METHOD = 'uniform'
radius = 5
n_points = 10 * radius

#load an image
img = io.imread('test_im.jpg')

#extract a band
image = img[:,:,0]

#compute LBP for the image
lbp = local_binary_pattern(image, n_points, radius, METHOD)

# plot histograms of LBP of textures
fig, (ax_img, ax_hist) = plt.subplots(nrows=2, ncols=3, figsize=(9, 6))
plt.gray()


#plot LBP distributions
titles = ('edge', 'flat', 'corner')
w = width = radius - 1
edge_labels = range(n_points // 2 - w, n_points // 2 + w + 1)
flat_labels = list(range(0, w + 1)) + list(range(n_points - w, n_points + 2))
i_14 = n_points // 4            # 1/4th of the histogram
i_34 = 3 * (n_points // 4)      # 3/4th of the histogram
corner_labels = (list(range(i_14 - w, i_14 + w + 1)) +
                 list(range(i_34 - w, i_34 + w + 1)))

label_sets = (edge_labels, flat_labels, corner_labels)

for ax, labels in zip(ax_img, label_sets):
    ax.imshow(overlay_labels(image, lbp, labels))

for ax, labels, name in zip(ax_hist, label_sets, titles):
    counts, _, bars = hist(ax, lbp)
    highlight_bars(bars, labels)
    ax.set_ylim(ymax=np.max(counts[:-1]))
    ax.set_xlim(xmax=n_points + 2)
    ax.set_title(name)

ax_hist[0].set_ylabel('Percentage')
for ax in ax_img:
    ax.axis('off')
