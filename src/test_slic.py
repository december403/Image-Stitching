import numpy as np
from cv2 import cv2
import time
from SLIC import MaskedSLIC
from mask import Mask
import pandas as pd
from skimage.segmentation import slic
from skimage import color
from skimage.future import graph 
import matplotlib.pyplot as plt
def get_labels_index(labels):
    d = labels.ravel()
    f = lambda x: np.unravel_index(x.index, labels.shape)
    return pd.Series(d).groupby(d).apply(f)


start_time = time.time()
tar_img = cv2.imread('./data/processed_image/warped_target.png')
ref_img = cv2.imread('./data/processed_image/warped_reference.png')
mask = Mask(tar_img, ref_img)
# segmentation = slic(tar_img, n_segments=9)#, mask=mask.overlap)
# g = graph.rag_mean_color(tar_img, segmentation)
# lc = graph.show_rag(segmentation, g ,tar_img)
# cbar = plt.colorbar(lc)
# numOfPixel = len(np.unique(segmentation))

maskedSLIC = MaskedSLIC(tar_img, mask.overlap, region_size=200)
tar_img[maskedSLIC.contour_mask>0] = (0,255,0)
cv2.imwrite('slic_contour.png', tar_img)
g = graph.rag_mean_color(tar_img, maskedSLIC.labels)

fig, ax = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(6, 8))

ax[0].set_title('RAG drawn with default settings')
lc = graph.show_rag(labels, g, img, ax=ax[0])
# specify the fraction of the plot area that will be used to draw the colorbar
fig.colorbar(lc, fraction=0.03, ax=ax[0])

ax[1].set_title('RAG drawn with grayscale image and viridis colormap')
lc = graph.show_rag(labels, g, img,
                    img_cmap='gray', edge_cmap='viridis', ax=ax[1])
fig.colorbar(lc, fraction=0.03, ax=ax[1])

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()



# for idx in range(maskedSLIC.numOfPixel):
#     tar_img[maskedSLIC.mask==idx] = (np.random.randint(0,256),np.random.randint(0,256),np.random.randint(0,256))
# a = maskedSLIC.idxsOfpixel
print(f'-------{time.time()-start_time:8.3f} sec------------')
# print(f'There is {numOfPixel} superpixels')

# for label in range(1,numOfPixel):
#     tar_img[segmentation== label] = (np.random.randint(0,256),np.random.randint(0,256),np.random.randint(0,256))
print(f'{time.time() - start_time} sec elapse')
# cv2.imwrite('slic_RAG.png', cbar)
# plt.imshow(cbar)
# plt.imsave('slic_RAG.png')

# print(np.any(label_mask==7724))
print(f'-------{time.time()-start_time:8.3f} sec------------')