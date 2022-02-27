#this file is for changing images to etas 


import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage import measure, color, io
from tensorflow.keras.utils import to_categorical
from skimage.filters import threshold_otsu
from skimage.filters import roberts, sobel, scharr, prewitt


IMG_WIDTH = 128
IMG_HEIGHT = 128


class Etas:
    def __init__(self, image, num_regions):
        self.image = image
        self.num_regions = num_regions
        
    def create_etas(self):
        
        img = self.image
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        kernel = np.ones((3,3),np.uint8) 
        eroded = cv2.erode(thresh,kernel,iterations = 1)
        dilated = cv2.dilate(eroded,kernel,iterations = 1)
        
        # mask = dilated==255
        # s = [[1,1,1],[1,1,1],[1,1,1]]
        # labeled_mask, num_labels = ndimage.label(mask, structure=s)
        
        labeled_mask = np.zeros(dilated.shape, dtype=np.uint8)
        
        labeled_mask[dilated == 255]=1
        # labeled_mask[np.all(dilated == 0, axis = -1)]=0
        
        
        # if (labeled_mask.any ==1):
        #     labeled_mask = 0
        
        
        labels = labeled_mask
        print("Unique labels in label dataset are: ", np.unique(labeled_mask))
        n_classes = len(np.unique(labels))
        labels_cat = to_categorical(labeled_mask, num_classes=n_classes)
            
        return labels_cat 
        
        
        
        
        
# test = cv2.imread('Background (99).png',0)  


# eta = Etas(test, 2)
      
# etas = eta.create_etas()      
        
        
        