#read images form input folder 
import numpy as np
import os
import cv2
from tqdm import tqdm


class ReadInput:
    
    def __init__(self):
        pass
    def read(self):
        
        path = os.getcwd()
        
        direct =  os.path.join(path, "Input")
        numberr= os.listdir(direct)
        number = int (len(numberr))
        
        img = []
        
        os.chdir(direct)
        
        for i in (range (number)):
            image = cv2.imread('Micro(%d).png'%i, 0)
            img.append(image)
            
        return img

        
        