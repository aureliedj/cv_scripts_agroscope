import cv2
import numpy as np
import matplotlib.pyplot as plt

def infoBlur(images, show=False):
    
    output = []
    
    for img in images:
        var = cv2.Laplacian(img, cv2.CV_64F).var()
        if var <=500:
            val =True
            output.append(val) #the image is blurry
        else:
            val = False
            output.append(val)
        
        if show == True:
            print('Image is blurry '+str(val)+ ', var '+str(var))
            plt.imshow(img)
            plt.show()
    return output