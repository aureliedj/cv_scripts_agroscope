import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd

def Intersect(L1, L2):

    d1 = L1[2] - L1[0]
    d2 = L2[2] - L2[0]

    if d1== 0:
        d1 = 1
    if d2 == 0:
        d2 = 1

    a1 = (L1[3]-L1[1])/d1
    b1 = L1[1] - a1*L1[0]

    a2 = (L2[3] - L2[1])/d2
    b2 = L2[1] - a2*L2[0]

    da = a1 - a2

    if da == 0:
        da = 1

    x = (b2 - b1)/da
    y = a1*x + b1

    return x, y
    

def getPoints(H, W, line_list):

    point_list = []
    check = []

    for i in range(len(line_list)):
        for j in range(len(line_list)):
            if [i,j] not in check:
                if i!= j:
                    x, y = Intersect(line_list[i], line_list[j])
                    
                    if np.logical_and(x > 0, y > 0):
                        point_list.append(np.array([int(x),int(y)]))

                check.append([i,j])
                check.append([j,i])
    
    points = np.array(point_list)
    
    if len(points) != 0 :
    
        cond = np.logical_and(points[:,0]<W,points[:,1]<H)
        points = points[cond,:]

    return points

def sortCrop(points, n_points):
    points = points[np.argsort(points[:,0])]
    order_p = points.copy()
    
    if len(points) > 4:
        order_p = order_p[[0,1,3,4]]
        new_points = order_p.copy()
        
    else:
        new_points = order_p.copy()
    
    for i in range(int(n_points/2)):
        if i == 0:
            if order_p[i*2, 1] > order_p[i*2 + 1, 1]:
                new_points[i*2] = order_p[i*2 + 1]
                new_points[i*2 + 1] = order_p[i*2]
        else:
            if order_p[i*2, 1] < order_p[i*2 + 1, 1]:
                new_points[i*2] = order_p[i*2 + 1]
                new_points[i*2 + 1] = order_p[i*2]
    
    return new_points

def orderPoints(points):
    
    df = pd.DataFrame(points, columns = ['X', 'Y'])
    
    df.sort_values(['X', 'Y'], ascending=(False, False))
    
    return df.values
    

def drawLines(img, line_list, save = False):
   for line in line_list:
      x1, y1, x2, y2 = line.astype('uint')
      cv2.line(img, (x1, y1), (x2, y2), (255,0,0), 20)
   plt.imshow(img)
   plt.show()

   if save:
      cv2.imwrite('detectedlines.png', img)

def drawPoints(img, point_list, save = False):
   for line in point_list:
      x, y = line
      cv2.circle(img, (int(x), int(y)), radius=2, color=(255, 0, 255), thickness=50)
   plt.imshow(img)
   plt.show()

   if save:
      cv2.imwrite('detectedpoints.png', img)

def getRedFrame(hsvImg, plot = False):

    cond_h0 = np.logical_or(hsvImg[:,:,0] < 12, hsvImg[:,:,0] > 150)
    cond = np.logical_and(cond_h0, hsvImg[:,:,2] > 180) #np.logical_and(hsvImg[:,:,0] < 20, hsvImg[:,:,2] > 150) #<10, >180 hsvImg[:,:,0]>150
    new = np.zeros(hsvImg[:,:,0].shape)
    new[cond] = 255
    new = new.astype('uint8')

    if plot:
        plt.imshow(hsvImg[:,:,0]) 
        plt.show()
        plt.imshow(hsvImg[:,:,2]) 
        plt.show()
        plt.imshow(new, cmap = 'gray') 
        plt.show()

    return new 

def getWhiteFrame(hsvImg, plot = False):
    
    #cond = np.logical_or( hsvImg[:,:,0]>70, np.logical_and( hsvImg[:,:,1] < 50, hsvImg[:,:,2] > 220 ) ) 
    cond = np.logical_and( hsvImg[:,:,1] < 35, hsvImg[:,:,2] > 220 )
    
    new = np.zeros(hsvImg[:,:,0].shape)
    new[cond] = 255
    new = new.astype('uint8')

    if plot:

        plt.imshow(new, cmap = 'gray') 
        plt.show()
    return new 

def sortDist(points):
    p_order = points[np.argsort(points[:,1])]
    p = p_order.copy()

    if p_order[0,0] > p_order[1,0]:
        p_order[0] = p[1]
        p_order[1] = p[0]

    if p_order[2,0] > p_order[3,0]:
        p_order[2] = p[3]
        p_order[3] = p[2]
    
    return p_order

def compDist(img, points):

    p_order = sortDist(points)

    pts1 = np.float32(p_order)
    pts2 = np.float32([[0,0],[2750,0],[0,2750],[2750,2750]])
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img,M,(2750,2750))

    return dst