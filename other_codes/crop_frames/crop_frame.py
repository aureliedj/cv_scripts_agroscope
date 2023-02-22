import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from glob import glob
from tqdm import tqdm

from utils import getPoints, drawLines, drawPoints, sortCrop, compDist,  getWhiteFrame, getRedFrame

class DebugArgs():
    """Class for setting arguments directly in this python script instead of through a command line"""
    def __init__(self, indir, outdir, img_extension, filename = '', saving = True, issquare = False, plot_lines = False, plot_points = False):
        self.input_dir = indir
        self.output_dir = outdir
        self.add_square = issquare #remet l'image de sortie carrée
        self.filename = ''
        if filename == '':
            self.batch = True
        else:
            self.batch = False
        self.save_output = saving
        self.plot_lines = plot_lines
        self.img_extension = img_extension
        self.plot_points = plot_points

   ########################################################################################################################

def crop(args):
    """_summary_

    Args:
        input_dir (str): path to folder containing the images
        output_dir (str): path to the output folder to save the result
        add_square (bool): True if you wish to redimension the output image such that it is squared
        batch : 
    """

    def cropImg(img, points):
                mask = np.zeros(img.shape[0:2], dtype=np.uint8)
                cv2.fillPoly(mask, [points], (255))

                res = cv2.bitwise_and(img,img,mask = mask)
                rect = cv2.boundingRect(points) # returns (x,y, w,h) of the rect
                cropped = res[rect[1]: rect[1] + rect[3], rect[0]: rect[0] + rect[2]]

                return cropped
    
    def getLines(mask1, n, p1 = 1200, p2 = 800): #1500, 800

                #### Get lines
                lines = cv2.HoughLinesP(mask1.astype('uint8'), 1, np.pi/180, p1, minLineLength = p2, maxLineGap=800) #1, pi/180, 2500, 1200, 800
                out = lines.reshape((lines.shape[0], 4))

                #### Cluster the lines in 4 groups
                kmeans = KMeans(n_clusters = n)
                kmeans.fit(out)
                preds = kmeans.predict(out)
                #Keep the median of each group to represent one edge
                line_list = []
                for i in range(n):
                    l = np.median(out[preds==i], axis=0)
                    line_list.append(l)
                
                return line_list

    print('\n\n')

    paths = [args.input_dir + '/' + args.filename]
    
    list_errors = []
    
    if args.add_square:
        n_clust = 5
        n_point = 6
    else:
        n_clust = 4
        n_point = 4

    if args.batch:

        paths = glob(args.input_dir + '/*' + args.img_extension)
        
        for p in tqdm(paths):
            fn = p.split('/')[-1]
            
            img = cv2.imread(p)
            H, W = img.shape[:2]
            hsvImg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            mask1 =  getRedFrame(hsvImg, plot = False)

            if np.any(mask1):
                line_list = getLines(mask1, n_clust)

                #Check
                if len(line_list)!= n_clust:
                    print('\n Wrong number of lines found', len(line_list))

                if args.plot_lines:
                    img = cv2.imread(p)
                    drawLines(img, line_list)

                ####### Get the corresponding intersection points

                if np.any(line_list):

                    points = getPoints(H, W, line_list)
                    
                    if len(points) != n_point :
                        
                        print('\n ** Filename error', len(points), fn)
                        
                        if len(points) < n_point:
                            line_list = getLines(mask1, n_clust, 2500, 1200)
                            points = getPoints(H,W,line_list)
                            
                            if len(points) < n_point:
                                list_errors.append(fn)
                                print('\n Wrong number of points found', len(points))
                                continue

                    if args.plot_points:
                        img = cv2.imread(p)
                        drawPoints(img, points)
                        
                    if args.save_output:
                    
                        p_crop = sortCrop(points, n_point)
                        crop_img = cropImg(img, p_crop)

                        cv2.imwrite(args.output_dir + '/crop_' + fn, crop_img)
                else:
                    print('Error with ', fn)
                    list_errors.append(fn)
            else:
                print('Error with ', fn)
                list_errors.append(fn)

            
        print('Number of errors', len(list_errors))
        print(list_errors)

    ########################################################################################################################