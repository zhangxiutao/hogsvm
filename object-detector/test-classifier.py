# Import the required modules
from __future__ import division
from skimage.transform import pyramid_gaussian
from skimage.transform import pyramid_expand
from skimage.io import imread
from skimage.feature import hog
from sklearn.externals import joblib
from skimage import util
from skimage.color import rgb2gray
import cv2
import argparse as ap
from nms import nms
from config import *
import numpy as np

patch_size = min_wdw_sz[0]*min_wdw_sz[1]
def checkIfWanted(img):

    img = util.img_as_ubyte(img)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    #red mask0
    lower_red = np.array([0, 43, 46])
    upper_red = np.array([10, 255, 255])
    mask0 = cv2.inRange(img_hsv,lower_red,upper_red)
    #red mask1
    lower_red = np.array([156, 43, 46])
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(img_hsv,lower_red,upper_red)

    mask_red = mask0 + mask1
    res_red = cv2.bitwise_and(img, img, mask=mask_red)

    #erosion
    erosionKernel = np.ones((5,5),np.uint8)
    erosion_red = cv2.erode(res_red,erosionKernel)
    bw_image_red = cv2.cvtColor(cv2.cvtColor(erosion_red, cv2.COLOR_HSV2BGR), cv2.COLOR_RGB2GRAY)

    #black mask0
    lower_black = np.array([0,0,0]) 
    upper_black = np.array([180,255,46]) 
    mask_black = cv2.inRange(img_hsv,lower_black,upper_black)
    res_black = cv2.bitwise_and(img, img, mask=mask_black)
    res_black = cv2.cvtColor(res_black, cv2.COLOR_BGR2GRAY)
    #erosion
    #erosionKernel = np.ones((5,5),np.uint8)
    #erosion_black = cv2.erode(res_black,erosionKernel)
    #bw_image_black = cv2.cvtColor(cv2.cvtColor(erosion_black, cv2.COLOR_HSV2BGR), cv2.COLOR_RGB2GRAY)


    #print(cv2.countNonZero(bw_image_red))
    #print(patch_size)

    # if cv2.countNonZero(bw_image_red)/patch_size < 0.25 or cv2.countNonZero(bw_image_red)/patch_size > 0.5:
        
    #     return False
    # else:
    #     return True
    return True

def sliding_window(image, window_size, step_size):
    '''
    This function returns a patch of the input image `image` of size equal
    to `window_size`. The first image returned top-left co-ordinates (0, 0) 
    and are increment in both x and y directions by the `step_size` supplied.
    So, the input parameters are -
    * `image` - Input Image
    * `window_size` - Size of Sliding Window
    * `step_size` - Incremented Size of Window

    The function returns a tuple -
    (x, y, im_window)
    where
    * x is the top-left x co-ordinate
    * y is the top-left y co-ordinate
    * im_window is the sliding window image
    '''
    for y in xrange(0, image.shape[0], step_size[1]):
        for x in xrange(0, image.shape[1], step_size[0]):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

if __name__ == "__main__":
    # Parse the command line arguments
    parser = ap.ArgumentParser()
    parser.add_argument('-i', "--image", help="Path to the test image", required=True)
    parser.add_argument('-d','--downscale', help="Downscale ratio", default=1.25,
            type=int)
    parser.add_argument('-v', '--visualize', help="Visualize the sliding window",
            action="store_true")
    args = vars(parser.parse_args())

    # Read the image
    img_origin_cropped = imread(args["image"])
    #img_origin_cropped = img_origin[img_origin.shape[0]/2:,:]

    # min_wdw_sz = (100, 40)
    # step_size = (10, 10)
    #downscale = args['downscale']
    downscale = 1.25
    upscale = 1.2
    visualize_det = args['visualize']

    # Load the classifier
    clf = joblib.load(model_path)

    # List to store the detections
    detections = []
    backProjectedDetections = []
    # The current scale of the image
    scale = 0
    
    # Downscale the image and iterate
    im_upscaled = pyramid_expand(img_origin_cropped, upscale=upscale)

    # This list contains detections at the current scale
    cd = []          
    for (x, y, im_window) in sliding_window(im_upscaled, min_wdw_sz, step_size):

        if checkIfWanted(im_window):
            if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
                continue
            im_window_gray = rgb2gray(im_window)
            # Calculate the HOG features
            fd = hog(im_window_gray, orientations, pixels_per_cell, cells_per_block, visualise=visualize, transform_sqrt=transform_sqrt)
            fd = [fd]
            pred = clf.predict(fd)
            if pred == 1:
                print  "Detection:: Location -> ({}, {})".format(x, y)
                print "Scale ->  {} | Confidence Score {} \n".format(-1,clf.decision_function(fd))
                backProjectedDetections.append((int(((x+(min_wdw_sz[0]/2))/(upscale)-(min_wdw_sz[0]/2))), int((y+(min_wdw_sz[1]/2))/(upscale)-(min_wdw_sz[1]/2)), clf.decision_function(fd),
                    int(min_wdw_sz[0]),
                    int(min_wdw_sz[1])))
                cd.append(backProjectedDetections[-1])
        
                   
    for im_downscaled in pyramid_gaussian(img_origin_cropped, downscale=downscale):
        # This list contains detections at the current scale
        cd = []
        # If the width or height of the scaled image is less than
        # the width or height of the window, then end the iterations.
        if im_downscaled.shape[0] < min_wdw_sz[1] or im_downscaled.shape[1] < min_wdw_sz[0]:           
            break
        for (x, y, im_window) in sliding_window(im_downscaled, min_wdw_sz, step_size):

            if checkIfWanted(im_window):
                im_window_gray = rgb2gray(im_window)
                if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
                    continue
                # Calculate the HOG features
                fd = hog(im_window_gray, orientations, pixels_per_cell, cells_per_block, visualise=visualize, transform_sqrt=transform_sqrt)
                fd = [fd]
                pred = clf.predict(fd)
                if pred == 1:
                    print  "Detection:: Location -> ({}, {})".format(x, y)
                    print "Scale ->  {} | Confidence Score {} \n".format(scale,clf.decision_function(fd))
                    backProjectedDetections.append((int(((x+(min_wdw_sz[0]/2))*(downscale**scale)-(min_wdw_sz[0]/2))), int((y+(min_wdw_sz[1]/2))*(downscale**scale)-(min_wdw_sz[1]/2)), clf.decision_function(fd),
                        int(min_wdw_sz[0]),
                        int(min_wdw_sz[1])))
                    cd.append(backProjectedDetections[-1])
                # If visualize is set to true, display the working
                # of the sliding window
                if visualize_det:
                    clone = im_downscaled.copy()
                    for x1, y1, _, _, _  in cd:
                        # Draw the detections at this scale
                        cv2.rectangle(clone, (x1, y1), (x1 + im_window.shape[1], y1 +
                            im_window.shape[0]), (0, 0, 0), thickness=2)
                    cv2.rectangle(clone, (x, y), (x + im_window.shape[1], y +
                        im_window.shape[0]), (255, 255, 255), thickness=2)
                    cv2.imshow("Sliding Window in Progress"+str(scale), clone)
                    
            # Move the the next scale
        scale+=1
    #cv2.waitKey(0)
    # Display the results before performing NMS
    clone = img_origin_cropped.copy()
    clone = cv2.cvtColor(clone, cv2.COLOR_RGB2BGR)
    # for (x_tl, y_tl, _, w, h, s) in backProjectedDetections:
    #     # Draw the detections
    #     if s == -1:
    #         cv2.rectangle(im, (x_tl, y_tl), (x_tl+w, y_tl+h), (0, 0, 0), thickness=2)
    #         cv2.imshow("Raw Detections before NMS", im)
    

    # Perform Non Maxima Suppression
    backProjectedDetections = nms(backProjectedDetections, threshold)
    
    x = 430 #confidence score 0.00858
    y = 160 #
    # Display the results after performing NMS
    for (x_tl, y_tl, _, w, h) in backProjectedDetections:
        # Draw the detections
        cv2.rectangle(clone, (x_tl, y_tl), (x_tl+w,y_tl+h), (0, 0, 0), thickness=2)
        #cv2.rectangle(clone, (int(((x+(w/2))/(upscale)-(w/2))), int((y+(h/2))/(upscale)-(h/2))),((int(((x+(w/2))/(upscale)-(w/2)))+w), (int((y+(h/2))/(upscale)-(h/2))+h)),(0,0,0), thickness=2)
    cv2.imshow("Final Detections after applying NMS", clone)
    cv2.waitKey(0)