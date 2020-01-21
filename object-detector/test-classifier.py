# Import the required modules
from __future__ import division
import cv2
import argparse as ap
from nms import nms
from config import *
import numpy as np

import torch
import torchvision.transforms as transforms
import torch.nn
import os.path
import cnn
import matplotlib.pyplot as plt

patch_size = min_wdw_sz[0]*min_wdw_sz[1]

def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out

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
    for y in xrange(0, image.shape[0], step_size[1]):
        for x in xrange(0, image.shape[1], step_size[0]):
            if (x+window_size[0]) <= image.shape[1] and (y+window_size[1]) <= image.shape[0]:
                yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

if __name__ == "__main__":

    model = cnn.Net()
    model.double()
    model.load_state_dict(torch.load("../data/models/nn_model.pt"))
    model.eval()

    # Parse the command line arguments
    parser = ap.ArgumentParser()
    parser.add_argument('-i', "--image", help="Path to the test image", required=True)
    parser.add_argument('-d','--downscale', help="Downscale ratio", default=1.25,
            type=int)
    parser.add_argument('-v', '--visualize', help="Visualize the sliding window",
            action="store_true")
    args = vars(parser.parse_args())

    # Read the image
    img_origin = cv2.imread(args["image"])
    # normalizedImg = np.zeros((img_origin.shape[0], img_origin.shape[1]))
    # normalizedImg = cv2.normalize(img_origin, None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)
    img_normalized = im2double(img_origin)
    print(img_normalized)
    #img_origin_cropped = img_origin[int(img_origin.shape[1]/2):img_origin.shape[1], :]

    visualize_det = args['visualize']
    # List to store the detections
    detections = []
    backProjectedDetections = []
    # The current scale of the image
    scale = 0

    norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # This list contains detections at the current scale
    cd = []          
    for (x, y, im_window) in sliding_window(img_normalized, min_wdw_sz, step_size):
        data = torch.from_numpy(im_window.astype(np.double))
        data = data.permute(2,0,1)
        data = norm(data)
        data = data.unsqueeze(0)
        
        #print(data)
        output = torch.max(model(data), 1)
        print(data)
        if 1 == int(output[1]):
            print  "Detection:: Location -> ({}, {})".format(x, y)
            backProjectedDetections.append((x, y, output,
                int(min_wdw_sz[0]),
                int(min_wdw_sz[1])))
            cd.append(backProjectedDetections[-1])
        
    clone = img_origin.copy()
    # Perform Non Maxima Suppression
    backProjectedDetections = nms(backProjectedDetections, threshold)
    # Display the results after performing NMS
    for (x_tl, y_tl, _, w, h) in backProjectedDetections:
        # Draw the detections
        cv2.rectangle(clone, (x_tl, y_tl), (x_tl+w,y_tl+h), (0, 0, 0), thickness=2)
    cv2.imshow("Final Detections after applying NMS", clone)
    cv2.waitKey(0)