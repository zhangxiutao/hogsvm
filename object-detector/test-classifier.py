# Import the required modules
from __future__ import division
import cv2
import argparse as ap
from nms import nms
from config import *
import numpy as np

import math
import torch
import torchvision.transforms as transforms
import torchvision.ops as ops
import torch.nn
import os.path
import cnn
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import uuid
import time
import random

patch_size = min_wdw_sz[0]*min_wdw_sz[1]

def red_mask(img):

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
    kernel = np.ones((5,5),np.uint8)
    res_red = cv2.erode(res_red,kernel)
    res_red = cv2.cvtColor(cv2.cvtColor(res_red, cv2.COLOR_HSV2BGR), cv2.COLOR_RGB2GRAY)
    #cv2.imshow("redmask",res_red)

    return res_red

def random_window(img_origin_pil, img_cv2, window_size):
    nonzeros = cv2.findNonZero(img_cv2)

    while True:
        anchor_point = random.choice(nonzeros)[0]
        mid_p1 = (anchor_point[0]-math.floor(window_size[0]/2),anchor_point[1]-math.floor(window_size[1]/2))
        mid_p2 = (anchor_point[0]+math.ceil(window_size[0]/2),anchor_point[1]+math.ceil(window_size[1]/2))
        if mid_p1[0] > 0 and mid_p1[1] > 0 and mid_p2[0] < img_origin_pil.size[0] and mid_p2[1] < img_origin_pil.size[1]:
            break

    img_window_pil = img_origin_pil.crop((mid_p1[0],mid_p1[1],mid_p2[0],mid_p2[1]))
    return (mid_p1[0],mid_p1[1],img_window_pil)

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
    visualize_det = args['visualize']

    # Read the image
    img_origin_pil = Image.open(args["image"])
    img_origin_cv2 = np.array(img_origin_pil.convert('RGB')) 
    img_origin_cv2 = img_origin_cv2[:, :, ::-1].copy() #convert rgb to bgr

    # List to store the detections
    detections = []
    scores = []
    # The current scale of the image
    scale = 0

    toTensor = transforms.ToTensor()
    norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    # This list contains detections at the current scale
    cd = []
    count = 1          

    for i in xrange(100):

        img_red_mask = red_mask(img_origin_cv2)
        (x,y,window_pil) = random_window(img_origin_pil,img_red_mask,(min_wdw_sz[0],min_wdw_sz[1]))
        data = toTensor(window_pil)
        data = data.double()[:3,:,:]
        data = norm(data)
        data = data.unsqueeze(0)
        output = torch.max(model(data), 1)

        if 1 == int(output[1]):
            fileName = uuid.uuid4().hex+".png"
            filePath = "../data/detectedWindows/" + fileName
            print(filePath)
            window_pil.save(filePath)
            print "Detection:: Location -> ({}, {})".format(x, y)
            detections.append((x, y, x+int(min_wdw_sz[0]), y+int(min_wdw_sz[1])))
            scores.append(output[0])
            cd.append(detections[-1])
            count=count+1

    detections = torch.tensor(detections).double()
    scores = torch.tensor(scores)
    detections_nms_idx = ops.nms(detections,scores,0.2)
    img1 = ImageDraw.Draw(img_origin_pil) 
    for idx in detections_nms_idx:
        # Draw the detections
        img1.rectangle(detections[idx].tolist(), fill=None, outline ="red")

    img_origin_pil.show()

    cv2.waitKey(0)