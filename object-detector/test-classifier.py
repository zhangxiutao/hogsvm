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

def adjust_gamma(image, gamma=1.0):

   invGamma = 1.0 / gamma
   table = np.array([((i / 255.0) ** invGamma) * 255
      for i in np.arange(0, 256)]).astype("uint8")

   return cv2.LUT(image, table)

def increase_brightness(img, value=30):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    return img

def red_mask(img_cv2):

    img_hsv = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2HSV)
    #red mask0
    lower_red = np.array([0, 43, 46])
    upper_red = np.array([10, 255, 255])
    mask0 = cv2.inRange(img_hsv,lower_red,upper_red)
    #red mask1
    lower_red = np.array([156, 43, 46])
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(img_hsv,lower_red,upper_red)

    mask_red = mask0 + mask1
    res_red = cv2.bitwise_and(img_cv2, img_cv2, mask=mask_red)
    
    #erosion
    kernel = np.ones((5,5),np.uint8)
    res_red = cv2.erode(res_red,kernel)
    h,s,red_gray=cv2.split(res_red)
    #res_red = cv2.cvtColor(cv2.cvtColor(res_red, cv2.COLOR_HSV2BGR), cv2.COLOR_RGB2GRAY)
    cv2.imshow("",red_gray)
    cv2.waitKey(0)
    return red_gray

def cv22PIL(img_cv2):

    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_cv2)

    return img_pil

def PIL2cv2(img_pil):

    img_cv2 = np.array(img_pil.convert('RGB'))[:, :, ::-1].copy() 

    return img_cv2

def random_window(img_origin_pil, img_cv2, window_size):
    nonzeros = cv2.findNonZero(img_cv2)
    windows_pil = []
    while True:
        anchor_point = random.choice(nonzeros)[0]
        mid_p1 = (anchor_point[0]-math.floor(window_size[0]/2),anchor_point[1]-math.floor(window_size[1]/2))
        mid_p2 = (anchor_point[0]+math.ceil(window_size[0]/2),anchor_point[1]+math.ceil(window_size[1]/2))
        if mid_p1[0] > 0 and mid_p1[1] > 0 and mid_p2[0] < img_origin_pil.size[0] and mid_p2[1] < img_origin_pil.size[1]:
            break
    
    if mid_p1[1]+int(window_size[1]/2) < img_origin_pil.size[1]:
        lower_window_pil = img_origin_pil.crop((mid_p1[0],mid_p1[1]+int(window_size[1]/2),mid_p2[0],mid_p2[1]+int(window_size[1]/2)))
    else:
        lower_window_pil = None

    mid_window_pil = img_origin_pil.crop((mid_p1[0],mid_p1[1],mid_p2[0],mid_p2[1]))
    
    if mid_p1[1]-int(window_size[1]/2) > 0:
        upper_window_pil = img_origin_pil.crop((mid_p1[0],mid_p1[1]-int(window_size[1]/2),mid_p2[0],mid_p2[1]-int(window_size[1]/2)))
    else:
        upper_window_pil = None
    
    windows_pil.append(lower_window_pil)
    windows_pil.append(mid_window_pil)
    windows_pil.append(upper_window_pil)

    return (mid_p1[0],mid_p1[1],windows_pil)

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
    img_origin_pil = img_origin_pil.crop((0,int(img_origin_pil.size[1]/2),img_origin_pil.size[0],img_origin_pil.size[1]))
    img_origin_cv2 = PIL2cv2(img_origin_pil)

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

    img_red_mask = red_mask(img_origin_cv2)

    for i in xrange(100):
  
        (x,y,windows_pil) = random_window(img_origin_pil,img_red_mask,(min_wdw_sz[0],min_wdw_sz[1]))

       
        for window_pil in windows_pil:

            if window_pil:
                
                img_window_cv2 = np.array(window_pil.convert('RGB'))[:, :, ::-1].copy()
                img_window_hsv_cv2 = cv2.cvtColor(img_window_cv2, cv2.COLOR_BGR2HSV)
                h,s,v = cv2.split(img_window_hsv_cv2)

                if np.mean(v) < 60:
                    img_window_cv2 = adjust_gamma(img_window_cv2,2)
                
                window_pil = cv22PIL(img_window_cv2)

                #window_pil.show()

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

    detections = torch.tensor(detections).double()
    scores = torch.tensor(scores)
    detections_nms_idx = ops.nms(detections,scores,0.01)
    img1 = ImageDraw.Draw(img_origin_pil) 
    for idx in detections_nms_idx:
        # Draw the detections
        img1.rectangle(detections[idx].tolist(), fill=None, outline ="red")

    img_origin_pil.show()

    cv2.waitKey(0)