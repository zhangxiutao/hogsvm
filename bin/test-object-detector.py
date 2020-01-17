#!/usr/bin/python
import os
import shutil
# Link to the UIUC Car Database
# http://l2r.cs.uiuc.edu/~cogcomp/Data/Car/CarData.tar.gz
# dataset_url = "http://l2r.cs.uiuc.edu/~cogcomp/Data/Car/CarData.tar.gz"
# dataset_path = "../data/dataset/CarData.tar.gz"

#Fetch and extract the dataset
# if not os.path.exists(dataset_path):
#     os.system("wget {} -O {}".format(dataset_url, dataset_path))
#     os.system("tar -xvzf {} -C {}".format(dataset_path, os.path.split(dataset_path)[0]))

mod = "1"

if mod == "Train":
    pos_path = "../data/dataset/BaustelleData/TrainingImages/pos"
    neg_path = "../data/dataset/BaustelleData/TrainingImages/neg"
    pos_feat_path =  "../data/features/pos"
    neg_feat_path =  "../data/features/neg"
    shutil.rmtree(pos_feat_path)
    shutil.rmtree(neg_feat_path)
    os.mkdir(pos_feat_path)
    os.mkdir(neg_feat_path)
    os.system("python ../object-detector/extract-features.py -p {} -n {}".format(pos_path, neg_path))
    os.system("python ../object-detector/train-classifier.py -p {} -n {}".format(pos_feat_path, neg_feat_path))
else:
    test_im_path = "../data/dataset/BaustelleData/TestImages/baustelle_10.jpg"
    os.system("python ../object-detector/test-classifier.py -i {} -d {}".format(test_im_path,3))
