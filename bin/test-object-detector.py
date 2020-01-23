#!/usr/bin/python
import os
import shutil
import platform


mod = "T"

if platform.system() == "Linux":
    print("Linux")
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
        test_im_path = "../data/dataset/BaustelleData/TestImages/baustelle/baustelle_14.jpg"
        os.system("python ../object-detector/test-classifier.py -i {} -d {}".format(test_im_path,3))

elif platform.system() == "Windows":
    print("Windows")
    if mod == "Train":
        pos_path = "..\\..\\data\\dataset\\BaustelleData\\TrainingImages\\pos"
        neg_path = "..\\..\\data\\dataset\\BaustelleData\\TrainingImages\\neg"
        pos_feat_path =  "..\\..\\data\\features\\pos"
        neg_feat_path =  "..\\..\\data\\features\\neg"
        shutil.rmtree(pos_feat_path)
        shutil.rmtree(neg_feat_path)
        os.mkdir(pos_feat_path)
        os.mkdir(neg_feat_path)
        os.system("python ..\\object-detector\\extract-features.py -p {} -n {} -pf {} -nf {}".format(pos_path, neg_path, pos_feat_path, neg_feat_path))
        os.system("python ..\\object-detector\\train-classifier.py -p {} -n {}".format(pos_feat_path, neg_feat_path))
    else:
        test_im_path = "..\\..\\data\\dataset\\BaustelleData\\TestImages\\baustelle2.jpg"
        os.system("python ..\\object-detector\\test-classifier.py -i {} -d {}".format(test_im_path,3))