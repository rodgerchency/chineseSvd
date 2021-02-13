# -*- coding: utf-8 -*-
"""
Created on Sat Feb 13 10:34:14 2021

@author: rodger
"""
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 12:38:56 2020

@author: rodger
"""

import numpy as np
from keras.preprocessing import image
from keras.utils import np_utils, plot_model
from numpy import loadtxt

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import random;
import cv2;

from DataLoader import DataLoader
from Util import Util

import matplotlib.pyplot as plt
import pandas as pd 

tf.reset_default_graph()

w = 50; h = 50
# w = 300; h = 300
area = w * h

loader = DataLoader(w, h)
x_train, y_train = loader.getTrain()
x_test, y_test = loader.getTest()
util = Util()
# util.show(x_train[10].reshape(50,50))
# loader.showLabel(getIndex(y_train[10]))
len_label = y_train.shape[1]

print(len_label)
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)
