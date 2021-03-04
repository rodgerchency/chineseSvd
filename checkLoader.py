# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 22:16:58 2021

@author: rodger
"""

import numpy as np
from keras.preprocessing import image
from keras.utils import np_utils, plot_model
from numpy import loadtxt

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import random;

from DataLoader import DataLoader


import matplotlib.pyplot as plt
import pandas as pd 

tf.reset_default_graph()

len_data = 109
w = 50; h = 50
# w = 300; h = 300
area = w * h

loader = DataLoader(w, h)
loader.saveImg()