# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 12:41:10 2020

@author: rodger
"""

import cv2
import os
import numpy as np
from keras.preprocessing import image as Image
from keras.utils import np_utils, plot_model
import random as Random

class DataLoader:

    def __init__(self, w, h):
        
        # self._w = 50; self._h = 50
        self._w = w; self._h = h
        self._area = self._w * self._h
        self._currIdx = 0
        kernel_mean = (np.array([[1, 1, 1],
                                    [1, 1, 1],
                                    [1, 1, 1]]))
        print(self._w, self._h)

        # train data
        trainFiles = os.listdir('./trainData')
        print(len(trainFiles))
        # Train DataSet
        self._trainImages=[[] for _ in range(109)]
        self._trainFileNames = [[] for _ in range(109)]
        lastName = ''
        idx = -1
        for f in trainFiles:
            img_path = './trainData/' + f
            if lastName != f[0]:
                if idx == -1:
                    idx = 0
                else:
                    idx = idx + 1                
                self._trainFileNames[idx] = f[0]
                lastName = f[0]
            # img = Image.load_img(img_path, grayscale=True)
            # img_array = Image.img_to_array(img)
            img = Image.load_img(img_path, grayscale=True)
            img_sobel = self.sobelFilter(Image.img_to_array(img) / 255)
            img_sobelMean = self.convolution(kernel_mean, img_sobel) * 255
            img_array = Image.img_to_array(img_sobelMean)
            self._trainImages[idx].append(img_array)
        
        # test data
        testFiles = os.listdir('./testData')
        print(len(testFiles))
        # Train DataSet
        self._testImages=[[] for _ in range(109)]
        self._testFileNames = [[] for _ in range(109)]
        lastName = ''
        idx = -1
        for f in testFiles:
            img_path = './testData/' + f
            if lastName != f[0]:    
                if idx == -1:
                    idx = 0
                else:
                    idx = idx + 1                            
                self._testFileNames[idx] = f[0]
                lastName = f[0]
            # img = Image.load_img(img_path, grayscale=True)
            # img_array = Image.img_to_array(img)            
            img = Image.load_img(img_path, grayscale=True)
            img_sobel = self.sobelFilter(Image.img_to_array(img) / 255)
            img_sobelMean = self.convolution(kernel_mean, img_sobel) * 255
            img_array = Image.img_to_array(img_sobelMean)
            self._testImages[idx].append(img_array)
        # 檢查資料
        # for i in range(109):            
        #     # print(self._trainFileNames[i] == self._testFileNames[i])
        #     print( i , ",",len(self._trainImages[i]) , len(self._testImages[i]))
        
    def getTrainDigits(self):
        return self._trainImages
    
    def getTestDigits(self):
        return self._testImages
    
    def getLabel(self, idx):
        return self._trainFileNames[idx]
        
    def saveImg(self):
        for i in range(10):
            cv2.imwrite('img_train' + str(i) + '.jpg', self._trainImages[i])
            cv2.imwrite('img_test' + str(i) + '.jpg', self._testImages[i])

    def convolution (self, _k, _image):

        # the weighed pixels have to be in range 0..1, so we divide by the sum of all kernel
        # values afterwards
        kernel_sum = _k.shape[0] * _k.shape[1]
        
        # fetch the dimensions for iteration over the pixels and weights
        i_width, i_height = _image.shape[0], _image.shape[1]
        k_width, k_height = _k.shape[0], _k.shape[1]
        
        # prepare the output array
        filtered = np.zeros_like(_image)
        
        # Iterate over each (x, y) pixel in the image ...
        for y in range(i_height):
            for x in range(i_width):
                weighted_pixel_sum = 0
                for ky in range(k_height):
                    for kx in range(k_width):                    
                        if y - 1 < 0 or y + 1 >= i_height or x - 1 < 0 or x + 1 >= i_width:
                            continue
                        pixel_y = y + (ky - 1)
                        pixel_x = x + (ky - 1)              
                        weighted_pixel_sum += _k[ky, kx] * _image[pixel_y, pixel_x]  
                        
                filtered[y, x] = weighted_pixel_sum / kernel_sum
        
        return filtered
        
    #
    #   z1 z2 z3
    #   z4 z5 z6
    #   z7 z8 z9
    #
    def sobelFilter (self, _image):
        i_width, i_height = _image.shape[0], _image.shape[1]
        # prepare the output array
        filtered = np.zeros_like(_image)
        for y in range(i_height):
            for x in range(i_width):
                    
                if y - 1 < 0 or y + 1 >= i_height or x - 1 < 0 or x + 1 >= i_width:
                    continue
                z1 = _image[y - 1, x - 1];
                z2 = _image[y - 1, x];
                z3 = _image[y - 1, x + 1];
                z4 = _image[y, x - 1];
                z6 = _image[y, x + 1];
                z7 = _image[y + 1, x - 1];
                z8 = _image[y + 1, x];
                z9 = _image[y + 1, x + 1];
                filtered[y, x] = abs(z7 + 2*z8 + z9 - z1 - 2*z2 - z3) + abs(z3 + 2*z6 + z9 - z1 - 2*z4 - z7)
        return filtered
