# -*- coding: utf-8 -*-
"""
Created on Tue Feb 16 17:29:19 2021

@author: rodger
"""


import numpy as np
from numpy import linalg
from numpy.linalg import svd

from keras.preprocessing import image as Image
import cv2
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np 

# print(f"均方误差(MSE)：{mean_squared_error(result, test_y)}")
# print(f"根均方误差(RMSE)：{np.sqrt(mean_squared_error(result, test_y))}")
def getReducePic(img, w, h, k):    
    u, s, vh = linalg.svd(img.reshape(h,w))
    return (u[:,0:k].dot(np.diag(s[0:k]))).dot(vh[0:k,:])

imgs = []
for i in range(41):
    print(i)
    img_path = './trainData/上_' + str(i) + '.jpg'
    imgs.append(np.array(Image.load_img(img_path, grayscale=True)))

dis = [0]
for i in range(1, 41):
    dis.append(np.sqrt(mean_squared_error(imgs[i], imgs[0])))
    # dis.append(np.linalg.norm(imgs[0]-imgs[i], ord='fro'))


cv2.imwrite('testimg0.jpg', imgs[0])
cv2.imwrite('testimg10.jpg', imgs[10])
cv2.imwrite('testimg23.jpg', imgs[23])