# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 16:44:26 2021

@author: rodger
"""
# 需要導入模塊: from numpy import linalg [as 別名]
# 或者: from numpy.linalg import svd [as 別名]


import os
import numpy as np
from numpy import linalg
from numpy.linalg import svd

def reduce_matrix(A, eps=None):
    if np.size(A) == 0:
        return A, 0, 0
    if np.size(A) == 1:
        return A, 1, []

    m, n = A.shape
    if m != n:
        M = np.zeros(2 * (max(A.shape), ))
        M[:m, :n] = A
    else:
        M = A

    u, s, v = svd(M)
    if eps is None:
        eps = s.max() * max(M.shape) * np.finfo(s.dtype).eps

    null_mask = (s <= eps)

    rank = sum(~null_mask)
    null_space = v[null_mask]

    u = u[~null_mask][:, ~null_mask]
    s = np.diag(s[~null_mask])
    v = v[~null_mask]
    reduced = u.dot(s.dot(v))

    return reduced, rank, null_space 

def getReducePic(img, w, h, k):    
    u, s, vh = linalg.svd(img.reshape(h,w))
    # return (u[:,0:k].dot(np.diag(s[0:k]))).dot(vh[0:k,:])
    return (u[:,k:k+1].dot(np.diag(s[k:k+1]))).dot(vh[k:k+1,:])
    
from keras.preprocessing import image as Image
import cv2
img_path = './trainData/乃_0.jpg'
img = Image.load_img(img_path, grayscale=True)
img_array = Image.img_to_array(img)

# cv2.imwrite('testimg1.jpg', img_array);

# u, s, vh = linalg.svd(img_array.reshape(50,50))
 

# for i in range(50):
#     cv2.imwrite('./result/u' + str(i), u)
#     cv2.imwrite('./result/u' + str(i), u)
#     cv2.imwrite('./result/u' + str(i), u)
    
# x = np.array([[1, 0.5], [0.5, 1]])
# u, s, vh = linalg.svd(img_array.reshape(50,50))

# u1 = u[:,0:25]
# s1 = np.diag(s[0:25])
# vh1 = vh[:,0:25]
# newP =(u.dot(np.diag(s))).dot(vh.transpose())
# newP1 = getReducePic(img_array,50,50,10)
# newP2 = getReducePic(img_array,50,50,20)
# newP3 = getReducePic(img_array,50,50,30)
# newP4 = getReducePic(img_array,50,50,40)
# cv2.imwrite('testimg1.jpg', newP1)
# cv2.imwrite('testimg2.jpg', newP2)
# cv2.imwrite('testimg3.jpg', newP3)
# cv2.imwrite('testimg4.jpg', newP4)

for i in range(50):
    newP1 = getReducePic(img_array,50,50,i)
    cv2.imwrite('testimg' + str(i) + '.jpg', newP1)


# 輸出多個
# trainFiles = os.listdir('./trainData')

# cnt = 0
# for f in trainFiles:
#     img_path = './trainData/' + f
#     img = Image.load_img(img_path, grayscale=True)
#     img_array = Image.img_to_array(img)
#     # os.mkdir('result/'+ f.replace(".jpg", '') + '/')
#     for i in range(50):
#         newP1 = getReducePic(img_array,50,50,i)
#         path = 'result/' + f.replace(".jpg", '') + '_' + str(i) + '.jpg'
#         print(path)
#         cv2.imwrite(path, newP1)
        
#     cnt = cnt + 1
#     if cnt > 50:
#         break


# # 矩陣相乘
# A = np.array([[1, 2], [3, 4]])
# B = np.array([[5, 6], [7, 8]])
# print("A, B相乘(矩陣乘法) =>\n{0}".format(A.dot(B)))
# print()
# print("A, B相對應位置相乘=>\n{0}".format(A*B))