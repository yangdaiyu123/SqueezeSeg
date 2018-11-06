#!/usr/bin/env python
#-*- coding:utf-8 -*-

# author:charles
# datetime:18-11-5 下午5:23
# software:PyCharm

import numpy as np
import math

ANGLE_PHI_MAX = 360.0
ANGLE_PHI_MIN = 0.0

ANGLE_THETA_MAX = 16.0
ANGLE_THETA_MIN = -16.0

theta_max = 0.0
theta_min = 0.0

def get_degree(x, y, z):
    
    sqrt_xy = np.sqrt(x ** 2 + y ** 2)
    # sqrt_xyz = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    
    theta = np.arctan(z / sqrt_xy) * 180 / math.pi
    
    # phi
    phi = np.arcsin(y / sqrt_xy) * 180 / math.pi
    
    
    # 调整角度
    if y > 0:
        phi = phi
    else:
        phi = phi + 180
    
    # print("degree: %f, %f" % (theta, phi))
    # 防止越界
    if phi > ANGLE_PHI_MAX:
        phi = ANGLE_PHI_MAX
    elif phi < ANGLE_PHI_MIN:
        phi = ANGLE_PHI_MIN
    
    if theta > ANGLE_THETA_MAX:
        theta = ANGLE_THETA_MAX
    elif theta < ANGLE_THETA_MIN:
        theta = ANGLE_THETA_MIN
    
    return theta, phi


def get_point(theta, phi):
    # image x(height) * y(width) 2d
    # 向下取整
    x = int((theta - ANGLE_THETA_MIN) / ((ANGLE_THETA_MAX - ANGLE_THETA_MIN) / 64))
    y = int((phi - ANGLE_PHI_MIN) / ((ANGLE_PHI_MAX - ANGLE_PHI_MIN) / 512))
    
    # 严防越界
    x = (x > 63) and 63 or x
    y = (y > 511) and 511 or y

    return x, y

def get_point_theta(x, y, z):
    theta, phi = get_degree(x, y, z)

    return get_point(theta, phi)[0]


def get_point_phi(x, y, z):
    theta, phi = get_degree(x, y, z)
    
    return get_point(theta, phi)[1]


def trainging_data(fnpy):
    
    return _tranfrom_data(fnpy)[:, :, :6]

def testing_data(fnpy):
    
    pass

def _tranfrom_data(fnpy):
    
    assert type(fnpy) == np.ndarray, "source is not a ndarray type!!!"
    data = fnpy
    
    x = [data[i][0] for i in range(len(data[:, 0]))]
    y = [data[i][1] for i in range(len(data[:, 0]))]
    z = [data[i][2] for i in range(len(data[:, 0]))]
    
    intensity = [data[i][3] for i in range(len(data[:, 0]))]
    distance = [data[i][4] for i in range(len(data[:, 0]))]
    label = [data[i][5] for i in range(len(data[:, 0]))]
    
    thetaPt = [get_point_theta(data[i][0], data[i][1], data[i][2]) for i in range(len(data[:, 0]))]  # x
    phiPt = [get_point_phi(data[i][0], data[i][1], data[i][2]) for i in range(len(data[:, 0]))]  # y
    
    # 生成数据 phi * theta * [x, y, z, i, r, c]
    image = np.zeros((64, 512, 6), dtype=np.float16)
    image_index = np.zeros((64, 512, 7), dtype=np.float16)
    
    def store_image(index):
        # print (theta[index], phi[index])
        
        image[thetaPt[index], phiPt[index], 0:3] = [x[index], y[index], z[index]]
        image[thetaPt[index], phiPt[index], 3] = intensity[index]
        image[thetaPt[index], phiPt[index], 4] = distance[index]
        image[thetaPt[index], phiPt[index], 5] = label[index]
        
        image_index[:, :, 0:6] = image
        
        xp = thetaPt[index]
        yp = phiPt[index]
        
        image_index[int(xp), int(yp), 6] = index
    
    for i in range(len(x)):
        
        if len(x) == 57888:
            pass
        
        tp = (thetaPt[i], phiPt[i])
        # print(tp)
        
        if self.isempty(image[tp[0], tp[1], 0:3]):
            store_image(i)
        elif label[i] == image[tp[0], tp[1], 5]:
            if distance[i] < image[tp[0], tp[1], 4]:
                image[tp[0], tp[1], 4] = distance[i]
        elif image[tp[0], tp[1], 5] == 0 and label[i] != 0:
            store_image(i)
        else:
            if distance[i] < image[tp[0], tp[1], 4]:
                store_image(i)
    
    return image_index