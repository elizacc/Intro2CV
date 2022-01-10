from typing import Union

from collections import defaultdict
from itertools import combinations

import numpy as np
import cv2
from skimage.transform import rescale
from skimage.measure import label, find_contours
from skimage.filters import gaussian
from skimage.feature import match_template
from scipy.spatial.distance import cdist
import scipy.stats as st


COLORS = ('blue', 'green', 'black', 'yellow', 'red')
TRAINS2SCORE = {1: 1, 2: 2, 3: 4, 4: 7, 6: 15, 8: 21}

#img_all = np.float32(cv2.imread('/autograder/source/train/all.jpg', 0))
#img_bry = np.float32(cv2.imread('/autograder/source/train/all.jpg', 0))
img_all = np.float32(cv2.imread('train/all.jpg', 0))
img_bry = np.float32(cv2.imread('train/black_red_yellow.jpg', 0))


template = img_all[796:834,1506:1544].copy()
template_r = img_all[1680:1707,1068:1095].copy()
template_b = img_all[792:820,1647:1672].copy()
template_y = img_all[655:685,1330:1355].copy()
template_k = img_bry[1765:1790, 1775:1800]

# this function is from seminar
def get_local_centers(corr, th):
    lbl, n = label(corr >= th, connectivity=2, return_num=True)
    return np.int16([np.round(np.mean(np.argwhere(lbl == i), axis=0)) for i in range(1, n + 1)])

def plot_circles(img, points, radius=30, color=(255,255,255)):
    points = np.int16(points)[::, ::-1]
    res_img = np.int16(img.copy())
    for pt in points:
        cv2.circle(res_img, (pt[0], pt[1]), radius, color, -5)
    return res_img


def predict_image(img: np.ndarray) -> (Union[np.ndarray, list], dict, dict):
    # raise NotImplementedError
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).copy()
    corr_skimage = match_template(img_gray, template, pad_input=True)
    city_centers = np.int64(get_local_centers(corr_skimage, 0.7))
    
    img_orig = img[..., ::-1].copy()[100:-100, 100:-100]
    img_cv = np.float32(img_gray).copy()[100:-100, 100:-100]
    
    corr_skimage = match_template(img_cv, template_r, pad_input=True)
    points1 = get_local_centers(corr_skimage, 0.5)
    res = plot_circles(img_orig, points1)
    
    corr_skimage = match_template(img_cv, template_b, pad_input=True)
    points2 = get_local_centers(corr_skimage, 0.5)
    res = plot_circles(res, points2)
    
    corr_skimage = match_template(img_cv, template_y, pad_input=True)
    points3 = get_local_centers(corr_skimage, 0.5)
    res = plot_circles(res, points3)
    
    corr_skimage = match_template(img_cv, template_k, pad_input=True)
    points4 = get_local_centers(corr_skimage, 0.5)
    res = plot_circles(res, points4)
    
    res = res.astype('uint8')
    
    
    HLS = cv2.cvtColor(res, cv2.COLOR_RGB2HLS).copy()
    HUE = HLS[:, :, 0]
    LIGHT = HLS[:, :, 1]
    SAT = HLS[:, :, 2]
    
    kernel_3 = np.ones((3,3))
    kernel_5 = np.ones((5,5))
    kernel_7 = np.ones((7,7))
    kernel_9 = np.ones((9,9))
    kernel_11 = np.ones((11,11))
    kernel_17 = np.ones((17,17))
    kernel_19 = np.ones((19,19))
    kernel_27 = np.ones((27,27))
    
    #red
    mask_int = (((HUE < 5) | (HUE > 177)) & (LIGHT > 80) & (LIGHT < 150) & (SAT > 110) & (SAT < 205)).astype(np.uint8)
    mask_int_1 = cv2.morphologyEx(mask_int, cv2.MORPH_CLOSE, kernel_7)
    mask_int_1 = cv2.morphologyEx(mask_int_1, cv2.MORPH_OPEN, kernel_5)
    contours, hierarchy = cv2.findContours(mask_int_1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    red = 0
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area < 2000:
            continue
        red += area // 6000 + 1
        
    mask_int_2 = cv2.morphologyEx(mask_int, cv2.MORPH_ERODE, kernel_5)
    mask_int_2 = cv2.morphologyEx(mask_int_2, cv2.MORPH_DILATE, kernel_5)
    mask_int_2 = cv2.morphologyEx(mask_int_2, cv2.MORPH_ERODE, kernel_9)
    mask_int_2 = cv2.morphologyEx(mask_int_2, cv2.MORPH_DILATE, kernel_27)
    mask_int_2 = plot_circles(mask_int_2, points1, 45, color=0)
    mask_int_2 = plot_circles(mask_int_2, points2, 45, color=0)
    mask_int_2 = plot_circles(mask_int_2, points3, 45, color=0)
    mask_int_2 = plot_circles(mask_int_2, points4, 45, color=0).astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask_int_2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    red_score = 0
    approx_area = 8000
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area < 6000:
            continue
        elif area // approx_area + 1 <= 2:
            red_score += 2
        elif area // approx_area + 1 == 3:
            red_score += 4
        elif area // approx_area + 1 == 4 or area // approx_area + 1 == 5:
            red_score += 7
        elif area // approx_area + 1 == 6 or area // approx_area + 1 == 7:
            red_score += 15
        elif area // approx_area + 1 == 8:
            red_score += 21
    if red < 10:
        red_score = 0
        red = 0
    
    #blue
    mask_int = ((HUE < 106) & (HUE > 97) & (LIGHT < 115) & (LIGHT > 50)).astype(np.uint8)
    mask_int_1 = cv2.morphologyEx(mask_int, cv2.MORPH_CLOSE, kernel_5)
    mask_int_1 = cv2.morphologyEx(mask_int_1, cv2.MORPH_OPEN, kernel_3)
    contours, hierarchy = cv2.findContours(mask_int_1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    blue = 0
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area < 2000:
             continue
        blue += area // 6000 + 1
        
    mask_int_2 = cv2.morphologyEx(mask_int, cv2.MORPH_ERODE, kernel_5)
    mask_int_2 = cv2.morphologyEx(mask_int_2, cv2.MORPH_DILATE, kernel_5)
    mask_int_2 = cv2.morphologyEx(mask_int_2, cv2.MORPH_ERODE, kernel_9)
    mask_int_2 = cv2.morphologyEx(mask_int_2, cv2.MORPH_DILATE, kernel_27)
    mask_int_2 = plot_circles(mask_int_2, points1, 45, color=0)
    mask_int_2 = plot_circles(mask_int_2, points2, 45, color=0)
    mask_int_2 = plot_circles(mask_int_2, points3, 45, color=0)
    mask_int_2 = plot_circles(mask_int_2, points4, 45, color=0).astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask_int_2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    blue_score = 0
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area < 6000:
            continue
        elif area // approx_area + 1 <= 2:
            blue_score += 2
        elif area // approx_area + 1 == 3:
            blue_score += 4
        elif area // approx_area + 1 == 4 or area // approx_area + 1 == 5:
            blue_score += 7
        elif area // approx_area + 1 == 6 or area // approx_area + 1 == 7:
            blue_score += 15
        elif area // approx_area + 1 == 8:
            blue_score += 21
    if blue < 10:
        blue_score = 0
        blue = 0
    
    #green
    mask_int = ((HUE < 81) & (HUE > 74)).astype(np.uint8)
    mask_int_1 = cv2.morphologyEx(mask_int, cv2.MORPH_CLOSE, kernel_5)
    mask_int_1 = cv2.morphologyEx(mask_int_1, cv2.MORPH_OPEN, kernel_3)
    contours, hierarchy = cv2.findContours(mask_int_1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    green = 0
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area < 2000:
             continue
        green += area // 6000 + 1
        
    mask_int_2 = cv2.morphologyEx(mask_int, cv2.MORPH_DILATE, kernel_5)
    mask_int_2 = cv2.morphologyEx(mask_int_2, cv2.MORPH_ERODE, kernel_11)
    mask_int_2 = cv2.morphologyEx(mask_int_2, cv2.MORPH_DILATE, kernel_17)
    mask_int_2 = plot_circles(mask_int_2, points1, 45, color=0)
    mask_int_2 = plot_circles(mask_int_2, points2, 45, color=0)
    mask_int_2 = plot_circles(mask_int_2, points3, 45, color=0)
    mask_int_2 = plot_circles(mask_int_2, points4, 45, color=0).astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask_int_2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    green_score = 0
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area < 6000:
            continue
        elif area // approx_area + 1 <= 2:
            green_score += 2
        elif area // approx_area + 1 == 3:
            green_score += 4
        elif area // approx_area + 1 == 4 or area // approx_area + 1 == 5:
            green_score += 7
        elif area // approx_area + 1 == 6 or area // approx_area + 1 == 7:
            green_score += 15
        elif area // approx_area + 1 == 8:
            green_score += 21
    if green < 10:
        green_score = 0
        green = 0
    
    #yellow
    mask_int = ((HUE < 28) & (HUE > 15) & (LIGHT > 70) & (LIGHT < 170) & (SAT > 95)).astype(np.uint8)
    mask_int_1 = cv2.morphologyEx(mask_int, cv2.MORPH_CLOSE, kernel_5)
    mask_int_1 = cv2.morphologyEx(mask_int_1, cv2.MORPH_OPEN, kernel_3)
    contours, hierarchy = cv2.findContours(mask_int_1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    yellow = 0
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area < 2000:
             continue
        yellow += area // 6000 + 1
        
    mask_int_2 = cv2.morphologyEx(mask_int, cv2.MORPH_ERODE, kernel_5)
    mask_int_2 = cv2.morphologyEx(mask_int_2, cv2.MORPH_DILATE, kernel_5)
    mask_int_2 = cv2.morphologyEx(mask_int_2, cv2.MORPH_ERODE, kernel_9)
    mask_int_2 = cv2.morphologyEx(mask_int_2, cv2.MORPH_DILATE, kernel_27)
    mask_int_2 = plot_circles(mask_int_2, points1, 45, color=0)
    mask_int_2 = plot_circles(mask_int_2, points2, 45, color=0)
    mask_int_2 = plot_circles(mask_int_2, points3, 45, color=0)
    mask_int_2 = plot_circles(mask_int_2, points4, 45, color=0).astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask_int_2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    yellow_score = 0
    approx_area = 9000
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area < 6000:
            continue
        elif area // approx_area + 1 <= 2:
            yellow_score += area // approx_area + 1
        elif area // approx_area + 1 == 3:
            yellow_score += 4
        elif area // approx_area + 1 == 4 or area // approx_area + 1 == 5:
            yellow_score += 7
        elif area // approx_area + 1 == 6 or area // approx_area + 1 == 7:
            yellow_score += 15
        elif area // approx_area + 1 == 8:
            yellow_score += 21
    if yellow < 10:
        yellow_score = 0
        yellow = 0
    
    #black
    mask_int = ((LIGHT < 30) & (HUE < 180) & (SAT < 210)).astype(np.uint8)
    mask_int_1 = cv2.morphologyEx(mask_int, cv2.MORPH_CLOSE, kernel_7)
    mask_int_1 = cv2.morphologyEx(mask_int_1, cv2.MORPH_OPEN, kernel_5)
    contours, hierarchy = cv2.findContours(mask_int_1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    black = 0
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area < 3000:
             continue
        black += area // 6000 + 1
    
    mask_int_2 = cv2.morphologyEx(mask_int, cv2.MORPH_DILATE, kernel_5)
    mask_int_2 = cv2.morphologyEx(mask_int_2, cv2.MORPH_ERODE, kernel_11)
    mask_int_2 = cv2.morphologyEx(mask_int_2, cv2.MORPH_DILATE, kernel_17)
    mask_int_2 = plot_circles(mask_int_2, points1, 45, color=0)
    mask_int_2 = plot_circles(mask_int_2, points2, 45, color=0)
    mask_int_2 = plot_circles(mask_int_2, points3, 45, color=0)
    mask_int_2 = plot_circles(mask_int_2, points4, 45, color=0).astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask_int_2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    black_score = 0
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area < 6000:
            continue
        elif area // approx_area + 1 <= 2:
            black_score += area // approx_area + 1
        elif area // approx_area + 1 == 3:
            black_score += 4
        elif area // approx_area + 1 == 4 or area // approx_area + 1 == 5:
            black_score += 7
        elif area // approx_area + 1 == 6 or area // approx_area + 1 == 7:
            black_score += 15
        elif area // approx_area + 1 == 8:
            black_score += 21
    if black < 10:
        black_score = 0
        black = 0
        
    #city_centers = np.int64([[0, 0], [0, 0], [0, 0]])
    #n_trains = {'blue': 200, 'green': 200, 'black': 200, 'yellow': 200, 'red': 200}
    n_trains = {'blue': blue, 'green': green, 'black': black, 'yellow': yellow, 'red': red}
    #scores = {'blue': 60, 'green': 90, 'black': 0, 'yellow': 45, 'red': 0}
    #scores = {'blue': 200, 'green': 200, 'black': 200, 'yellow': 200, 'red': 200}
    scores = {'blue': blue_score, 'green': green_score, 'black': black_score, 'yellow': yellow_score, 'red': red_score}
    return city_centers, n_trains, scores
