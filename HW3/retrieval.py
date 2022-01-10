import cv2 as cv
import numpy as np
from skimage.feature import match_template
from skimage.measure import label, find_contours


def plot_rectangles(img, points, bbox_shape, k=2):
    points = np.int16(points)
    res_img = np.int16(img.copy())
    for pt in points:
        cv.rectangle(res_img, (pt[0] - int(bbox_shape[1] / k), pt[1] - int(bbox_shape[0] / k)),
                      (pt[0] + int(bbox_shape[1] / k), pt[1] + int(bbox_shape[0] / k)), 255, -1)
    return res_img

def predict_image(img: np.ndarray, query: np.ndarray) -> list:
    img1 = cv.cvtColor(query, cv.COLOR_BGR2GRAY).copy()
    img2 = cv.cvtColor(img, cv.COLOR_BGR2GRAY).copy()
    bboxes_list = []
    print('step 1')
    
    padd = int(img1.shape[0] / 9.4)
    cutoff = img1[padd:-padd, padd:-padd]
    print('step 2')

    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(cutoff,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des2, des1, k=2)
    good = []
    for t in matches:
        for i in range(len(t)-1):
            if t[i].distance < 0.5*t[i+1].distance:
                good.append([t[i]])
    print('step 3')
                
    list_kp2 = []

    for mat in good:
        img2_idx = mat[0].queryIdx
        p2 = kp2[img2_idx]
        list_kp2.append(p2)
    print('step 4')
    
    img3 = cv.drawKeypoints(img2, tuple(list_kp2), None, color=255)
    img3 = cv.cvtColor(img3, cv.COLOR_BGR2GRAY)
    print('step 5')
    
    list_kp2 = []

    for mat in good:
        img2_idx = mat[0].queryIdx
        (x2, y2) = kp2[img2_idx].pt
        list_kp2.append((x2, y2))
    print('step 6')
        
        
    k=5.7
    scale = 3.4
    mask = np.zeros_like(img3)
    res = plot_rectangles(mask, list_kp2, cutoff.shape, k=k)
    print('step 7')
    
    if len(np.unique(res)) > 1:
        lbl, n = label(res > 1, connectivity=2, return_num=True)
        areas = np.array([np.sum(lbl==i) for i in range(1,len(np.unique(lbl)))])
        min_area = np.max([cutoff.shape[0] * cutoff.shape[1] / scale, np.min(areas)])
        max_area = cutoff.shape[0] * cutoff.shape[1]
        for i in range(3):
            if np.sum(areas > max_area) > 0:
                mask = np.zeros_like(img3)
                res = plot_rectangles(mask, list_kp2, cutoff.shape, k=k+1)
                lbl, n = label(res > 1, connectivity=2, return_num=True)
                areas = np.array([np.sum(lbl==i) for i in range(1,len(np.unique(lbl)))])
                min_area = np.max([cutoff.shape[0] * cutoff.shape[1] / (scale+k/(k-1)), np.min(areas)])
                max_area = cutoff.shape[0] * cutoff.shape[1]
            else:
                break
            
        filt = (areas > min_area) & (areas <= max_area)
        labels_left = np.unique(lbl)[1:][filt]
        
        if len(labels_left) > 1:
            lbl = lbl * (np.isin(lbl, labels_left))
            points = np.int16([np.round(np.mean(np.argwhere(lbl == i), axis=0)) for i in labels_left])
            points = np.int16(points)[::, ::-1]
            print('step 8')
            
            for pt in points:
                xmin, ymin = (pt[0] - img1.shape[1] // 3, pt[1] - img1.shape[0] // 3)
                xmin, ymin = xmin / img2.shape[1], ymin / img2.shape[0]
                width, height = img1.shape[1] / 1.5 / img2.shape[1], img1.shape[0] / 1.5  / img2.shape[0]
                bboxes_list.append((xmin, ymin, width, height))
    print('done')
        
    return bboxes_list
