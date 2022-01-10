import joblib

import numpy as np
from numpy import logical_and as land
from numpy import logical_not as lnot
from skimage.transform import resize
from imutils.perspective import four_point_transform
from skimage.morphology import dilation, disk
from skimage.measure import label
import cv2


def predict_image(image: np.ndarray):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    image = cv2.medianBlur(image,5)
    mask = np.zeros_like(image)
    th = cv2.adaptiveThreshold(image,np.max(image),cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,61,15)
    contours, hierarchy = cv2.findContours(th.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    areas = (cv2.contourArea(cnt) for cnt in contours)
    areas = filter(lambda x: x[0] != 0, enumerate(areas))
    cnt_id, area = max(areas, key=lambda x: x[1])
    cv2.drawContours(mask, contours, cnt_id, (1,1,1), -30)
    mask = mask.astype(int)
    
    perimeter = cv2.arcLength(contours[cnt_id], True) 
    epsilon = 0.1 * perimeter
    approx = cv2.approxPolyDP(contours[cnt_id], epsilon, True)
    
    ######
    
    if len(approx) < 4:
        approx = contours[cnt_id].reshape(-1,2)
        p0 = list(approx[np.argmin(approx[:,0])])
        p1 = list(approx[np.argmax(approx[:,0])])
        p2 = list(approx[np.argmin(approx[:,1])])
        p3 = list(approx[np.argmax(approx[:,1])])
        approx = np.array([[p0],[p1],[p2],[p3]])
    sudoku = four_point_transform(image, approx.reshape(-1,2))
    
    padd = 7
    sudoku = sudoku[padd:-padd, padd:-padd]

    sudoku_bl = cv2.medianBlur(sudoku.astype('uint8'),5)
    
    height, width = np.array(sudoku_bl.shape) // 9
    sorted_cells = []
    for i in range(9):
        for j in range(9):
            digit = sudoku_bl[i*height:(i+1)*height, j*width:(j+1)*width][padd:-padd, padd:-padd]
            mask_int = digit < 111
            mask0 = mask_int.copy()
            mask0[np.mean(mask_int, axis=1) == 1] = 0
            mask0[:, (np.mean(mask_int, axis=0) == 1)] = 0
            l = label(mask0)
            if len(np.unique(l)) > 1:
                digit_label = np.argmax([np.sum(l==i) for i in np.unique(l) if i!=0])
                mask0 = mask0 * (l==digit_label+1)
            sorted_cells.append(resize(mask0, (28,28)))
    sorted_cells = np.array(sorted_cells).reshape(9,9,28,28)
            
    sorted_cells = np.array(sorted_cells).reshape(9,9,28,28)
    
    sudoku_table = np.zeros((9,9))
    
    svm = joblib.load('/autograder/submission/model.joblib')
    #svm = joblib.load('model.joblib')
    
    for i in range(sorted_cells.shape[0]):
        for j in range(sorted_cells.shape[1]):
            if np.sum(sorted_cells[i][j]) < 31:
                sudoku_table[i][j] = -1
            else:
                pred = svm.predict(sorted_cells[i][j].reshape(1,-1))[0]
                if pred == 0:
                    pred = 8
                sudoku_table[i][j] = pred


    sudoku_digits = [sudoku_table.astype(np.int16)]

    return mask, sudoku_digits
