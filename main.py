import os
import pathlib as Path

import cv2
import numpy as np


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

def template_matching(image, templates):
    image = cv2.imread(image, -1)
    result = image.copy()
    
    for templ in templates :
        
        templ = cv2.imread(templ, -1)
        result = cv2.matchTemplate(image, templ, cv2.TM_CCOEFF)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        top_left = max_loc
        h, w = templ.shape
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv2.rectangle(result, top_left, bottom_right,(0,0,255),4)

    cv2.imshow('1', image)
    cv2.imshow('2', result)
    cv2.waitKey(0)
    return image, result


dir = os.getcwd()
template_dir = os.path.join(dir, 'templates')
template_files = listdir_fullpath(template_dir)

image1 = os.path.join(dir, 'THERMAL_GAL_NAVI_0_PASS.png')
image2 = os.path.join(dir, 'THERMAL_GAL_NAVI_0_FAIL.png')

image1, result_image1 = template_matching(image1, template_files)
image2, result_image2 = template_matching(image2, template_files)

# Show result
imgBoth1 = np.dstack((image1, result_image1))
imgBoth.shape 
imgBoth2 = np.dstack((image2, result_image2))
cv2.imshow('Compare Result1', imgBoth1)
cv2.imshow('Compare Result2', imgBoth2)

cv2.waitKey(0)
