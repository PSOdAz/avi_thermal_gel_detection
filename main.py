import os
import cv2
import numpy as np
from matplotlib import pyplot as plt


def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

def template_matching(image, templates):
    
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    for template in templates :
        templ = cv2.imread(template, 0)
        w, h = templ.shape[::-1]

        res = cv2.matchTemplate(img_gray, templ, cv2.TM_CCOEFF_NORMED)
        threshold = 0.8
        loc = np.where( res >= threshold)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0,255,0), 2)

    return image

_dir = os.path.join(os.getcwd(), 'templates')

templates = listdir_fullpath(_dir)
img1 = cv2.imread(os.path.join(os.getcwd(), 'THERMAL_GAL_NAVI_0_PASS.png'))
img2 = cv2.imread(os.path.join(os.getcwd(), 'THERMAL_GAL_NAVI_0_FAIL.png'))
img3 = cv2.imread(os.path.join(os.getcwd(), 'heatsink.png'))

cv2.imwrite('res_img1.png', template_matching(img1, templates))
cv2.imwrite('res_img2.png', template_matching(img2, templates))
cv2.imwrite('res_img3.png', template_matching(img3, templates))

