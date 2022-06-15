#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np


def safe_crop(img, crop_box):
    crop_box = [int(c) for c in crop_box]
    cropped_img = None
    if len(img.shape) == 2:
        c_nb = 0
        cropped_img = np.zeros((crop_box[3], crop_box[2]))
    else:
        c_nb = img.shape[2]
        cropped_img = np.zeros((crop_box[3], crop_box[2], c_nb))
    x_start = int(max(crop_box[0], 0))
    x_end = int(min(crop_box[0] + crop_box[2], img.shape[1]))
    y_start = int(max(crop_box[1], 0))
    y_end = int(min(crop_box[1] + crop_box[3], img.shape[0]))
    w = x_end - x_start
    h = y_end - y_start
    if w == 0 or h == 0:
        return cropped_img
    x_s = 0 if crop_box[0] >= 0 else -crop_box[0]
    y_s = 0 if crop_box[1] >= 0 else -crop_box[1]
    x_e = x_s + w
    y_e = y_s + h
    cropped_img[y_s:y_e, x_s:x_e] = img[y_start:y_end, x_start:x_end]

    return cropped_img.astype(np.uint8)


def str2img(img_str: str) -> np.ndarray:
    img_arr = np.fromstring(img_str, np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_UNCHANGED)
    return img


def img2str(img: np.ndarray) -> str:
    img_str = cv2.imencode(".jpg", img)[1].tostring()
    return img_str
