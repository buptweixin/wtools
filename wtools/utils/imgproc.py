#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np


def str2img(img_str: str) -> np.ndarray:
    img_arr = np.fromstring(img_str, np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_UNCHANGED)
    return img


def img2str(img: np.ndarray) -> str:
    img_str = cv2.imencode(".jpg", img)[1].tostring()
    return img_str
