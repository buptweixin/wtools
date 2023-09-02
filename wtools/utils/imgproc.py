#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
import struct

class UnknownImageFormat(Exception):
    pass

def get_image_size(file_path):
    """
    Return (width, height) for a given img file content - no external
    dependencies except the os and struct modules from core
    """
    size = os.path.getsize(file_path)

    with open(file_path) as input:
        height = -1
        width = -1
        data = input.read(25)

        if (size >= 10) and data[:6] in ('GIF87a', 'GIF89a'):
            # GIFs
            w, h = struct.unpack("<HH", data[6:10])
            width = int(w)
            height = int(h)
        elif ((size >= 24) and data.startswith('\211PNG\r\n\032\n')
              and (data[12:16] == 'IHDR')):
            # PNGs
            w, h = struct.unpack(">LL", data[16:24])
            width = int(w)
            height = int(h)
        elif (size >= 16) and data.startswith('\211PNG\r\n\032\n'):
            # older PNGs?
            w, h = struct.unpack(">LL", data[8:16])
            width = int(w)
            height = int(h)
        elif (size >= 2) and data.startswith('\377\330'):
            # JPEG
            msg = " raised while trying to decode as JPEG."
            input.seek(0)
            input.read(2)
            b = input.read(1)
            try:
                while (b and ord(b) != 0xDA):
                    while (ord(b) != 0xFF): b = input.read(1)
                    while (ord(b) == 0xFF): b = input.read(1)
                    if (ord(b) >= 0xC0 and ord(b) <= 0xC3):
                        input.read(3)
                        h, w = struct.unpack(">HH", input.read(4))
                        break
                    else:
                        input.read(int(struct.unpack(">H", input.read(2))[0])-2)
                    b = input.read(1)
                width = int(w)
                height = int(h)
            except struct.error:
                raise UnknownImageFormat("StructError" + msg)
            except ValueError:
                raise UnknownImageFormat("ValueError" + msg)
            except Exception as e:
                raise UnknownImageFormat(e.__class__.__name__ + msg)
        else:
            raise UnknownImageFormat(
                "Sorry, don't know how to get information from this file."
            )

    return width, height

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
