#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
from typing import Union, List, Tuple
from matplotlib import pyplot as plt


def display_image_grid(images, cols=5, max_num=100):
    images = images[:max_num]
    rows = len(images) // cols
    crop_size = 112
    canvas = np.zeros((rows * crop_size, cols * crop_size, 3), dtype=np.uint8)
    for i, image in enumerate(images):
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (crop_size, crop_size))
        row = i // cols
        col = i % cols
        canvas[
            row * crop_size : (row + 1) * crop_size,
            col * crop_size : (col + 1) * crop_size,
        ] = image
    return canvas


def draw_bbox(
    img: np.ndarray,
    xyxy: Union[List, Tuple, np.ndarray],
    color: Tuple,
    text: str = None,
    text_offset_wh: Tuple = (0, -7),
    box_thick: Union[int, float] = 2,
    font_scale: Union[int, float] = 1,
    font_thick: Union[int, float] = 2,
):
    """画框

    Arguments:
        img {np.ndarray} -- 输入图像
        xyxy {Union[List, Tuple, np.ndarray]} -- 框坐标
        color {Tuple} -- 框颜色

    Keyword Arguments:
        text {str} -- 框附带文本说明 (default: {None})
        text_offset_wh {Tuple} -- 文本离框左上角坐标的 offset (default: {(0, -7)})
        box_thick {Union[int, float]} -- 框粗细 (default: {2})
        font_scale {Union[int, float]} -- 字体大小 (default: {1})
        font_thick {Union[int, float]} -- 字体粗细 (default: {2})
    """
    x1, y1, x2, y2 = list(map(int, xyxy))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, box_thick)
    if text:
        cv2.putText(
            img,
            text,
            (int(x1 + text_offset_wh[0]), int(y1 + text_offset_wh[1])),
            0,
            font_scale,
            color,
            font_thick,
        )


def draw_keypoints(
    image,
    keypoints,
    color=(0, 255, 0),
    diameter=None,
    use_index=False,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=1,
    thickness=1,
    draw=False,
):
    """

    Arguments:
        image {_type_} -- _description_
        keypoints {_type_} -- _description_

    Keyword Arguments:
        color {tuple} -- _description_ (default: {(0, 255, 0)})
        diameter {_type_} -- _description_ (default: {None})
        use_index {bool} -- _description_ (default: {False})
    """
    image = image.copy()
    diameter = (max(image.shape) - 1) // 112 + 1 if diameter is None else diameter
    for i, (x, y) in enumerate(keypoints):
        x, y = int(x), int(y)
        cv2.circle(image, (x, y), diameter, color, -1)
        if use_index:
            cv2.putText(image, str(i), (x + 5, y + 5), font, font_scale, (255, 0, 0))

    if draw:
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.imshow(image)
    else:
        return image
