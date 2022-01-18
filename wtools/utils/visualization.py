#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import cv2
from matplotlib import pyplot as plt

def display_image_grid(images_filepaths, labels=(), cols=5, label_color="green"):
    rows = len(images_filepaths) // cols
    figure, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(12, 6))
    for i, image_filepath in enumerate(images_filepaths):
        image = cv2.imread(image_filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        color = label_color
        label = labels[i]
        ax.ravel()[i].imshow(image)
        ax.ravel()[i].set_title(label, color=color)
        ax.ravel()[i].set_axis_off()
    plt.tight_layout()
    plt.show()


def vis_keypoints(image, keypoints, color=KEYPOINT_COLOR, diameter=15):
    image = image.copy()

    for (x, y) in keypoints:
        cv2.circle(image, (int(x), int(y)), diameter, (0, 255, 0), -1)

    plt.figure(figsize=(8, 8))
    plt.axis('off')
    plt.imshow(image)
