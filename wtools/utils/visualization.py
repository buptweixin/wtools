#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from matplotlib import pyplot as plt


def display_image_grid(
    images: List[Union[np.ndarray, str]],
    cols: int = 5,
    max_num: int = 100,
) -> np.ndarray:
    """Arrange a collection of images into a single grid canvas.

    Each image is resized to 112x112 pixels and placed into a grid with
    ``cols`` columns. The number of rows is determined automatically from
    the number of images displayed. If an image is given as a file path
    string, it is loaded with OpenCV and converted from BGR to RGB.

    Args:
        images: A list of images. Each element can be a NumPy array or a
            file path string.
        cols: Number of columns in the grid. Defaults to 5.
        max_num: Maximum number of images to display. Extra images are
            truncated. Defaults to 100.

    Returns:
        A ``np.uint8`` array of shape ``(rows * 112, cols * 112, 3)``
        containing all resized images arranged in a grid.

    Examples:
        >>> import cv2
        >>> imgs = [cv2.imread(f"img{i}.jpg") for i in range(12)]
        >>> canvas = display_image_grid(imgs, cols=4)
        >>> canvas.shape
        (336, 448, 3)
    """
    images = images[:max_num]
    rows = len(images) // cols
    crop_size = 112
    canvas = np.zeros((rows * crop_size, cols * crop_size, 3), dtype=np.uint8)
    for i, image in enumerate(images):
        if isinstance(image, str):
            loaded = cv2.imread(image)
            if loaded is None:
                raise FileNotFoundError(f"Could not read image: {image}")
            image = cv2.cvtColor(loaded, cv2.COLOR_BGR2RGB)
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
    text: Optional[str] = None,
    text_offset_wh: Tuple = (0, -7),
    box_thick: Union[int, float] = 2,
    font_scale: Union[int, float] = 1,
    font_thick: Union[int, float] = 2,
) -> None:
    """Draw a bounding box (with optional text) on an image in-place.

    Args:
        img: The input image as a NumPy array. Modified in-place.
        xyxy: Bounding box coordinates ``[x1, y1, x2, y2]`` where
            ``(x1, y1)`` is the top-left corner and ``(x2, y2)`` is the
            bottom-right corner.
        color: Box (and text) color as a BGR tuple, e.g. ``(0, 255, 0)``
            for green.
        text: Optional text label drawn near the top-left corner of the
            box. If ``None``, no text is drawn. Defaults to ``None``.
        text_offset_wh: Offset ``(dx, dy)`` from the box's top-left corner
            ``(x1, y1)`` for the text position. Defaults to ``(0, -7)``
            which places the text just above the box.
        box_thick: Line thickness of the rectangle in pixels. Defaults
            to 2.
        font_scale: Font scale for the optional text. Defaults to 1.
        font_thick: Font thickness for the optional text. Defaults to 2.

    Examples:
        >>> import cv2
        >>> img = cv2.imread("photo.jpg")
        >>> draw_bbox(img, [10, 20, 200, 300], (0, 255, 0), text="face")
        >>> cv2.imwrite("annotated.jpg", img)
    """
    x1, y1, x2, y2 = list(map(int, xyxy))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, int(box_thick))
    if text:
        cv2.putText(
            img,
            text,
            (int(x1 + text_offset_wh[0]), int(y1 + text_offset_wh[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            int(font_thick),
        )


def draw_keypoints(
    image: np.ndarray,
    keypoints: Union[List[Tuple[float, float]], np.ndarray],
    color: Tuple[int, int, int] = (0, 255, 0),
    font_color: Tuple[int, int, int] = (255, 0, 0),
    diameter: Optional[int] = None,
    use_index: bool = False,
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
    font_scale: float = 1,
    thickness: int = 1,
    draw: bool = False,
) -> Optional[np.ndarray]:
    """Draw keypoints (landmarks) on a copy of an image.

    Each keypoint is drawn as a filled circle. Optionally, the zero-based
    index of each keypoint can be annotated next to it. The function can
    either return the annotated image or display it inline via matplotlib.

    Args:
        image: The input image as a NumPy array. A copy is made so the
            original is not modified.
        keypoints: An iterable of ``(x, y)`` coordinate pairs specifying
            the keypoint locations in pixel coordinates.
        color: BGR color tuple for the keypoint circles. Defaults to
            ``(0, 255, 0)`` (green).
        font_color: BGR color tuple for the index labels. Defaults to
            ``(255, 0, 0)`` (blue).
        diameter: Radius of the keypoint circles in pixels. If ``None``,
            it is auto-computed as ``(max(image.shape) - 1) // 112 + 1``
            so the markers scale with image size. Defaults to ``None``.
        use_index: If ``True``, draw the zero-based index of each
            keypoint next to it. Defaults to ``False``.
        font: OpenCV font face for the index labels. Defaults to
            ``cv2.FONT_HERSHEY_SIMPLEX``.
        font_scale: Font scale for the index labels. Defaults to 1.
        thickness: Line thickness for the index labels. Defaults to 1.
        draw: If ``True``, display the annotated image using matplotlib
            and return ``None``. If ``False`` (default), return the
            annotated image array without displaying.

    Returns:
        The annotated image as a NumPy array, or ``None`` if ``draw`` is
        ``True`` (in which case the image is displayed via matplotlib).

    Examples:
        >>> import cv2
        >>> img = cv2.imread("face.jpg")
        >>> kpts = [(30, 40), (60, 40), (45, 70)]
        >>> annotated = draw_keypoints(img, kpts, use_index=True)
        >>> # Or display inline:
        >>> # draw_keypoints(img, kpts, draw=True)
    """
    image = image.copy()
    diameter = (max(image.shape) - 1) // 112 + 1 if diameter is None else diameter
    for i, (x, y) in enumerate(keypoints):
        x, y = int(x), int(y)
        cv2.circle(image, (x, y), diameter, color, -1)
        if use_index:
            cv2.putText(image, str(i), (x + 5, y + 5), font, font_scale, font_color)

    if draw:
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.imshow(image)
        return None
    else:
        return image
