#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
import struct
from typing import List, Optional, Tuple

class UnknownImageFormat(Exception):
    """Raised when an image file format is not recognized or supported."""
    pass

def get_image_size(file_path: str) -> Tuple[int, int]:
    """Return the (width, height) of an image file without full decoding.

    Only the first few bytes of the file are read to extract the dimensions
    from the image header. Supports GIF, PNG, and JPEG formats. No external
    image libraries (e.g. PIL, OpenCV) are required -- only the standard
    ``os`` and ``struct`` modules are used.

    Args:
        file_path: Path to the image file.

    Returns:
        A tuple ``(width, height)`` of integers.

    Raises:
        UnknownImageFormat: If the file format is not recognized or an error
            occurs while parsing the header.
        FileNotFoundError: If ``file_path`` does not exist.

    Examples:
        >>> get_image_size("photo.jpg")
        (1920, 1080)
        >>> get_image_size("icon.png")
        (64, 64)
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"Image file not found: {file_path!r}"
        )
    size = os.path.getsize(file_path)

    with open(file_path, "rb") as input:
        height = -1
        width = -1
        data = input.read(25)

        if (size >= 10) and data[:6] in (b'GIF87a', b'GIF89a'):
            # GIFs
            w, h = struct.unpack("<HH", data[6:10])
            width = int(w)
            height = int(h)
        elif ((size >= 24) and data.startswith(b'\211PNG\r\n\032\n')
              and (data[12:16] == b'IHDR')):
            # PNGs
            w, h = struct.unpack(">LL", data[16:24])
            width = int(w)
            height = int(h)
        elif (size >= 16) and data.startswith(b'\211PNG\r\n\032\n'):
            # older PNGs?
            w, h = struct.unpack(">LL", data[8:16])
            width = int(w)
            height = int(h)
        elif (size >= 2) and data.startswith(b'\377\330'):
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

def safe_crop(img: np.ndarray, crop_box: List[int]) -> np.ndarray:
    """Crop an image to a bounding box, padding with zeros for out-of-bounds regions.

    Unlike a plain slice, this function handles crop boxes that extend beyond
    the image boundaries. Out-of-bounds areas are filled with zeros, so the
    output always has the exact size requested by ``crop_box``.

    Args:
        img: Input image as a NumPy array. Can be 2-D (grayscale) or 3-D
            (multi-channel).
        crop_box: A 4-element sequence ``[x, y, width, height]`` specifying
            the crop region. ``x`` and ``y`` are the top-left corner
            coordinates and may be negative.

    Returns:
        A ``np.uint8`` array of shape ``(height, width)`` for grayscale
        input or ``(height, width, channels)`` for multi-channel input,
        containing the cropped image with zero-padding where needed.

    Examples:
        >>> import numpy as np
        >>> img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        >>> cropped = safe_crop(img, [80, 80, 50, 50])
        >>> cropped.shape
        (50, 50, 3)
        >>> # Top-right corner of cropped will be zeros (out of bounds).
    """
    crop_box = [int(c) for c in crop_box]
    x_start = int(max(crop_box[0], 0))
    x_end = int(min(crop_box[0] + crop_box[2], img.shape[1]))
    y_start = int(max(crop_box[1], 0))
    y_end = int(min(crop_box[1] + crop_box[3], img.shape[0]))
    w = x_end - x_start
    h = y_end - y_start

    # Fast path: crop box is entirely inside the image -- just slice,
    # no zero-padding array needed.
    needs_top_pad = crop_box[1] < 0
    needs_left_pad = crop_box[0] < 0
    needs_bottom_pad = crop_box[1] + crop_box[3] > img.shape[0]
    needs_right_pad = crop_box[0] + crop_box[2] > img.shape[1]

    if not (needs_top_pad or needs_left_pad or needs_bottom_pad or needs_right_pad):
        return img[y_start:y_end, x_start:x_end].astype(np.uint8)

    # Slow path: crop extends beyond image boundaries -- allocate a
    # zero-filled output and copy the overlapping region.
    if len(img.shape) == 2:
        cropped_img = np.zeros((crop_box[3], crop_box[2]), dtype=np.uint8)
    else:
        c_nb = img.shape[2]
        cropped_img = np.zeros((crop_box[3], crop_box[2], c_nb), dtype=np.uint8)

    if w == 0 or h == 0:
        return cropped_img
    x_s = 0 if crop_box[0] >= 0 else -crop_box[0]
    y_s = 0 if crop_box[1] >= 0 else -crop_box[1]
    x_e = x_s + w
    y_e = y_s + h
    cropped_img[y_s:y_e, x_s:x_e] = img[y_start:y_end, x_start:x_end]

    return cropped_img


def str2img(img_str: bytes) -> Optional[np.ndarray]:
    """Decode a raw image byte string into a NumPy array.

    Equivalent to converting ``bytes`` (encoded image) to ``np.ndarray``
    (decoded pixel array).  Uses OpenCV's ``imdecode`` with
    ``IMREAD_UNCHANGED`` so the original channel count and bit depth are
    preserved.

    Args:
        img_str: A ``bytes`` object containing an encoded image (e.g. JPEG,
            PNG, BMP).

    Returns:
        A ``np.ndarray`` representing the decoded image, or ``None`` if the
        byte string could not be decoded as an image.

    Examples:
        >>> with open("photo.jpg", "rb") as f:
        ...     raw = f.read()
        >>> img = str2img(raw)
        >>> img.shape
        (1080, 1920, 3)

    See Also:
        :func:`img2str`: The inverse operation -- encode an ``np.ndarray``
        back into ``bytes``.
    """
    img_arr = np.frombuffer(img_str, np.uint8)
    img = cv2.imdecode(img_arr, cv2.IMREAD_UNCHANGED)
    return img


def img2str(img: np.ndarray) -> bytes:
    """Encode a NumPy image array into a JPEG byte string.

    Equivalent to converting ``np.ndarray`` (decoded pixel array) to
    ``bytes`` (encoded image).  Uses OpenCV's ``imencode`` with ``.jpg``
    format.

    Args:
        img: Input image as a NumPy array (e.g. ``uint8`` with shape
            ``(H, W, C)``).

    Returns:
        A ``bytes`` object containing the JPEG-encoded image data.

    Examples:
        >>> import numpy as np
        >>> img = np.zeros((100, 100, 3), dtype=np.uint8)
        >>> raw = img2str(img)
        >>> isinstance(raw, bytes)
        True

    See Also:
        :func:`str2img`: The inverse operation -- decode ``bytes`` back
        into an ``np.ndarray``.
    """
    img_str = cv2.imencode(".jpg", img)[1].tobytes()
    return img_str
