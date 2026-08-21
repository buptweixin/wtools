#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

# --- Module-level constants for calculate_pitch_yaw_roll ---
# Hoisted out of the function so they are created only once at import time
# instead of being rebuilt on every call (they never change).

# Canonical 3-D facial landmark model (14 points).
# X-Y-Z with X pointing forward, Y on the left, Z up.
_LEFT_EYEBROW_LEFT = [6.825897, 6.760612, 4.402142]
_LEFT_EYEBROW_RIGHT = [1.330353, 7.122144, 6.903745]
_RIGHT_EYEBROW_LEFT = [-1.330353, 7.122144, 6.903745]
_RIGHT_EYEBROW_RIGHT = [-6.825897, 6.760612, 4.402142]
_LEFT_EYE_LEFT = [5.311432, 5.485328, 3.987654]
_LEFT_EYE_RIGHT = [1.789930, 5.393625, 4.413414]
_RIGHT_EYE_LEFT = [-1.789930, 5.393625, 4.413414]
_RIGHT_EYE_RIGHT = [-5.311432, 5.485328, 3.987654]
_NOSE_LEFT = [2.005628, 1.409845, 6.165652]
_NOSE_RIGHT = [-2.005628, 1.409845, 6.165652]
_MOUTH_LEFT = [2.774015, -2.080775, 5.048531]
_MOUTH_RIGHT = [-2.774015, -2.080775, 5.048531]
_LOWER_LIP = [0.000000, -3.116408, 6.097667]
_CHIN = [0.000000, -7.415691, 4.070434]

LANDMARKS_3D = np.float32(
    [  # type: ignore
        _LEFT_EYEBROW_LEFT,
        _LEFT_EYEBROW_RIGHT,
        _RIGHT_EYEBROW_LEFT,
        _RIGHT_EYEBROW_RIGHT,
        _LEFT_EYE_LEFT,
        _LEFT_EYE_RIGHT,
        _RIGHT_EYE_LEFT,
        _RIGHT_EYE_RIGHT,
        _NOSE_LEFT,
        _NOSE_RIGHT,
        _MOUTH_LEFT,
        _MOUTH_RIGHT,
        _LOWER_LIP,
        _CHIN,
    ]
)

# Camera intrinsics are fixed (cam_w = cam_h = 112, 60-deg horizontal FOV).
_CAM_W = 112
_CAM_H = 112
_C_X = _CAM_W / 2
_C_Y = _CAM_H / 2
_F_X = _C_X / np.tan(60 / 2 * np.pi / 180)
CAMERA_MATRIX = np.float32(
    [[_F_X, 0.0, _C_X], [0.0, _F_X, _C_Y], [0.0, 0.0, 1.0]]  # type: ignore
)
CAMERA_DISTORTION = np.float32([0.0, 0.0, 0.0, 0.0, 0.0]  # type: ignore
)


def preprocess(
    pts: np.ndarray,
    expand_ratio: float = 1.3,
    crop_size: int = 112,
) -> np.ndarray:
    """Normalize a set of 2-D points into a centered, square coordinate system.

    The points are translated so their bounding-box center moves to the
    origin, then scaled so the longer side of the (expanded) bounding box
    maps to ``crop_size`` pixels.

    Args:
        pts: Array of shape ``(N, 2)`` containing the 2-D point
            coordinates.
        expand_ratio: Ratio by which to expand the bounding box before
            cropping. A value of ``1.0`` means no expansion. Defaults to
            ``1.3``.
        crop_size: Target size (in pixels) of the square crop. The points
            are scaled so the expanded bounding box maps to this size.
            Defaults to ``112``.

    Returns:
        A ``np.ndarray`` of shape ``(N, 2)`` containing the normalized
        point coordinates.

    Examples:
        >>> import numpy as np
        >>> pts = np.array([[10, 20], [50, 80]], dtype=np.float32)
        >>> norm = preprocess(pts, expand_ratio=1.3, crop_size=112)
        >>> norm.shape
        (2, 2)
    """
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    center = (lo + hi) / 2
    size = (hi - lo).max() * expand_ratio
    lo = center - size / 2
    hi = center + size / 2
    pts = pts - lo.reshape(-1, 2)
    scale = size / crop_size
    pts = pts / scale
    return pts


def calculate_pitch_yaw_roll(
    landmarks_2D: Union[List, np.ndarray]
) -> Tuple[float, float, float]:
    """Calculate head pose (pitch, yaw, roll) from 2-D facial landmarks.

    Uses a Perspective-n-Point (PnP) solver to estimate the 3-D rotation
    of the head by matching 2-D landmark positions to a canonical 3-D
    face model. The resulting rotation matrix is decomposed into Euler
    angles.

    Reference: https://github.com/guoqiangqi/PFLD/blob/master/euler_angles_utils.py

    Args:
        landmarks_2D: 2-D facial landmarks as a list or NumPy array of
            shape ``(14, 2)``. Exactly **14 points** are required,
            arranged in the following order:

            1. Left eyebrow left corner
            2. Left eyebrow right corner
            3. Right eyebrow left corner
            4. Right eyebrow right corner
            5. Left eye left corner
            6. Left eye right corner
            7. Right eye left corner
            8. Right eye right corner
            9. Nose left corner
            10. Nose right corner
            11. Mouth left corner
            12. Mouth right corner
            13. Lower lip center
            14. Chin

            The point order must match the canonical 3-D model defined
            inside this function. For dlib 68-point landmarks the
            corresponding indices are
            ``[17, 21, 22, 26, 36, 39, 42, 45, 31, 35, 48, 54, 57, 8]``;
            for WFLW 98-point landmarks they are
            ``[33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16]``.

    Returns:
        A tuple ``(pitch, yaw, roll)`` of floats representing the head
        pose angles in degrees.

    Raises:
        AssertionError: If ``landmarks_2D`` is ``None``.

    Examples:
        >>> import numpy as np
        >>> pts = np.random.rand(14, 2).astype(np.float32) * 112
        >>> pitch, yaw, roll = calculate_pitch_yaw_roll(pts)
        >>> abs(pitch) < 180 and abs(yaw) < 180 and abs(roll) < 180
        True
    """

    # Use module-level constants to avoid rebuilding arrays on every call.
    landmarks_2D = preprocess(np.asarray(landmarks_2D), expand_ratio=1.3, crop_size=_CAM_W)

    # Return the 2D position of our landmarks
    assert landmarks_2D is not None, "landmarks_2D is None"
    landmarks_2D = np.asarray(landmarks_2D, dtype=np.float32)
    if landmarks_2D.size % 2 != 0:
        raise ValueError(
            f"landmarks_2D must contain an even number of elements to reshape "
            f"into (N, 2), got {landmarks_2D.size} elements."
        )
    landmarks_2D = landmarks_2D.reshape(-1, 2)
    # Applying the PnP solver to find the 3D pose
    # of the head from the 2D position of the
    # landmarks.
    # retval - bool
    # rvec - Output rotation vector that, together with tvec, brings
    # points from the world coordinate system to the camera coordinate system.
    # tvec - Output translation vector. It is the position of the world origin (SELLION) in camera co-ords
    retval, rvec, tvec = cv2.solvePnP(
        LANDMARKS_3D, landmarks_2D, CAMERA_MATRIX, CAMERA_DISTORTION
    )

    # Get as input the rotational vector
    # Return a rotational matrix
    rmat, _ = cv2.Rodrigues(rvec)
    pose_mat = cv2.hconcat((rmat, tvec))

    # euler_angles contain (pitch, yaw, roll)
    # euler_angles = cv2.DecomposeProjectionMatrix(projMatrix=rmat, cameraMatrix=self.camera_matrix, rotMatrix, transVect, rotMatrX=None, rotMatrY=None, rotMatrZ=None)
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)
    pitch, yaw, roll = [a[0] for a in euler_angles]
    return pitch, yaw, roll


def rotation_matrix_to_euler_angles(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to Euler angles (roll, pitch, yaw).

    Uses a Tait-Bryan angle convention. A singularity check is performed to
    avoid Gimbal lock: when the matrix is nearly singular the roll component
    is set to zero.

    Args:
        R: A 3x3 rotation matrix as a NumPy array.

    Returns:
        A ``np.ndarray`` of shape ``(3,)`` containing the Euler angles
        ``[x, y, z]`` (roll, pitch, yaw) in radians.

    Examples:
        >>> import numpy as np
        >>> R = np.eye(3, dtype=np.float64)
        >>> angles = rotation_matrix_to_euler_angles(R)
        >>> np.allclose(angles, [0, 0, 0])
        True
    """
    # assert(isRotationMatrix(R))
    # To prevent the Gimbal Lock it is possible to use
    # a threshold of 1e-6 for discrimination
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])


# Backward-compatible alias (camelCase -> snake_case).
rotationMatrixToEulerAngles = rotation_matrix_to_euler_angles
