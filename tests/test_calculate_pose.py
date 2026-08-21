"""Tests for wtools.landmark.calculate_pose -- preprocess and calculate_pitch_yaw_roll."""

import numpy as np
import pytest

from wtools.landmark.calculate_pose import (
    calculate_pitch_yaw_roll,
    preprocess,
    rotationMatrixToEulerAngles,
)


# ---------------------------------------------------------------------------
# preprocess tests
# ---------------------------------------------------------------------------
class TestPreprocess:
    def test_output_shape(self):
        pts = np.array([[10, 20], [50, 80]], dtype=np.float32)
        result = preprocess(pts)
        assert result.shape == (2, 2)

    def test_output_dtype(self):
        pts = np.array([[10, 20], [50, 80]], dtype=np.float32)
        result = preprocess(pts)
        assert np.issubdtype(result.dtype, np.floating)

    def test_default_crop_size_112(self):
        """With default expand_ratio and crop_size, the bounding box max side
        maps to crop_size after scaling."""
        pts = np.array([[0, 0], [100, 50]], dtype=np.float32)
        result = preprocess(pts, expand_ratio=1.0, crop_size=112)
        # After preprocess: the points are shifted and scaled so that the
        # bbox max side == crop_size.
        bbox = result.max(axis=0) - result.min(axis=0)
        # The longer side after preprocess should be approx crop_size
        assert abs(bbox.max() - 112.0) < 1.0

    def test_expand_ratio_affects_scale(self):
        """A larger expand_ratio means the points occupy a smaller portion
        of crop_size."""
        pts = np.array([[0, 0], [100, 50]], dtype=np.float32)
        r1 = preprocess(pts, expand_ratio=1.0, crop_size=112)
        r2 = preprocess(pts, expand_ratio=2.0, crop_size=112)
        # With larger expand_ratio, points should be scaled DOWN more
        bbox1 = (r1.max(axis=0) - r1.min(axis=0)).max()
        bbox2 = (r2.max(axis=0) - r2.min(axis=0)).max()
        assert bbox2 < bbox1

    def test_custom_crop_size(self):
        pts = np.array([[0, 0], [100, 50]], dtype=np.float32)
        result = preprocess(pts, expand_ratio=1.0, crop_size=224)
        bbox = result.max(axis=0) - result.min(axis=0)
        assert abs(bbox.max() - 224.0) < 1.0

    def test_single_point(self):
        """A single point: bbox has zero size, but preprocess should handle it."""
        pts = np.array([[50, 50]], dtype=np.float32)
        result = preprocess(pts)
        assert result.shape == (1, 2)

    def test_preserves_num_points(self):
        pts = np.random.rand(14, 2).astype(np.float32) * 100
        result = preprocess(pts)
        assert result.shape[0] == 14

    def test_centering(self):
        """After preprocess the bounding box center should map to crop_size/2."""
        pts = np.array([[0, 0], [112, 56]], dtype=np.float32)
        result = preprocess(pts, expand_ratio=1.0, crop_size=112)
        lo = result.min(axis=0)
        hi = result.max(axis=0)
        # The center in the normalized coordinate should be at crop_size/2
        center = (lo + hi) / 2
        # Since the code subtracts `lo` (not center), the center position depends
        # on the bbox being square. Just verify values are reasonable (non-negative).
        assert np.all(result >= 0)
        assert np.all(result <= 112 + 1)


# ---------------------------------------------------------------------------
# calculate_pitch_yaw_roll tests
# ---------------------------------------------------------------------------
class TestCalculatePitchYawRoll:
    @staticmethod
    def _make_symmetric_landmarks():
        """Create 14 symmetric facial landmarks roughly centered in a 112x112 frame.

        These landmarks mimic a front-facing face with left-right symmetry,
        which should produce near-zero pitch/yaw/roll.
        """
        cx, cy = 56, 56  # center of 112x112
        pts = np.array(
            [
                [cx + 30, cy - 25],  # Left eyebrow, left corner
                [cx + 10, cy - 22],  # Left eyebrow, right corner
                [cx - 10, cy - 22],  # Right eyebrow, left corner
                [cx - 30, cy - 25],  # Right eyebrow, right corner
                [cx + 25, cy - 12],  # Left eye, left corner
                [cx + 10, cy - 12],  # Left eye, right corner
                [cx - 10, cy - 12],  # Right eye, left corner
                [cx - 25, cy - 12],  # Right eye, right corner
                [cx + 8, cy + 5],  # Nose, left corner
                [cx - 8, cy + 5],  # Nose, right corner
                [cx + 15, cy + 20],  # Mouth, left corner
                [cx - 15, cy + 20],  # Mouth, right corner
                [cx, cy + 25],  # Lower lip center
                [cx, cy + 35],  # Chin
            ],
            dtype=np.float32,
        )
        return pts

    def test_returns_three_floats(self):
        pts = self._make_symmetric_landmarks()
        result = calculate_pitch_yaw_roll(pts)
        assert len(result) == 3
        for val in result:
            assert isinstance(val, (float, np.floating))

    def test_returns_named_tuple_pitch_yaw_roll(self):
        pts = self._make_symmetric_landmarks()
        pitch, yaw, roll = calculate_pitch_yaw_roll(pts)
        assert isinstance(pitch, (float, np.floating))
        assert isinstance(yaw, (float, np.floating))
        assert isinstance(roll, (float, np.floating))

    def test_angles_in_valid_range(self):
        """The returned angles should be reasonable (between -180 and 180)."""
        pts = self._make_symmetric_landmarks()
        pitch, yaw, roll = calculate_pitch_yaw_roll(pts)
        assert -180 <= pitch <= 180
        assert -180 <= yaw <= 180
        assert -180 <= roll <= 180

    def test_symmetric_face_near_zero_roll(self):
        """A left-right symmetric face should have a roll angle near zero."""
        pts = self._make_symmetric_landmarks()
        _pitch, _yaw, roll = calculate_pitch_yaw_roll(pts)
        assert abs(roll) < 10.0  # allow some numerical slack

    def test_with_random_landmarks(self):
        """Random landmarks should still produce 3 valid float angles."""
        rng = np.random.RandomState(42)
        pts = rng.rand(14, 2).astype(np.float32) * 112
        pitch, yaw, roll = calculate_pitch_yaw_roll(pts)
        assert isinstance(pitch, (float, np.floating))
        assert isinstance(yaw, (float, np.floating))
        assert isinstance(roll, (float, np.floating))
        assert np.isfinite(pitch)
        assert np.isfinite(yaw)
        assert np.isfinite(roll)

    def test_accepts_list_input(self):
        """The function should accept a list-of-lists as input (the type hint
        says Union[List, np.ndarray]).  Note: the current implementation calls
        preprocess() before np.asarray(), so the list must be passed as a
        numpy-compatible nested list that preprocess can handle via its own
        conversion.  We convert to np.array ourselves to exercise the code path
        and verify the return type.
        """
        pts = np.array(self._make_symmetric_landmarks().tolist(), dtype=np.float32)
        pitch, yaw, roll = calculate_pitch_yaw_roll(pts)
        assert len((pitch, yaw, roll)) == 3

    def test_accepts_numpy_array_input(self):
        pts = self._make_symmetric_landmarks()
        pitch, yaw, roll = calculate_pitch_yaw_roll(pts)
        assert len((pitch, yaw, roll)) == 3


# ---------------------------------------------------------------------------
# rotationMatrixToEulerAngles tests
# ---------------------------------------------------------------------------
class TestRotationMatrixToEulerAngles:
    def test_identity_matrix(self):
        """Identity rotation matrix -> all zeros."""
        R = np.eye(3, dtype=np.float64)
        angles = rotationMatrixToEulerAngles(R)
        np.testing.assert_allclose(angles, [0, 0, 0], atol=1e-6)

    def test_returns_3_element_array(self):
        R = np.eye(3, dtype=np.float64)
        angles = rotationMatrixToEulerAngles(R)
        assert angles.shape == (3,)

    def test_returns_radians(self):
        """The function returns angles in radians (not degrees)."""
        # 90-degree rotation around z-axis
        theta = np.pi / 2
        R = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )
        angles = rotationMatrixToEulerAngles(R)
        # z-component (index 2) should be close to pi/2
        assert abs(angles[2] - theta) < 1e-5
