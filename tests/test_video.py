"""Tests for wtools.utils.video -- JPEGBIN1 read/write."""

import os
import struct

import cv2
import numpy as np
import pytest

from wtools.utils.video import (
    FORMAT_VERSION,
    HEADER_SIZE,
    HEADER_STRUCT,
    MAGIC,
    JPEGBinError,
    read_jpeg_bin,
    read_jpeg_bin_metadata,
    video_to_jpeg_bin,
    write_jpeg_bin,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_frames(n=5, height=32, width=48, seed=42):
    """Generate a list of random BGR frames."""
    rng = np.random.RandomState(seed)
    return [rng.randint(0, 256, (height, width, 3), dtype=np.uint8) for _ in range(n)]


def _write_bin(
    path,
    frames,
    source_fps=30.0,
    sample_fps=2.0,
    total_num_frames=300,
    frame_indices=None,
    jpeg_quality=95,
):
    """Convenience wrapper around write_jpeg_bin."""
    write_jpeg_bin(
        output_path=str(path),
        frames=frames,
        source_fps=source_fps,
        sample_fps=sample_fps,
        total_num_frames=total_num_frames,
        frame_indices=frame_indices,
        jpeg_quality=jpeg_quality,
    )


# ---------------------------------------------------------------------------
# write_jpeg_bin tests
# ---------------------------------------------------------------------------
class TestWriteJpegBin:
    def test_basic_write(self, tmp_path):
        """A valid bin file should be created and match expected size."""
        frames = _make_frames(n=5, height=32, width=48)
        path = tmp_path / "test.bin"
        _write_bin(path, frames, total_num_frames=300)

        assert path.exists()
        # Expected file size: header + 2 index arrays + JPEG payloads
        meta = read_jpeg_bin_metadata(str(path))
        expected = meta["payload_offset"] + sum(meta["jpeg_lengths"])
        assert os.path.getsize(str(path)) == expected

    def test_header_magic(self, tmp_path):
        """The first 8 bytes should be the JPEGBIN1 magic."""
        frames = _make_frames(n=3)
        path = tmp_path / "magic.bin"
        _write_bin(path, frames)

        with open(str(path), "rb") as f:
            magic = f.read(8)
        assert magic == MAGIC

    def test_header_version(self, tmp_path):
        """The version field should be FORMAT_VERSION."""
        frames = _make_frames(n=3)
        path = tmp_path / "ver.bin"
        _write_bin(path, frames)

        with open(str(path), "rb") as f:
            f.read(8)  # skip magic
            version = struct.unpack("<I", f.read(4))[0]
        assert version == FORMAT_VERSION

    def test_header_size_is_60(self):
        """HEADER_STRUCT should pack to exactly 60 bytes."""
        assert HEADER_SIZE == 60
        assert HEADER_STRUCT.size == 60

    def test_empty_frames_raises(self, tmp_path):
        """An empty frame list should raise ValueError."""
        path = tmp_path / "empty.bin"
        with pytest.raises(ValueError, match="empty"):
            _write_bin(path, [])

    def test_inconsistent_frame_shapes_raises(self, tmp_path):
        """Frames with different dimensions should raise ValueError."""
        frames = [
            np.zeros((32, 48, 3), dtype=np.uint8),
            np.zeros((64, 48, 3), dtype=np.uint8),
        ]
        path = tmp_path / "bad.bin"
        with pytest.raises(ValueError, match="shape"):
            _write_bin(path, frames)

    def test_custom_frame_indices(self, tmp_path):
        """Custom frame_indices should be stored and retrievable."""
        frames = _make_frames(n=4)
        indices = [10, 25, 50, 100]
        path = tmp_path / "custom_idx.bin"
        _write_bin(path, frames, frame_indices=indices)

        meta = read_jpeg_bin_metadata(str(path))
        assert meta["frame_indices"] == indices

    def test_wrong_frame_indices_count_raises(self, tmp_path):
        """frame_indices with wrong length should raise ValueError."""
        frames = _make_frames(n=4)
        path = tmp_path / "bad_idx.bin"
        with pytest.raises(ValueError, match="frame_indices"):
            _write_bin(path, frames, frame_indices=[0, 1, 2])

    def test_jpeg_quality_affects_size(self, tmp_path):
        """Lower quality should produce a smaller file (for random frames)."""
        frames = _make_frames(n=10, height=64, width=64, seed=123)
        path_hq = tmp_path / "hq.bin"
        path_lq = tmp_path / "lq.bin"
        _write_bin(path_hq, frames, jpeg_quality=95)
        _write_bin(path_lq, frames, jpeg_quality=20)

        # Random data compresses poorly with JPEG, but quality=20 should
        # still be noticeably smaller than quality=95.
        assert os.path.getsize(str(path_lq)) < os.path.getsize(str(path_hq))


# ---------------------------------------------------------------------------
# read_jpeg_bin_metadata tests
# ---------------------------------------------------------------------------
class TestReadJpegBinMetadata:
    def test_basic_metadata(self, tmp_path):
        """Metadata fields should match what was written."""
        frames = _make_frames(n=5, height=32, width=48)
        path = tmp_path / "meta.bin"
        _write_bin(
            path,
            frames,
            source_fps=25.0,
            sample_fps=5.0,
            total_num_frames=250,
            frame_indices=[10, 15, 20, 25, 30],
            jpeg_quality=90,
        )

        meta = read_jpeg_bin_metadata(str(path))
        assert meta["nframes"] == 5
        assert meta["source_fps"] == 25.0
        assert meta["sample_fps"] == 5.0
        assert meta["total_num_frames"] == 250
        assert meta["width"] == 48
        assert meta["height"] == 32
        assert meta["jpeg_quality"] == 90
        assert meta["frame_indices"] == [10, 15, 20, 25, 30]
        assert meta["video_backend"] == "jpeg_bin"
        assert len(meta["jpeg_lengths"]) == 5
        assert all(l > 0 for l in meta["jpeg_lengths"])
        assert meta["payload_offset"] == 60 + 5 * 16

    def test_selected_duration_default(self, tmp_path):
        """selected_duration should default to total_num_frames / source_fps."""
        frames = _make_frames(n=3)
        path = tmp_path / "dur.bin"
        _write_bin(path, frames, source_fps=30.0, total_num_frames=300)

        meta = read_jpeg_bin_metadata(str(path))
        assert meta["selected_duration"] == pytest.approx(10.0)

    def test_validate_size_passes(self, tmp_path):
        """A valid file should pass size validation without error."""
        frames = _make_frames(n=3)
        path = tmp_path / "ok.bin"
        _write_bin(path, frames)
        # Should not raise
        read_jpeg_bin_metadata(str(path), validate_size=True)

    def test_validate_size_truncated_raises(self, tmp_path):
        """A truncated file should fail size validation."""
        frames = _make_frames(n=5)
        path = tmp_path / "trunc.bin"
        _write_bin(path, frames)

        # Truncate the file
        real_size = os.path.getsize(str(path))
        with open(str(path), "r+b") as f:
            f.truncate(real_size - 10)

        with pytest.raises(JPEGBinError, match="Corrupt"):
            read_jpeg_bin_metadata(str(path), validate_size=True)

    def test_validate_size_can_be_disabled(self, tmp_path):
        """validate_size=False should skip the size check on a truncated file."""
        frames = _make_frames(n=5)
        path = tmp_path / "trunc2.bin"
        _write_bin(path, frames)

        real_size = os.path.getsize(str(path))
        with open(str(path), "r+b") as f:
            f.truncate(real_size - 10)

        # Should not raise when validation is disabled
        meta = read_jpeg_bin_metadata(str(path), validate_size=False)
        assert meta["nframes"] == 5

    def test_invalid_magic_raises(self, tmp_path):
        """A file with wrong magic bytes should raise JPEGBinError."""
        path = tmp_path / "badmagic.bin"
        with open(str(path), "wb") as f:
            f.write(b"BADMAGIC" + b"\x00" * 52)

        with pytest.raises(JPEGBinError, match="magic"):
            read_jpeg_bin_metadata(str(path), validate_size=False)

    def test_unsupported_version_raises(self, tmp_path):
        """A file with unsupported version should raise JPEGBinError."""
        path = tmp_path / "badver.bin"
        header = HEADER_STRUCT.pack(
            MAGIC,
            999,  # unsupported version
            1,
            100,
            30.0,
            2.0,
            10.0,
            32,
            32,
            95,
        )
        with open(str(path), "wb") as f:
            f.write(header)
            f.write(b"\x00" * 16)  # minimal index arrays for 1 frame
            f.write(b"\xff\xd8\xff\xd9")  # minimal JPEG (SOI + EOI)

        with pytest.raises(JPEGBinError, match="version"):
            read_jpeg_bin_metadata(str(path), validate_size=False)

    def test_truncated_header_raises(self, tmp_path):
        """A file shorter than the header should raise JPEGBinError."""
        path = tmp_path / "short.bin"
        with open(str(path), "wb") as f:
            f.write(b"JPEGBIN1" + b"\x00" * 10)

        with pytest.raises(JPEGBinError, match="Truncated"):
            read_jpeg_bin_metadata(str(path), validate_size=False)

    def test_file_not_found(self):
        """A non-existent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            read_jpeg_bin_metadata("/nonexistent/path/video.bin")


# ---------------------------------------------------------------------------
# read_jpeg_bin tests
# ---------------------------------------------------------------------------
class TestReadJpegBin:
    def test_roundtrip_shape_and_dtype(self, tmp_path):
        """Read back frames and verify shape and dtype."""
        frames = _make_frames(n=5, height=32, width=48)
        path = tmp_path / "rt.bin"
        _write_bin(path, frames)

        video, meta, sample_fps = read_jpeg_bin(str(path))
        assert video.shape == (5, 32, 48, 3)
        assert video.dtype == np.uint8

    def test_roundtrip_rgb_color_order(self, tmp_path):
        """Frames should be returned in RGB order (not BGR)."""
        # Create a frame with a known R > G > B pattern
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        frame[:, :, 0] = 200  # B channel (high)
        frame[:, :, 1] = 100  # G channel
        frame[:, :, 2] = 50  # R channel (low)

        path = tmp_path / "color.bin"
        _write_bin(path, [frame])

        video, _, _ = read_jpeg_bin(str(path))
        # JPEG is lossy, so check approximate values
        # In the output RGB array, channel 0 (R) should be ~50,
        # channel 2 (B) should be ~200.
        assert video[0, 0, 0, 0] < video[0, 0, 0, 2]  # R < B

    def test_roundtrip_content_similarity(self, tmp_path):
        """Decoded frames should be visually similar to originals.

        Uses smooth gradient images (which JPEG handles well) instead of
        random noise, and accounts for the BGR->RGB conversion done by
        read_jpeg_bin.
        """
        # Create smooth gradient frames (BGR format, as OpenCV expects)
        frames = []
        for i in range(3):
            img = np.zeros((64, 64, 3), dtype=np.uint8)
            img[:, :, 0] = np.linspace(0, 200, 64, dtype=np.uint8)[:, None]  # B
            img[:, :, 1] = np.linspace(50, 150, 64, dtype=np.uint8)[:, None]  # G
            img[:, :, 2] = np.linspace(100, 250, 64, dtype=np.uint8)[:, None]  # R
            frames.append(img)

        path = tmp_path / "similar.bin"
        _write_bin(path, frames, jpeg_quality=95)

        video, _, _ = read_jpeg_bin(str(path))
        for i, orig_bgr in enumerate(frames):
            # read_jpeg_bin returns RGB, so convert original BGR -> RGB for comparison
            orig_rgb = orig_bgr[:, :, ::-1].copy()
            mse = np.mean(
                (video[i].astype(np.float64) - orig_rgb.astype(np.float64)) ** 2
            )
            assert mse < 50.0, f"Frame {i} MSE too high: {mse}"

    def test_metadata_returned(self, tmp_path):
        """The returned metadata should have the expected keys."""
        frames = _make_frames(n=3)
        path = tmp_path / "meta2.bin"
        _write_bin(
            path,
            frames,
            source_fps=24.0,
            sample_fps=3.0,
            total_num_frames=120,
        )

        _, meta, sample_fps = read_jpeg_bin(str(path))
        assert meta["source_fps"] == 24.0
        assert meta["sample_fps"] == 3.0
        assert meta["total_num_frames"] == 120
        assert sample_fps == 3.0
        # jpeg_lengths and payload_offset should have been popped
        assert "jpeg_lengths" not in meta
        assert "payload_offset" not in meta

    def test_sample_fps_returned(self, tmp_path):
        """The third return value should be sample_fps."""
        frames = _make_frames(n=2)
        path = tmp_path / "fps.bin"
        _write_bin(path, frames, sample_fps=4.0)

        _, _, sample_fps = read_jpeg_bin(str(path))
        assert sample_fps == 4.0

    def test_single_frame(self, tmp_path):
        """A bin file with a single frame should work."""
        frames = _make_frames(n=1, height=16, width=16)
        path = tmp_path / "single.bin"
        _write_bin(path, frames)

        video, meta, _ = read_jpeg_bin(str(path))
        assert video.shape == (1, 16, 16, 3)
        assert meta["nframes"] == 1

    def test_corrupt_jpeg_payload_raises(self, tmp_path):
        """A corrupt JPEG payload should raise JPEGBinError."""
        frames = _make_frames(n=3)
        path = tmp_path / "corrupt.bin"
        _write_bin(path, frames)

        # Corrupt one of the JPEG payloads by overwriting with garbage
        meta = read_jpeg_bin_metadata(str(path), validate_size=False)
        offset = meta["payload_offset"] + meta["jpeg_lengths"][0] + 2
        with open(str(path), "r+b") as f:
            f.seek(offset)
            f.write(b"\x00" * 20)

        with pytest.raises(JPEGBinError, match="JPEG decode failed"):
            read_jpeg_bin(str(path))


# ---------------------------------------------------------------------------
# read_jpeg_bin return_format tests
# ---------------------------------------------------------------------------
class TestReadJpegBinReturnFormat:
    def test_base64_returns_list_of_str(self, tmp_path):
        """return_format='base64' should return a list of str."""
        frames = _make_frames(n=5, height=32, width=48)
        path = tmp_path / "b64.bin"
        _write_bin(path, frames)

        result, meta, fps = read_jpeg_bin(str(path), return_format="base64")
        assert isinstance(result, list)
        assert len(result) == 5
        assert all(isinstance(s, str) for s in result)

    def test_base64_decodes_to_valid_jpeg(self, tmp_path):
        """Each base64 string should decode to valid JPEG bytes."""
        import base64 as b64

        frames = _make_frames(n=3, height=32, width=48)
        path = tmp_path / "b64valid.bin"
        _write_bin(path, frames)

        result, _, _ = read_jpeg_bin(str(path), return_format="base64")
        for i, s in enumerate(result):
            raw = b64.b64decode(s)
            # JPEG files start with FFD8 and end with FFD9
            assert raw[:2] == b"\xff\xd8", f"Frame {i}: not a valid JPEG start"
            assert raw[-2:] == b"\xff\xd9", f"Frame {i}: not a valid JPEG end"

    def test_base64_no_decode_skips_corrupt_payload(self, tmp_path):
        """base64 mode should not raise on corrupt JPEG data (no decode)."""
        frames = _make_frames(n=3)
        path = tmp_path / "b64corrupt.bin"
        _write_bin(path, frames)

        meta = read_jpeg_bin_metadata(str(path), validate_size=False)
        offset = meta["payload_offset"] + meta["jpeg_lengths"][0] + 2
        with open(str(path), "r+b") as f:
            f.seek(offset)
            f.write(b"\x00" * 20)

        # Should NOT raise in base64 mode (no JPEG decoding)
        result, _, _ = read_jpeg_bin(str(path), return_format="base64")
        assert len(result) == 3

    def test_base64_consistent_with_numpy_count(self, tmp_path):
        """Both modes should return the same number of frames."""
        frames = _make_frames(n=7, height=16, width=16)
        path = tmp_path / "both.bin"
        _write_bin(path, frames)

        video_np, _, _ = read_jpeg_bin(str(path), return_format="numpy")
        frames_b64, _, _ = read_jpeg_bin(str(path), return_format="base64")
        assert video_np.shape[0] == len(frames_b64)

    def test_base64_metadata_same_as_numpy(self, tmp_path):
        """Metadata should be identical regardless of return_format."""
        frames = _make_frames(n=3)
        path = tmp_path / "metacmp.bin"
        _write_bin(path, frames, source_fps=25.0, sample_fps=5.0, total_num_frames=100)

        _, meta_np, fps_np = read_jpeg_bin(str(path), return_format="numpy")
        _, meta_b64, fps_b64 = read_jpeg_bin(str(path), return_format="base64")
        assert meta_np == meta_b64
        assert fps_np == fps_b64

    def test_invalid_return_format_raises(self, tmp_path):
        """An invalid return_format should raise ValueError."""
        frames = _make_frames(n=2)
        path = tmp_path / "badfmt.bin"
        _write_bin(path, frames)

        with pytest.raises(ValueError, match="return_format"):
            read_jpeg_bin(str(path), return_format="png")


# ---------------------------------------------------------------------------
# read_jpeg_bin sub-sampling tests
# ---------------------------------------------------------------------------
class TestReadJpegBinSubSampling:
    def test_num_frames_uniform(self, tmp_path):
        """num_frames should uniformly sub-sample to exactly that count."""
        frames = _make_frames(n=20, height=16, width=16)
        path = tmp_path / "sub1.bin"
        _write_bin(path, frames)

        video, meta, fps = read_jpeg_bin(str(path), num_frames=8)
        assert video.shape == (8, 16, 16, 3)
        assert meta["nframes"] == 8

    def test_num_frames_equals_total(self, tmp_path):
        """num_frames == total should return all frames unchanged."""
        frames = _make_frames(n=5, height=16, width=16)
        path = tmp_path / "sub2.bin"
        _write_bin(path, frames)

        video, meta, _ = read_jpeg_bin(str(path), num_frames=5)
        assert video.shape == (5, 16, 16, 3)
        assert meta["nframes"] == 5

    def test_num_frames_one(self, tmp_path):
        """num_frames=1 should return a single frame."""
        frames = _make_frames(n=10, height=16, width=16)
        path = tmp_path / "sub3.bin"
        _write_bin(path, frames)

        video, meta, _ = read_jpeg_bin(str(path), num_frames=1)
        assert video.shape == (1, 16, 16, 3)
        assert meta["nframes"] == 1

    def test_num_frames_exceeds_available_raises(self, tmp_path):
        """num_frames > available frames should raise ValueError."""
        frames = _make_frames(n=5, height=16, width=16)
        path = tmp_path / "sub4.bin"
        _write_bin(path, frames)

        with pytest.raises(ValueError, match="exceeds"):
            read_jpeg_bin(str(path), num_frames=10)

    def test_num_frames_zero_raises(self, tmp_path):
        """num_frames=0 should raise ValueError."""
        frames = _make_frames(n=3, height=16, width=16)
        path = tmp_path / "sub5.bin"
        _write_bin(path, frames)

        with pytest.raises(ValueError, match="num_frames"):
            read_jpeg_bin(str(path), num_frames=0)

    def test_frame_interval(self, tmp_path):
        """frame_interval=2 should take every other frame."""
        frames = _make_frames(n=10, height=16, width=16)
        path = tmp_path / "int1.bin"
        _write_bin(path, frames)

        video, meta, _ = read_jpeg_bin(str(path), frame_interval=2)
        assert video.shape == (5, 16, 16, 3)
        assert meta["nframes"] == 5

    def test_frame_interval_3(self, tmp_path):
        """frame_interval=3 should take every 3rd frame."""
        frames = _make_frames(n=12, height=16, width=16)
        path = tmp_path / "int2.bin"
        _write_bin(path, frames)

        video, _, _ = read_jpeg_bin(str(path), frame_interval=3)
        assert video.shape == (4, 16, 16, 3)

    def test_frame_interval_1_returns_all(self, tmp_path):
        """frame_interval=1 should return all frames."""
        frames = _make_frames(n=6, height=16, width=16)
        path = tmp_path / "int3.bin"
        _write_bin(path, frames)

        video, _, _ = read_jpeg_bin(str(path), frame_interval=1)
        assert video.shape == (6, 16, 16, 3)

    def test_frame_interval_zero_raises(self, tmp_path):
        """frame_interval=0 should raise ValueError."""
        frames = _make_frames(n=3, height=16, width=16)
        path = tmp_path / "int4.bin"
        _write_bin(path, frames)

        with pytest.raises(ValueError, match="frame_interval"):
            read_jpeg_bin(str(path), frame_interval=0)

    def test_start_end_frame(self, tmp_path):
        """start_frame and end_frame should clip the range."""
        frames = _make_frames(n=10, height=16, width=16)
        path = tmp_path / "range1.bin"
        _write_bin(path, frames)

        video, meta, _ = read_jpeg_bin(str(path), start_frame=2, end_frame=7)
        assert video.shape == (5, 16, 16, 3)
        assert meta["nframes"] == 5

    def test_start_frame_only(self, tmp_path):
        """start_frame without end_frame should go to the end."""
        frames = _make_frames(n=10, height=16, width=16)
        path = tmp_path / "range2.bin"
        _write_bin(path, frames)

        video, _, _ = read_jpeg_bin(str(path), start_frame=8)
        assert video.shape == (2, 16, 16, 3)

    def test_end_frame_only(self, tmp_path):
        """end_frame without start_frame should start from 0."""
        frames = _make_frames(n=10, height=16, width=16)
        path = tmp_path / "range3.bin"
        _write_bin(path, frames)

        video, _, _ = read_jpeg_bin(str(path), end_frame=3)
        assert video.shape == (3, 16, 16, 3)

    def test_full_range_default(self, tmp_path):
        """No range params should return all frames."""
        frames = _make_frames(n=6, height=16, width=16)
        path = tmp_path / "range4.bin"
        _write_bin(path, frames)

        video, _, _ = read_jpeg_bin(str(path))
        assert video.shape == (6, 16, 16, 3)

    def test_invalid_range_raises(self, tmp_path):
        """start_frame >= end_frame should raise ValueError."""
        frames = _make_frames(n=5, height=16, width=16)
        path = tmp_path / "range5.bin"
        _write_bin(path, frames)

        with pytest.raises(ValueError, match="Empty frame range"):
            read_jpeg_bin(str(path), start_frame=5, end_frame=5)

    def test_combined_interval_and_num_frames(self, tmp_path):
        """frame_interval then num_frames should compose correctly."""
        frames = _make_frames(n=20, height=16, width=16)
        path = tmp_path / "combo1.bin"
        _write_bin(path, frames)

        # interval=2 -> 10 frames, then num_frames=4 from those 10
        video, meta, _ = read_jpeg_bin(str(path), frame_interval=2, num_frames=4)
        assert video.shape == (4, 16, 16, 3)
        assert meta["nframes"] == 4

    def test_combined_range_interval_num_frames(self, tmp_path):
        """All three params should compose: clip -> stride -> resample."""
        frames = _make_frames(n=20, height=16, width=16)
        path = tmp_path / "combo2.bin"
        _write_bin(path, frames)

        # range [5,20) = 15 frames, interval=2 -> 8 frames, num_frames=3
        video, meta, _ = read_jpeg_bin(
            str(path),
            start_frame=5,
            end_frame=20,
            frame_interval=2,
            num_frames=3,
        )
        assert video.shape == (3, 16, 16, 3)
        assert meta["nframes"] == 3

    def test_frame_indices_updated(self, tmp_path):
        """frames_indices in metadata should reflect the sub-sampled selection."""
        frames = _make_frames(n=10, height=16, width=16)
        path = tmp_path / "idx1.bin"
        _write_bin(path, frames, frame_indices=list(range(100, 110)))

        _, meta, _ = read_jpeg_bin(str(path), start_frame=2, end_frame=5)
        assert meta["frame_indices"] == [102, 103, 104]

    def test_frame_indices_with_interval(self, tmp_path):
        """frames_indices should reflect interval sub-sampling."""
        frames = _make_frames(n=6, height=16, width=16)
        path = tmp_path / "idx2.bin"
        _write_bin(path, frames, frame_indices=[10, 20, 30, 40, 50, 60])

        _, meta, _ = read_jpeg_bin(str(path), frame_interval=2)
        assert meta["frame_indices"] == [10, 30, 50]

    def test_sample_fps_recalculated(self, tmp_path):
        """sample_fps should be recalculated based on sub-sampled frame count."""
        # source_fps=30, total_num_frames=300 -> duration=10s
        # 10 stored frames -> after num_frames=5, sample_fps = 5/10 = 0.5
        frames = _make_frames(n=10, height=16, width=16)
        path = tmp_path / "fps1.bin"
        _write_bin(path, frames, source_fps=30.0, total_num_frames=300)

        _, _, sample_fps = read_jpeg_bin(str(path), num_frames=5)
        assert sample_fps == pytest.approx(0.5)

    def test_sub_sampling_with_base64(self, tmp_path):
        """Sub-sampling should work with return_format='base64'."""
        frames = _make_frames(n=10, height=16, width=16)
        path = tmp_path / "b64sub.bin"
        _write_bin(path, frames)

        result, meta, _ = read_jpeg_bin(str(path), return_format="base64", num_frames=4)
        assert isinstance(result, list)
        assert len(result) == 4
        assert meta["nframes"] == 4

    def test_no_subsampling_matches_default(self, tmp_path):
        """Explicitly disabling all sub-sampling should match default behavior."""
        frames = _make_frames(n=8, height=16, width=16)
        path = tmp_path / "default.bin"
        _write_bin(path, frames)

        video_default, _, _ = read_jpeg_bin(str(path))
        video_explicit, _, _ = read_jpeg_bin(
            str(path), num_frames=None, frame_interval=None
        )
        assert video_default.shape == video_explicit.shape

    def test_num_frames_with_range_and_base64(self, tmp_path):
        """num_frames after range clipping with base64 output."""
        frames = _make_frames(n=20, height=16, width=16)
        path = tmp_path / "combo_b64.bin"
        _write_bin(path, frames)

        result, meta, _ = read_jpeg_bin(
            str(path),
            return_format="base64",
            start_frame=5,
            end_frame=15,
            num_frames=3,
        )
        assert len(result) == 3
        assert meta["nframes"] == 3


# ---------------------------------------------------------------------------
# video_to_jpeg_bin tests (requires a real video file)
# ---------------------------------------------------------------------------
class TestVideoToJpegBin:
    def _make_video_file(self, path, n_frames=30, fps=30.0, width=64, height=48):
        """Create a minimal .avi video file using OpenCV."""
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        if not writer.isOpened():
            pytest.skip("OpenCV VideoWriter not available on this platform")
        rng = np.random.RandomState(7)
        for _ in range(n_frames):
            frame = rng.randint(0, 256, (height, width, 3), dtype=np.uint8)
            writer.write(frame)
        writer.release()

    def test_video_to_bin_roundtrip(self, tmp_path):
        """Convert a video to bin and read it back."""
        video_path = tmp_path / "test.avi"
        self._make_video_file(video_path, n_frames=30, fps=30.0, width=64, height=48)

        bin_path = tmp_path / "output.bin"
        info = video_to_jpeg_bin(
            str(video_path),
            str(bin_path),
            sample_fps=2.0,
            jpeg_quality=95,
        )

        assert info["source_fps"] == pytest.approx(30.0)
        assert info["sample_fps"] == 2.0
        assert info["nframes"] > 0
        assert info["width"] == 64
        assert info["height"] == 48
        assert os.path.exists(str(bin_path))

        video, meta, sample_fps = read_jpeg_bin(str(bin_path))
        assert video.shape[0] == info["nframes"]
        assert video.shape[1] == 48
        assert video.shape[2] == 64
        assert video.shape[3] == 3
        assert sample_fps == 2.0

    def test_video_to_bin_with_max_size(self, tmp_path):
        """max_size should resize frames to fit within the given dimension."""
        video_path = tmp_path / "big.avi"
        self._make_video_file(video_path, n_frames=20, fps=30.0, width=128, height=96)

        bin_path = tmp_path / "small.bin"
        max_size = 48
        info = video_to_jpeg_bin(
            str(video_path),
            str(bin_path),
            sample_fps=2.0,
            max_size=max_size,
        )

        assert max(info["width"], info["height"]) <= max_size

        video, _, _ = read_jpeg_bin(str(bin_path))
        assert max(video.shape[1], video.shape[2]) <= max_size

    def test_video_to_bin_hwaccel_disabled(self, tmp_path):
        """hwaccel=False should work with pure software decoding."""
        video_path = tmp_path / "sw.avi"
        self._make_video_file(video_path, n_frames=15, fps=30.0, width=64, height=48)

        bin_path = tmp_path / "sw.bin"
        info = video_to_jpeg_bin(
            str(video_path),
            str(bin_path),
            sample_fps=2.0,
            hwaccel=False,
        )
        assert info["nframes"] > 0
        assert os.path.exists(str(bin_path))

    def test_video_to_bin_hwaccel_auto(self, tmp_path):
        """hwaccel=None (auto-detect) should not crash."""
        video_path = tmp_path / "auto.avi"
        self._make_video_file(video_path, n_frames=15, fps=30.0, width=64, height=48)

        bin_path = tmp_path / "auto.bin"
        info = video_to_jpeg_bin(
            str(video_path),
            str(bin_path),
            sample_fps=2.0,
            hwaccel=None,
        )
        assert info["nframes"] > 0
        assert os.path.exists(str(bin_path))

    def test_video_to_bin_nonexistent_input(self, tmp_path):
        """A non-existent input video should raise an error."""
        bin_path = tmp_path / "out.bin"
        with pytest.raises((FileNotFoundError, OSError, ValueError)):
            video_to_jpeg_bin("/nonexistent/video.mp4", str(bin_path))
