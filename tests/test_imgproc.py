"""Tests for wtools.utils.imgproc -- safe_crop and get_image_size."""

import struct

import numpy as np
import pytest

from wtools.utils.imgproc import (
    UnknownImageFormat,
    get_image_size,
    img2str,
    safe_crop,
    str2img,
)


# ---------------------------------------------------------------------------
# safe_crop tests
# ---------------------------------------------------------------------------
class TestSafeCrop:
    def test_normal_crop_2d(self):
        """Crop a fully in-bounds region from a 2D grayscale image."""
        img = np.arange(100, dtype=np.uint8).reshape(10, 10)
        # Crop from (2, 3) with size (4, 5) => rows 3..7, cols 2..6
        result = safe_crop(img, [2, 3, 4, 5])
        assert result.shape == (5, 4)
        expected = img[3:8, 2:6]
        np.testing.assert_array_equal(result, expected)

    def test_normal_crop_3d(self):
        """Crop a fully in-bounds region from a 3D multi-channel image."""
        img = np.ones((10, 10, 3), dtype=np.uint8) * 128
        result = safe_crop(img, [1, 1, 5, 5])
        assert result.shape == (5, 5, 3)
        np.testing.assert_array_equal(result, img[1:6, 1:6])

    def test_out_of_bounds_negative_xy_2d(self):
        """Crop with negative x/y: out-of-bounds area should be zeros."""
        img = np.ones((10, 10), dtype=np.uint8) * 100
        # Crop starting at x=-3, y=-2, size 5x5
        # The visible part: x=0..1 (2 cols), y=0..2 (3 rows) go to
        # result[2:5, 3:5]; rest is zeros.
        result = safe_crop(img, [-3, -2, 5, 5])
        assert result.shape == (5, 5)
        # Check that the in-bounds area has value 100
        assert np.all(result[2:5, 3:5] == 100)
        # Check that the out-of-bounds area is zero
        assert np.all(result[:2, :] == 0)
        assert np.all(result[:, :3] == 0)

    def test_out_of_bounds_positive_overflow_2d(self):
        """Crop extends past the right and bottom edges of the image."""
        img = np.ones((10, 10), dtype=np.uint8) * 50
        # Crop starting at x=8, y=8, size 5x5 -> only 2x2 visible
        result = safe_crop(img, [8, 8, 5, 5])
        assert result.shape == (5, 5)
        # Visible region: result[0:2, 0:2] should be 50
        assert np.all(result[:2, :2] == 50)
        # Out-of-bounds should be zero
        assert np.all(result[2:, :] == 0)
        assert np.all(result[:, 2:] == 0)

    def test_out_of_bounds_negative_and_positive_3d(self):
        """Crop of a 3D image that is out-of-bounds on all sides."""
        img = np.ones((10, 10, 3), dtype=np.uint8) * 200
        # x=-2, y=-2, w=14, h=14 -> visible region is the full 10x10 image
        # placed inside a 14x14 canvas at [2:12, 2:12]
        result = safe_crop(img, [-2, -2, 14, 14])
        assert result.shape == (14, 14, 3)
        # Visible area has value 200
        assert np.all(result[2:12, 2:12, :] == 200)
        # Border is zeros
        assert np.all(result[:2, :, :] == 0)
        assert np.all(result[12:, :, :] == 0)
        assert np.all(result[:, :2, :] == 0)
        assert np.all(result[:, 12:, :] == 0)

    def test_crop_completely_out_of_bounds(self):
        """A crop box entirely outside the image should return all zeros."""
        img = np.ones((10, 10), dtype=np.uint8) * 99
        result = safe_crop(img, [100, 100, 5, 5])
        assert result.shape == (5, 5)
        assert np.all(result == 0)

    def test_crop_full_image_2d(self):
        """Crop the entire image."""
        img = np.arange(25, dtype=np.uint8).reshape(5, 5)
        result = safe_crop(img, [0, 0, 5, 5])
        assert result.shape == (5, 5)
        np.testing.assert_array_equal(result, img)

    def test_crop_full_image_3d(self):
        """Crop the entire 3D image."""
        img = np.arange(75, dtype=np.uint8).reshape(5, 5, 3)
        result = safe_crop(img, [0, 0, 5, 5])
        assert result.shape == (5, 5, 3)
        np.testing.assert_array_equal(result, img)

    def test_crop_returns_uint8(self):
        """Result should always be np.uint8."""
        img = np.ones((10, 10, 3), dtype=np.float32) * 0.5
        result = safe_crop(img, [0, 0, 3, 3])
        assert result.dtype == np.uint8

    def test_crop_with_float_box_values(self):
        """Float values in crop_box should be accepted (cast to int)."""
        img = np.ones((10, 10), dtype=np.uint8) * 7
        result = safe_crop(img, [1.9, 1.9, 3.0, 3.0])
        assert result.shape == (3, 3)
        np.testing.assert_array_equal(result, img[1:4, 1:4])


# ---------------------------------------------------------------------------
# get_image_size tests
# ---------------------------------------------------------------------------
class TestGetImageSize:
    def _write_gif(self, path, width, height):
        """Write a minimal valid GIF file with the given dimensions."""
        with open(path, "wb") as f:
            # GIF87a header
            f.write(b"GIF87a")
            # Logical screen width and height (little-endian uint16)
            f.write(struct.pack("<HH", width, height))
            # GCT flag + background + aspect (minimal: global color table with 2 colors)
            f.write(b"\x80\x00\x00")
            # Global color table (2 entries x 3 bytes)
            f.write(b"\x00\x00\x00")  # color 0: black
            f.write(b"\xff\xff\xff")  # color 1: white
            # Image descriptor
            f.write(b"\x2c")  # Image separator
            f.write(struct.pack("<HH", 0, 0))  # left, top position
            f.write(struct.pack("<HH", width, height))  # image width, height
            f.write(b"\x00")  # no local color table
            # Image data (minimal: 1 sub-block with LZW minimum code size + terminator)
            f.write(b"\x02")  # LZW minimum code size
            f.write(b"\x02\x4c\x01")  # sub-block: 2 bytes of data
            f.write(b"\x00")  # block terminator
            # Trailer
            f.write(b"\x3b")

    def _write_gif89a(self, path, width, height):
        """Write a minimal GIF89a file."""
        with open(path, "wb") as f:
            f.write(b"GIF89a")
            f.write(struct.pack("<HH", width, height))
            f.write(b"\x80\x00\x00")
            f.write(b"\x00\x00\x00")
            f.write(b"\xff\xff\xff")
            f.write(b"\x2c")
            f.write(struct.pack("<HH", 0, 0))
            f.write(struct.pack("<HH", width, height))
            f.write(b"\x00")
            f.write(b"\x02")
            f.write(b"\x02\x4c\x01")
            f.write(b"\x00")
            f.write(b"\x3b")

    def _write_png(self, path, width, height):
        """Write a minimal valid PNG file with the given dimensions."""
        import zlib

        def _png_chunk(chunk_type, data):
            chunk = chunk_type + data
            crc = struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)
            return struct.pack(">I", len(data)) + chunk + crc

        with open(path, "wb") as f:
            # PNG signature
            f.write(b"\x89PNG\r\n\x1a\n")
            # IHDR chunk: width, height, bit depth=8, color type=2 (RGB),
            # compression=0, filter=0, interlace=0
            ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
            f.write(_png_chunk(b"IHDR", ihdr_data))
            # IDAT chunk: raw image data (filtered)
            raw = b""
            for _y in range(height):
                raw += b"\x00"  # filter byte: None
                raw += b"\x00\x00\x00" * width  # RGB pixels: all black
            compressed = zlib.compress(raw)
            f.write(_png_chunk(b"IDAT", compressed))
            # IEND
            f.write(_png_chunk(b"IEND", b""))

    def _write_jpeg(self, path, width, height):
        """Write a minimal JPEG file with the given dimensions."""
        with open(path, "wb") as f:
            # SOI (Start of Image)
            f.write(b"\xff\xd8")
            # APP0 marker (JFIF)
            app0_data = b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
            f.write(b"\xff\xe0")
            f.write(struct.pack(">H", len(app0_data) + 2))
            f.write(app0_data)
            # DQT marker (Define Quantization Table) - minimal 8-bit table
            dqt_data = b"\x00" + bytes(range(64))
            f.write(b"\xff\xdb")
            f.write(struct.pack(">H", len(dqt_data) + 2))
            f.write(dqt_data)
            # SOF0 marker (Start of Frame, Baseline DCT)
            # precision=8, height, width, num_components=1, component_id=1,
            # sampling=0x11, quant_table_id=0
            sof_data = struct.pack(">BHH", 8, height, width) + b"\x01\x01\x11\x00"
            f.write(b"\xff\xc0")
            f.write(struct.pack(">H", len(sof_data) + 2))
            f.write(sof_data)
            # DHT marker (Define Huffman Table) - minimal DC table
            dht_data = b"\x00" + bytes(16) + b"\x00"
            f.write(b"\xff\xc4")
            f.write(struct.pack(">H", len(dht_data) + 2))
            f.write(dht_data)
            # SOS marker (Start of Scan)
            sos_data = b"\x01\x01\x00\x00\x3f\x00"
            f.write(b"\xff\xda")
            f.write(struct.pack(">H", len(sos_data) + 2))
            f.write(sos_data)
            # Minimal scan data + EOI
            f.write(b"\x00" * 4)
            f.write(b"\xff\xd9")

    def test_gif87a(self, tmp_path):
        path = tmp_path / "test.gif"
        self._write_gif(str(path), 64, 48)
        w, h = get_image_size(str(path))
        assert (w, h) == (64, 48)

    def test_gif89a(self, tmp_path):
        path = tmp_path / "test89a.gif"
        self._write_gif89a(str(path), 320, 240)
        w, h = get_image_size(str(path))
        assert (w, h) == (320, 240)

    def test_png(self, tmp_path):
        path = tmp_path / "test.png"
        self._write_png(str(path), 128, 96)
        w, h = get_image_size(str(path))
        assert (w, h) == (128, 96)

    def test_png_square(self, tmp_path):
        path = tmp_path / "square.png"
        self._write_png(str(path), 1, 1)
        w, h = get_image_size(str(path))
        assert (w, h) == (1, 1)

    def test_jpeg(self, tmp_path):
        path = tmp_path / "test.jpg"
        self._write_jpeg(str(path), 100, 200)
        w, h = get_image_size(str(path))
        assert (w, h) == (100, 200)

    def test_jpeg_small(self, tmp_path):
        path = tmp_path / "small.jpg"
        self._write_jpeg(str(path), 16, 16)
        w, h = get_image_size(str(path))
        assert (w, h) == (16, 16)

    def test_unknown_format_raises(self, tmp_path):
        path = tmp_path / "unknown.bin"
        with open(path, "wb") as f:
            f.write(b"\x00" * 50)
        with pytest.raises(UnknownImageFormat):
            get_image_size(str(path))

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            get_image_size("/nonexistent/path/file.png")

    def test_gif_round_trip_with_real_dims(self, tmp_path):
        """Verify consistency: write GIF with known dims and read back."""
        for w, h in [(1, 1), (100, 50), (1920, 1080)]:
            path = tmp_path / f"gif_{w}x{h}.gif"
            self._write_gif(str(path), w, h)
            rw, rh = get_image_size(str(path))
            assert (rw, rh) == (w, h)

    def test_png_round_trip_with_real_dims(self, tmp_path):
        """Verify consistency: write PNG with known dims and read back."""
        for w, h in [(1, 1), (100, 50), (640, 480)]:
            path = tmp_path / f"png_{w}x{h}.png"
            self._write_png(str(path), w, h)
            rw, rh = get_image_size(str(path))
            assert (rw, rh) == (w, h)

    def test_jpeg_round_trip_with_real_dims(self, tmp_path):
        """Verify consistency: write JPEG with known dims and read back."""
        for w, h in [(32, 32), (128, 64), (256, 256)]:
            path = tmp_path / f"jpg_{w}x{h}.jpg"
            self._write_jpeg(str(path), w, h)
            rw, rh = get_image_size(str(path))
            assert (rw, rh) == (w, h)


# ---------------------------------------------------------------------------
# img2str / str2img round-trip tests
# ---------------------------------------------------------------------------
class TestImgEncodeDecode:
    def test_roundtrip_3d(self):
        """Encode an image to JPEG bytes and decode back; dimensions match."""
        img = np.random.randint(0, 256, (50, 60, 3), dtype=np.uint8)
        encoded = img2str(img)
        assert isinstance(encoded, bytes)
        decoded = str2img(encoded)
        assert decoded is not None
        assert decoded.shape[0] == 50
        assert decoded.shape[1] == 60
        assert decoded.shape[2] == 3

    def test_roundtrip_grayscale(self):
        """Encode a grayscale image and decode back."""
        img = np.random.randint(0, 256, (40, 40), dtype=np.uint8)
        encoded = img2str(img)
        decoded = str2img(encoded)
        assert decoded is not None
        assert decoded.shape[0] == 40
        assert decoded.shape[1] == 40
