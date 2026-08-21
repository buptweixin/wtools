#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Read and write JPEGBIN1 video files.

JPEGBIN1 is a custom binary container that stores pre-extracted, JPEG-compressed
video frames together with metadata (FPS, frame indices, resolution, etc.) in a
single ``.bin`` file.  It avoids the need to decode a video codec (e.g. H.264)
at training time -- frames are individually JPEG-encoded and can be decoded with
OpenCV alone.

File layout (all integers little-endian)::

    +-------------------------------+ offset = 0
    |        Header (60 bytes)      |
    +-------------------------------+ offset = 60
    |  frame_indices  (N * 8 bytes) |  uint64[N]
    +-------------------------------+
    |  jpeg_lengths   (N * 8 bytes) |  uint64[N]
    +-------------------------------+ offset = 60 + N * 16
    |      JPEG Payload Section     |  concatenated JPEG byte streams
    |  [jpeg_0][jpeg_1]...[jpeg_N-1]|
    +-------------------------------+
"""

import base64
import os
import struct
import tempfile
from typing import Any, Dict, List, Optional, Tuple, Union, overload

try:
    import av
    from av.codec.hwaccel import HWAccel
except ImportError:  # pragma: no cover
    av = None  # type: ignore[assignment]
    HWAccel = None  # type: ignore[assignment,misc]

import cv2
import numpy as np

MAGIC = b"JPEGBIN1"
FORMAT_VERSION = 1
MAX_NFRAMES = 10_000_000  # Safety limit to prevent OOM from malicious files

# magic, version, nframes, total_num_frames, source_fps, sample_fps,
# selected_duration, width, height, jpeg_quality
HEADER_STRUCT = struct.Struct("<8sIIQdddIII")
#: Size of the header in bytes (60).
HEADER_SIZE = HEADER_STRUCT.size


class JPEGBinError(Exception):
    """Raised when a JPEGBIN1 file is corrupt, truncated, or unsupported."""


# Device-type names recognised by FFmpeg's hardware decoders.
_HWACCEL_DEVICE_MAP: Dict[str, str] = {
    "cuda": "cuda",
    "videotoolbox": "videotoolbox",
    "qsv": "qsv",
    "vaapi": "vaapi",
}


def _get_hwaccel(hwaccel: Union[None, str, bool]) -> Optional[Any]:
    # Returns Optional[Any] because HWAccel may not be importable (av is
    # optional). When av is installed, the return is Optional[HWAccel].
    """Try to create a :class:`av.codec.hwaccel.HWAccel` instance.

    Args:
        hwaccel: ``None`` for auto-detection (tries ``videotoolbox`` on
            macOS), ``False`` to disable hardware decoding entirely, or a
            string such as ``"cuda"``, ``"videotoolbox"``, ``"qsv"``,
            ``"vaapi"`` to request a specific device type.

    Returns:
        An ``HWAccel`` instance, or ``None`` if hardware decoding is
        disabled or unavailable.
    """
    if hwaccel is False:
        return None

    if av is None or HWAccel is None:
        return None

    if hwaccel is None or hwaccel is True:
        # Auto-detect: default to videotoolbox (macOS).  The call will
        # simply fail on other platforms and fall back to software.
        device_type = "videotoolbox"
    elif isinstance(hwaccel, str):
        device_type = _HWACCEL_DEVICE_MAP.get(hwaccel)
        if device_type is None:
            raise ValueError(
                f"Unknown hwaccel type: {hwaccel!r}. "
                f"Valid options: {list(_HWACCEL_DEVICE_MAP)}"
            )
    else:
        return None

    try:
        return HWAccel(device_type=device_type, allow_software_fallback=True)
    except Exception:
        return None


def _read_jpeg_payload(
    fin: Any, offset: int, length: int, bin_path: str, frame_idx: int
) -> bytes:
    """Read a single JPEG payload from an open binary file.

    Args:
        fin: Open binary file handle.
        offset: Byte offset of the payload.
        length: Expected byte length of the payload.
        bin_path: Path to the bin file (for error messages).
        frame_idx: Frame index (for error messages).

    Returns:
        The JPEG payload bytes.

    Raises:
        JPEGBinError: If the payload is truncated.
    """
    fin.seek(offset)
    payload = fin.read(length)
    if len(payload) != length:
        raise JPEGBinError(
            f"Truncated JPEG payload at frame {frame_idx} in {bin_path!r}: "
            f"expected {length} bytes, got {len(payload)}"
        )
    return payload


def write_jpeg_bin(
    output_path: str,
    frames: List[np.ndarray],
    source_fps: float,
    sample_fps: float,
    total_num_frames: int,
    frame_indices: Optional[List[int]] = None,
    selected_duration: Optional[float] = None,
    jpeg_quality: int = 95,
) -> None:
    """Write a list of frames to a JPEGBIN1 ``.bin`` file.

    Each frame is JPEG-encoded and packed together with a 60-byte header and
    two uint64 index arrays (frame indices and JPEG byte lengths).

    Args:
        output_path: Destination file path.
        frames: List of ``np.ndarray`` images in BGR format (as produced by
            OpenCV).  All frames must have the same height and width.
        source_fps: Frame rate of the original source video.
        sample_fps: Sampling frame rate used to extract *frames*.
        total_num_frames: Total number of frames in the original source video.
        frame_indices: Original frame indices for each entry in *frames*.
            If ``None``, defaults to ``range(len(frames))``.
        selected_duration: Duration of the selected segment in seconds.
            If ``None``, defaults to ``total_num_frames / source_fps``.
        jpeg_quality: JPEG encoding quality (1-100).  Higher values produce
            larger files with less compression artifacts.

    Raises:
        ValueError: If *frames* is empty or frames have inconsistent shapes.
        JPEGBinError: If JPEG encoding fails for any frame.

    Examples:
        >>> import numpy as np
        >>> frames = [np.zeros((112, 112, 3), dtype=np.uint8) for _ in range(10)]
        >>> write_jpeg_bin("video.bin", frames, source_fps=30.0,
        ...               sample_fps=2.0, total_num_frames=300)
    """
    nframes = len(frames)
    if nframes == 0:
        raise ValueError("frames must not be empty")

    height, width = frames[0].shape[:2]
    for i, f in enumerate(frames):
        if f.shape[:2] != (height, width):
            raise ValueError(
                f"Frame {i} has shape {f.shape[:2]}, expected ({height}, {width})"
            )

    if frame_indices is None:
        frame_indices = list(range(nframes))
    if len(frame_indices) != nframes:
        raise ValueError(
            f"frame_indices has {len(frame_indices)} entries, expected {nframes}"
        )

    if selected_duration is None:
        if source_fps == 0:
            raise ValueError("source_fps must not be zero when selected_duration is None")
        selected_duration = total_num_frames / source_fps

    # JPEG-encode all frames
    jpeg_data_list: List[bytes] = []
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    for i, frame in enumerate(frames):
        ok, buf = cv2.imencode(".jpg", frame, encode_param)
        if not ok:
            raise JPEGBinError(f"JPEG encoding failed for frame {i}")
        jpeg_data_list.append(buf.tobytes())

    jpeg_lengths = [len(j) for j in jpeg_data_list]

    # Build header
    header = HEADER_STRUCT.pack(
        MAGIC,
        FORMAT_VERSION,
        nframes,
        total_num_frames,
        float(source_fps),
        float(sample_fps),
        float(selected_duration),
        width,
        height,
        int(jpeg_quality),
    )

    # Build index arrays
    indices_array = np.array(frame_indices, dtype="<u8").tobytes()
    lengths_array = np.array(jpeg_lengths, dtype="<u8").tobytes()

    # Write file atomically: write to a temp file then os.replace()
    dir_name = os.path.dirname(os.path.abspath(output_path))
    with tempfile.NamedTemporaryFile(dir=dir_name, delete=False, suffix=".tmp") as ftmp:
        ftmp.write(header)
        ftmp.write(indices_array)
        ftmp.write(lengths_array)
        for jd in jpeg_data_list:
            ftmp.write(jd)
        tmp_path = ftmp.name
    os.replace(tmp_path, output_path)


def video_to_jpeg_bin(
    input_path: str,
    output_path: str,
    sample_fps: float = 2.0,
    jpeg_quality: int = 95,
    max_size: Optional[int] = None,
    hwaccel: Union[None, str, bool] = None,
) -> Dict[str, Any]:
    """Extract frames from a video file and write a JPEGBIN1 ``.bin`` file.

    Opens *input_path* with PyAV (FFmpeg), samples frames at *sample_fps*,
    optionally resizes so the longest side does not exceed *max_size*,
    JPEG-encodes each frame, and writes the result to *output_path*.

    Args:
        input_path: Path to the input video (any format supported by
            FFmpeg).
        output_path: Destination ``.bin`` file path.
        sample_fps: Target sampling frame rate.  Frames are picked every
            ``round(source_fps / sample_fps)`` frames.
        jpeg_quality: JPEG encoding quality (1-100).
        max_size: If not ``None``, frames whose longest side exceeds this
            value are resized proportionally.
        hwaccel: Hardware decoding configuration.

            - ``None`` (default) -- auto-detect (tries ``videotoolbox``
              on macOS, falls back to software decoding).
            - ``False`` -- disable hardware decoding entirely.
            - ``"cuda"``, ``"videotoolbox"``, ``"qsv"``, ``"vaapi"`` --
              request a specific hardware decoder.

    Returns:
        A dictionary with keys: ``nframes``, ``width``, ``height``,
        ``source_fps``, ``sample_fps``, ``total_num_frames``,
        ``file_size``.

    Raises:
        ValueError: If the video cannot be opened or contains no frames.
        JPEGBinError: If JPEG encoding fails.

    Examples:
        >>> meta = video_to_jpeg_bin("input.mp4", "output.bin",
        ...                         sample_fps=2.0, jpeg_quality=95,
        ...                         max_size=448)
        >>> print(meta["nframes"], meta["width"], meta["height"])
    """
    if av is None:
        raise ImportError(
            "PyAV (av) is required for video_to_jpeg_bin. "
            "Install it with: pip install av"
        )

    hwaccel_obj = _get_hwaccel(hwaccel)

    container = av.open(input_path)
    try:
        if not container.streams.video:
            raise ValueError(f"No video streams found in {input_path!r}")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"

        source_fps = float(stream.average_rate) if stream.average_rate else 0.0
        if source_fps <= 0:
            raise ValueError(f"Invalid source FPS: {source_fps}")

        if sample_fps <= 0:
            raise ValueError(f"sample_fps must be > 0, got {sample_fps}")

        total_num_frames = stream.frames or 0
        if total_num_frames == 0:
            duration = (
                float(stream.duration * stream.time_base) if stream.duration else 0.0
            )
            if duration > 0 and source_fps > 0:
                total_num_frames = int(duration * source_fps)

        frame_interval = max(1, round(source_fps / sample_fps))

        frames: List[np.ndarray] = []
        frame_indices: List[int] = []
        width = height = 0

        decode_kwargs: Dict[str, Any] = {}
        if hwaccel_obj is not None:
            decode_kwargs["hwaccel"] = hwaccel_obj

        frame_idx = 0
        for frame in container.decode(stream):
            if frame_idx % frame_interval == 0:
                img = frame.to_ndarray(format="bgr24")

                if max_size is not None and max(img.shape[:2]) > max_size:
                    scale = max_size / max(img.shape[:2])
                    new_w = int(img.shape[1] * scale)
                    new_h = int(img.shape[0] * scale)
                    img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

                if width == 0:
                    height, width = img.shape[:2]

                frames.append(img)
                frame_indices.append(frame_idx)

            frame_idx += 1
    finally:
        container.close()

    if not frames:
        raise ValueError(f"No frames extracted from {input_path!r}")

    write_jpeg_bin(
        output_path=output_path,
        frames=frames,
        source_fps=float(source_fps),
        sample_fps=sample_fps,
        total_num_frames=total_num_frames,
        frame_indices=frame_indices,
        jpeg_quality=jpeg_quality,
    )

    file_size = os.path.getsize(output_path)
    return {
        "nframes": len(frames),
        "width": width,
        "height": height,
        "source_fps": float(source_fps),
        "sample_fps": sample_fps,
        "total_num_frames": total_num_frames,
        "file_size": file_size,
    }


def read_jpeg_bin_metadata(bin_path: str, validate_size: bool = True) -> Dict[str, Any]:
    """Read metadata from a JPEGBIN1 file without decoding JPEG frames.

    Only the 60-byte header and the two uint64 index arrays are read; the
    JPEG payloads are skipped entirely.  This is useful for quickly inspecting
    a file's properties or validating integrity.

    Args:
        bin_path: Path to the ``.bin`` file.
        validate_size: If ``True``, verify that the file size matches the
            expected size computed from the header and index arrays.

    Returns:
        A dictionary with the following keys:

        - ``source_fps`` (float): Original source video FPS.
        - ``frame_indices`` (list[int]): Original frame indices stored.
        - ``total_num_frames`` (int): Total frames in the source video.
        - ``video_backend`` (str): Always ``"jpeg_bin"``.
        - ``sample_fps`` (float): Sampling FPS used during extraction.
        - ``selected_duration`` (float): Duration of the segment (seconds).
        - ``width`` (int): Frame width in pixels.
        - ``height`` (int): Frame height in pixels.
        - ``jpeg_quality`` (int): JPEG quality used during encoding.
        - ``nframes`` (int): Number of frames stored.
        - ``jpeg_lengths`` (list[int]): Byte length of each JPEG payload.
        - ``payload_offset`` (int): Byte offset where JPEG data starts.

    Raises:
        JPEGBinError: If the file is truncated, has an invalid magic bytes,
            unsupported version, or (when *validate_size* is ``True``) a
            size mismatch.
        FileNotFoundError: If *bin_path* does not exist.

    Examples:
        >>> meta = read_jpeg_bin_metadata("video.bin")
        >>> print(f"{meta['nframes']} frames, {meta['width']}x{meta['height']}")
    """
    with open(bin_path, "rb") as fin:
        raw_header = fin.read(HEADER_SIZE)
        if len(raw_header) != HEADER_SIZE:
            raise JPEGBinError(f"Truncated bin header: {bin_path!r}")

        (
            magic,
            version,
            nframes,
            total_num_frames,
            source_fps,
            sample_fps,
            selected_duration,
            width,
            height,
            jpeg_quality,
        ) = HEADER_STRUCT.unpack(raw_header)

        if magic != MAGIC:
            raise JPEGBinError(
                f"Invalid magic in {bin_path!r}: {magic!r} (expected {MAGIC!r})"
            )
        if version != FORMAT_VERSION:
            raise JPEGBinError(
                f"Unsupported bin version in {bin_path!r}: "
                f"{version} != {FORMAT_VERSION}"
            )
        if nframes <= 0:
            raise JPEGBinError(f"Invalid nframes in {bin_path!r}: {nframes}")
        if nframes > MAX_NFRAMES:
            raise JPEGBinError(
                f"nframes={nframes} exceeds safety limit {MAX_NFRAMES} in {bin_path!r}"
            )
        # Verify file is large enough to contain the declared index tables
        min_expected = HEADER_SIZE + nframes * 16
        if os.path.getsize(bin_path) < min_expected:
            raise JPEGBinError(
                f"File too small for {nframes} frames in {bin_path!r}: "
                f"need at least {min_expected} bytes"
            )

        indices_raw = fin.read(nframes * 8)
        lengths_raw = fin.read(nframes * 8)
        if len(indices_raw) != nframes * 8 or len(lengths_raw) != nframes * 8:
            raise JPEGBinError(f"Truncated bin index table: {bin_path!r}")

        frame_indices = np.frombuffer(indices_raw, dtype="<u8").tolist()
        jpeg_lengths = np.frombuffer(lengths_raw, dtype="<u8").tolist()
        payload_offset = HEADER_SIZE + nframes * 16

    if validate_size:
        expected_size = payload_offset + sum(jpeg_lengths)
        actual_size = os.path.getsize(bin_path)
        if actual_size != expected_size:
            raise JPEGBinError(
                f"Corrupt bin size for {bin_path!r}: "
                f"actual={actual_size}, expected={expected_size}"
            )

    return {
        "source_fps": float(source_fps),
        "frame_indices": frame_indices,
        "total_num_frames": int(total_num_frames),
        "video_backend": "jpeg_bin",
        "sample_fps": float(sample_fps),
        "selected_duration": float(selected_duration),
        "width": int(width),
        "height": int(height),
        "jpeg_quality": int(jpeg_quality),
        "nframes": int(nframes),
        "jpeg_lengths": jpeg_lengths,
        "payload_offset": payload_offset,
    }


@overload
def read_jpeg_bin(
    bin_path: str,
    return_format: str = "numpy",
    num_frames: Optional[int] = None,
    frame_interval: Optional[int] = None,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
) -> Tuple[np.ndarray, Dict[str, Any], float]:
    ...


@overload
def read_jpeg_bin(
    bin_path: str,
    *,
    return_format: str = "base64",
    num_frames: Optional[int] = None,
    frame_interval: Optional[int] = None,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
) -> Tuple[List[str], Dict[str, Any], float]:
    ...


def read_jpeg_bin(
    bin_path: str,
    return_format: str = "numpy",
    num_frames: Optional[int] = None,
    frame_interval: Optional[int] = None,
    start_frame: Optional[int] = None,
    end_frame: Optional[int] = None,
) -> Tuple[Union[np.ndarray, List[str]], Dict[str, Any], float]:
    """Decode a JPEGBIN1 file into frames, with optional sub-sampling.

    Reads the header and index tables via :func:`read_jpeg_bin_metadata`,
    then either decodes each JPEG payload into an RGB NumPy array or
    returns the raw JPEG bytes as base64-encoded strings.

    Sub-sampling is controlled by three mutually-composable parameters
    (applied in the order: *start_frame/end_frame* clipping ->
    *frame_interval* stride -> *num_frames* uniform re-sample):

    Args:
        bin_path: Path to the ``.bin`` file.
        return_format: Output format for the frames:

            - ``"numpy"`` (default) -- Decode each JPEG into an RGB
              ``np.ndarray`` of shape ``(T, H, W, 3)`` with dtype
              ``uint8``.
            - ``"base64"`` -- Return raw JPEG payloads as a list of
              base64-encoded ``str`` objects, one per frame.  No image
              decoding is performed, making this faster and free of
              OpenCV decode overhead.

        num_frames: If given, uniformly sub-sample to exactly this many
            frames via ``np.linspace``.  Must be >= 1 and <= the number
            of frames available after any *start_frame*/*end_frame*
            clipping and *frame_interval* stride.
        frame_interval: If given, take every *frame_interval*-th frame
            (1 = all frames, 2 = every other, etc.).  Applied after
            range clipping but before *num_frames* re-sampling.
        start_frame: Zero-based index of the first frame to include
            (in the bin file's frame ordering, not the original video).
            Defaults to 0.
        end_frame: Zero-based index **after** the last frame to include
            (Python slice semantics).  Defaults to the total frame
            count (i.e. include all remaining frames).

    Returns:
        A tuple ``(video, metadata, sample_fps)`` where:

        - **video** is either a ``np.ndarray`` of shape ``(T, H, W, 3)``
          (when *return_format* is ``"numpy"``) or a ``list[str]`` of
          base64-encoded JPEG strings (when *return_format* is
          ``"base64"``).  ``T`` equals the number of selected frames
          after sub-sampling.
        - **metadata** is the dictionary from
          :func:`read_jpeg_bin_metadata` minus the ``jpeg_lengths`` and
          ``payload_offset`` keys, with ``nframes`` and
          ``frame_indices`` updated to reflect the sub-sampled
          selection.
        - **sample_fps** is adjusted to reflect the effective frame
          rate after sub-sampling.

    Raises:
        JPEGBinError: If the file is corrupt or a JPEG payload cannot be
            decoded (only applies to ``"numpy"`` mode).
        ValueError: If *return_format* is invalid, *num_frames* is out
            of range, or *frame_interval* is < 1.

    Examples:
        >>> # Default: decode all frames to NumPy array
        >>> video, meta, fps = read_jpeg_bin("video.bin")
        >>> video.shape        # (T, H, W, 3)

        >>> # Uniformly sample 16 frames
        >>> video, meta, fps = read_jpeg_bin("video.bin", num_frames=16)

        >>> # Take every 3rd frame
        >>> video, meta, fps = read_jpeg_bin("video.bin", frame_interval=3)

        >>> # Frames 10..50 only, as base64 strings
        >>> b64, meta, fps = read_jpeg_bin("video.bin", return_format="base64",
        ...                                start_frame=10, end_frame=50)
    """
    if return_format not in ("numpy", "base64"):
        raise ValueError(
            f"return_format must be 'numpy' or 'base64', got {return_format!r}"
        )

    metadata = read_jpeg_bin_metadata(bin_path, validate_size=True)
    jpeg_lengths = metadata.pop("jpeg_lengths")
    payload_offset = metadata.pop("payload_offset")
    total_stored = metadata["nframes"]

    # --- determine which stored-frame indices to read ---
    s = 0 if start_frame is None else max(0, start_frame)
    e = total_stored if end_frame is None else min(end_frame, total_stored)
    if s >= e:
        raise ValueError(
            f"Empty frame range: start_frame={s} >= end_frame={e} "
            f"(total stored={total_stored})"
        )

    sel = list(range(s, e))

    if frame_interval is not None:
        if frame_interval < 1:
            raise ValueError(f"frame_interval must be >= 1, got {frame_interval}")
        sel = sel[::frame_interval]

    if num_frames is not None:
        if num_frames < 1:
            raise ValueError(f"num_frames must be >= 1, got {num_frames}")
        if num_frames > len(sel):
            raise ValueError(
                f"num_frames={num_frames} exceeds available frames "
                f"({len(sel)}) after clipping/striding"
            )
        if num_frames < len(sel):
            positions = np.linspace(0, len(sel) - 1, num=num_frames, dtype=int)
            sel = [sel[i] for i in positions]

    # --- read selected JPEG payloads ---
    # Build cumulative offset table via np.cumsum (vectorized)
    # offsets[i] = payload_offset + sum(jpeg_lengths[:i])
    cumlen = np.empty(len(jpeg_lengths) + 1, dtype=np.int64)
    cumlen[0] = 0
    np.cumsum(jpeg_lengths, out=cumlen[1:], dtype=np.int64)
    offsets = cumlen + payload_offset

    if return_format == "base64":
        result: List[str] = []
        with open(bin_path, "rb") as fin:
            for idx in sel:
                payload = _read_jpeg_payload(
                    fin, offsets[idx], jpeg_lengths[idx], bin_path, idx
                )
                result.append(base64.b64encode(payload).decode("ascii"))
    else:
        frames: List[np.ndarray] = []
        with open(bin_path, "rb") as fin:
            for idx in sel:
                payload = _read_jpeg_payload(
                    fin, offsets[idx], jpeg_lengths[idx], bin_path, idx
                )
                bgr = cv2.imdecode(
                    np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if bgr is None:
                    raise JPEGBinError(
                        f"JPEG decode failed at frame {idx} in {bin_path!r}"
                    )
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                frames.append(rgb)
        result = np.stack(frames, axis=0)

    # --- update metadata to reflect the sub-sampled selection ---
    old_indices = metadata["frame_indices"]
    metadata["frame_indices"] = [old_indices[i] for i in sel]
    metadata["nframes"] = len(sel)

    # Determine sample_fps: if no sub-sampling was requested, keep the
    # stored value; otherwise recalculate from the effective frame count.
    sub_sampled = (
        num_frames is not None
        or frame_interval is not None
        or start_frame is not None
        or end_frame is not None
    )
    if sub_sampled:
        original_fps = metadata["source_fps"]
        total_num_frames = metadata["total_num_frames"]
        if total_num_frames > 0 and original_fps > 0:
            video_duration = total_num_frames / original_fps
            sample_fps = len(sel) / video_duration if video_duration > 0 else 0.0
        else:
            sample_fps = metadata["sample_fps"]
    else:
        sample_fps = metadata["sample_fps"]

    return result, metadata, sample_fps
