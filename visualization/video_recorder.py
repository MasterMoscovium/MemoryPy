"""
Video recorder — captures simulation frames to GIF or MP4.

Works by collecting pygame screen captures or matplotlib figures
and encoding them with imageio.
"""

import numpy as np
import os
from typing import Optional, List


class VideoRecorder:
    """
    Records frames from the simulation and saves as GIF or MP4.

    Usage:
        recorder = VideoRecorder("output.gif", fps=15)
        for frame in simulation:
            recorder.add_frame(frame)  # numpy array (H, W, 3)
        recorder.save()
    """

    def __init__(self, output_path: str, fps: int = 15,
                 max_frames: int = 2000):
        """
        Args:
            output_path: Path to save the video (.gif or .mp4).
            fps: Frames per second.
            max_frames: Maximum frames to store (prevents OOM).
        """
        self.output_path = output_path
        self.fps = fps
        self.max_frames = max_frames
        self.frames: List[np.ndarray] = []
        self._frame_count = 0

    def add_frame(self, frame: np.ndarray):
        """
        Add a frame to the recording.

        Args:
            frame: RGB numpy array of shape (H, W, 3), dtype uint8.
        """
        if self._frame_count >= self.max_frames:
            return

        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        self.frames.append(frame)
        self._frame_count += 1

    def add_frame_skip(self, frame: np.ndarray, interval: int = 5):
        """
        Add a frame, but only keep every Nth frame (for size reduction).

        Args:
            frame: RGB numpy array.
            interval: Keep every Nth frame.
        """
        self._frame_count += 1
        if self._frame_count % interval == 0:
            self.add_frame(frame)

    def save(self) -> Optional[str]:
        """
        Save the recorded frames to disk.

        Returns:
            Path to the saved file, or None if no frames.
        """
        if not self.frames:
            print("⚠️  No frames to save")
            return None

        try:
            import imageio.v3 as iio
        except ImportError:
            try:
                import imageio as iio
            except ImportError:
                print("❌ imageio not installed. Cannot save video.")
                return None

        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)

        ext = os.path.splitext(self.output_path)[1].lower()

        if ext == ".gif":
            duration = 1000.0 / self.fps  # ms per frame
            iio.imwrite(
                self.output_path,
                self.frames,
                duration=duration,
                loop=0,
            )
        elif ext in (".mp4", ".avi", ".mov"):
            iio.imwrite(
                self.output_path,
                self.frames,
                fps=self.fps,
            )
        else:
            # Default to GIF
            self.output_path += ".gif"
            iio.imwrite(
                self.output_path,
                self.frames,
                duration=1000.0 / self.fps,
                loop=0,
            )

        size_mb = os.path.getsize(self.output_path) / (1024 * 1024)
        print(f"🎬 Video saved: {self.output_path} "
              f"({len(self.frames)} frames, {size_mb:.1f}MB)")

        return self.output_path

    def clear(self):
        """Clear all recorded frames."""
        self.frames.clear()
        self._frame_count = 0

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    def __repr__(self) -> str:
        return (f"VideoRecorder(path='{self.output_path}', "
                f"frames={len(self.frames)}, fps={self.fps})")
