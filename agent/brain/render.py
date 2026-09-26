"""RENDER — screens as PNG images for a vision model (play mode). Pure Python + zlib: no image library needed.

    FramePainter.png(grid, mark=(x, y))   the 64x64 screen scaled up, thin lines every 8 cells, a ruler with cell
                                          numbers (0, 8, ..., 56) on the top and left edges so the model can read
                                          click coordinates, and an optional ring around the last clicked cell
Colours are the ARC palette the games use.
"""
from __future__ import annotations

import struct
import zlib

import numpy as np

PALETTE = np.array([(255, 255, 255), (204, 204, 204), (153, 153, 153), (102, 102, 102), (51, 51, 51), (0, 0, 0),
                    (229, 58, 163), (255, 123, 204), (249, 60, 49), (30, 147, 255), (136, 216, 241),
                    (255, 220, 0), (255, 133, 27), (146, 18, 49), (79, 204, 48), (163, 86, 214)], dtype=np.uint8)
MARGIN = 16  # pixels of ruler above and left of the screen
RULER_EVERY = 8  # cells between ruler numbers / grid lines
DIGITS = {  # 3x5 pixel font
    "0": "111101101101111", "1": "010110010010111", "2": "111001111100111", "3": "111001111001111",
    "4": "101101111001001", "5": "111100111001111", "6": "111100111101111", "7": "111001001001001",
    "8": "111101111101111", "9": "111101111001111",
}


class FramePainter:
    """Grid -> PNG bytes (a utility class)."""

    @staticmethod
    def png(grid: np.ndarray, scale: int = 8, mark: tuple[int, int] | None = None) -> bytes:
        return FramePainter.encode(FramePainter.pixels(grid, scale, mark))

    @staticmethod
    def pixels(grid: np.ndarray, scale: int, mark: tuple[int, int] | None = None) -> np.ndarray:
        """RGB image: ruler margin + the screen, `scale` pixels per cell."""
        h, w = grid.shape
        colours = PALETTE[np.clip(grid, 0, 15)]
        screen = colours.repeat(scale, axis=0).repeat(scale, axis=1)
        image = np.full((MARGIN + h * scale, MARGIN + w * scale, 3), 235, dtype=np.uint8)
        image[MARGIN:, MARGIN:] = screen
        for cell in range(RULER_EVERY, max(h, w), RULER_EVERY):  # faint lines every 8 cells
            p = MARGIN + cell * scale
            if cell < h:
                image[p, MARGIN:] = image[p, MARGIN:] // 2 + 64
            if cell < w:
                image[MARGIN:, p] = image[MARGIN:, p] // 2 + 64
        for cell in range(0, max(h, w), RULER_EVERY):
            p = MARGIN + cell * scale
            if cell < w:
                FramePainter._text(image, str(cell), 2, p + 1)
            if cell < h:
                FramePainter._text(image, str(cell), p + 2, 1)
        if mark is not None:
            x, y = mark
            y0, x0 = MARGIN + y * scale - 2, MARGIN + x * scale - 2
            ring = np.array([255, 0, 255], dtype=np.uint8)
            for d in range(scale + 4):
                for yy, xx in ((y0, x0 + d), (y0 + scale + 3, x0 + d), (y0 + d, x0), (y0 + d, x0 + scale + 3)):
                    if 0 <= yy < image.shape[0] and 0 <= xx < image.shape[1]:
                        image[yy, xx] = ring
        return image

    @staticmethod
    def _text(image: np.ndarray, text: str, top: int, left: int) -> None:
        """Dark digits, each 3x5 font pixels drawn 2x2."""
        for i, ch in enumerate(text):
            bits = DIGITS[ch]
            for k, bit in enumerate(bits):
                if bit == "1":
                    y, x = top + (k // 3) * 2, left + i * 7 + (k % 3) * 2
                    image[y:y + 2, x:x + 2] = 40

    @staticmethod
    def encode(image: np.ndarray) -> bytes:
        """RGB uint8 array -> PNG (8-bit truecolour, no filtering)."""
        h, w, _ = image.shape
        raw = b"".join(b"\x00" + image[y].tobytes() for y in range(h))

        def chunk(kind: bytes, data: bytes) -> bytes:
            return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

        return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
