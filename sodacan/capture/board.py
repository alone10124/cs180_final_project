"""The ChArUco board: construction from configs/board.yaml, printable image, board <-> world.

World = diag(1, -1, -1) (board - c), with c the can centre on the board: world x follows
the board's columns, world z points up out of the board, and the can axis is world z.
"""
import cv2
import numpy as np

from sodacan.config import load_yaml

FLIP = np.diag([1.0, -1.0, -1.0])


def load_board(cfg="board"):
    cfg = load_yaml(cfg) if isinstance(cfg, str) else cfg
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, cfg["dictionary"]))
    board = cv2.aruco.CharucoBoard(tuple(cfg["squares"]), float(cfg["square_length"]), float(cfg["marker_length"]), dictionary)
    return board, cfg


def board_size_m(cfg):
    return cfg["squares"][0] * cfg["square_length"], cfg["squares"][1] * cfg["square_length"]


def board_image(board, cfg, ppm, can_radius=None):
    """Greyscale uint8 image of exactly the board area at ppm pixels per metre.

    With can_radius, the can's footprint is outlined so it can be placed by eye."""
    w, h = board_size_m(cfg)
    img = board.generateImage((round(w * ppm), round(h * ppm)), marginSize=0, borderBits=1)
    if can_radius:
        c = (round(cfg["can_centre"][0] * ppm), round(cfg["can_centre"][1] * ppm))
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.circle(img, c, round(can_radius * ppm), (0, 0, 255), max(2, round(ppm * 0.0008)), cv2.LINE_AA)
        cv2.drawMarker(img, c, (0, 0, 255), cv2.MARKER_CROSS, round(ppm * 0.01), max(1, round(ppm * 0.0005)))
    return img


def board_to_world(cfg):
    """4x4 transform taking board coordinates to world (can) coordinates."""
    c = np.array([cfg["can_centre"][0], cfg["can_centre"][1], 0.0])
    T = np.eye(4)
    T[:3, :3] = FLIP
    T[:3, 3] = -FLIP @ c
    return T


def render_texture(cfg, ppm=2000):
    """Board albedo texture for the simulator: (3, H, W) float, plus ppm and centre."""
    import torch

    board, _ = load_board(cfg)
    img = board_image(board, cfg, ppm).astype(np.float32) / 255.0
    albedo = 0.04 + 0.76 * img
    tex = torch.from_numpy(albedo)[None].expand(3, -1, -1).contiguous()
    return {"texture": tex, "ppm": ppm, "centre": tuple(cfg["can_centre"])}
