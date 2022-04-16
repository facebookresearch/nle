import cv2
import os
import numpy as np

from nle.visualization.glyph2tile import glyph2tile

tileset_path = 'nle/visualization/3.6.1tiles32.png'
tileset = cv2.imread(tileset_path)[..., ::-1]

tile_size = 32
h = tileset.shape[0] // tile_size
w = tileset.shape[1] // tile_size
tiles = []
for y in range(h):
    y *= tile_size
    for x in range(w):
        x *= tile_size
        tiles.append(tileset[y:y + tile_size, x:x + tile_size])


tileset = np.array(tiles)
glyph2tile = np.array(glyph2tile)


def draw_grid(imgs, ncol):
    grid = imgs.reshape((-1, ncol, *imgs[0].shape))
    rows = []
    for row in grid:
        rows.append(np.concatenate(row, axis=1))

    return np.concatenate(rows, axis=0)


def draw_frame(img, color=(90, 90, 90), thickness=3):
    return cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), color, thickness)


def draw_glyph(glyphs):
    tiles_idx = glyph2tile[glyphs]
    tiles = tileset[tiles_idx.reshape(-1)]
    scene_vis = draw_grid(tiles, glyphs.shape[1])
    frame = draw_frame(scene_vis)
    frame =  cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    cv2.imshow("frame", frame)
    cv2.waitKey(1)