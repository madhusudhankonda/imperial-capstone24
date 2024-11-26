# -*- coding: utf-8 -*-
import os
import sys

sys.path.append('src/model')
sys.path.append('src/data')

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm import tqdm

from satellite_image import SatelliteImage
from segmentation import SegmentationModel

mean = np.load('model/mean.npy')
model = SegmentationModel('model/model', mean)


def overlay_mask(image, mask, alpha=0.5, rgb=[255, 0, 0]):
    overlay = image.copy()
    overlay[mask] = np.array(rgb, dtype=np.uint8)
    output = image.copy()
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)
    return output


def plot_segmentation(image_path, score, figsize=(4, 4)):
    image = np.array(Image.open(image_path))
    building_score = score[1]

    building_mask_pred = (np.argmax(score, axis=0) == 1)
    building_overlay_pred = overlay_mask(image, building_mask_pred)

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(3 * figsize[0], figsize[1]))

    ax0.imshow(image)
    ax0.set_title(f'Input ({image_path})')

    ax1.imshow(building_score, vmin=0.0, vmax=1.0)
    ax1.set_title('Prediction')

    ax2.imshow(building_overlay_pred)
    ax2.set_title('Prediction on top of input image')

    plt.show()


def damage_estimate(x1: float, y1: float, x2: float, y2: float,
                    date1: str = '2023-03-15', date2: str = '2024-02-08') -> None:
    """Estimate the damage in the given bounding box between two `date1` and `date2`

    The predicted building mask will be saved in the `output` directory. Filename format is
    `output/{TILE_X}_{TILE_Y}_{DATE}.png`. The damage estimates will be saved in
    `output/damage.csv`, with cols. `tile_x, tile_y, lon., lat., damage`.

    :param x1: Left coord.
    :param y1: Top coord.
    :param x2: Right coord.
    :param y2: Bottom coord.
    :param date1: Default = "2023-03-15"
    :param date2: Default = "2024-02-08"
    """
    # Get top left and bottom right tile coords.
    bbox = [x1, y1, x2, y2]
    x1, y1 = SatelliteImage.lon_lat_to_tile_xy(bbox[0], bbox[1], zoom=18)
    x2, y2 = SatelliteImage.lon_lat_to_tile_xy(bbox[2], bbox[3], zoom=18)

    # Download all satellite images in the bounding box, in the given years
    imgs = []
    for x in tqdm(range(x1, x2 + 1), desc='Downloading satellite images'):
        for y in range(y1, y2 + 1):
            image = SatelliteImage([x, y], zoom=18, is_tile=True)
            for date in [date1, date2]:
                image.get_satellite_images(date)
            imgs.append(image)

    # Inference
    # TODO: should be batched
    os.makedirs('output', exist_ok=True)
    if not os.path.exists('output/damage.csv'):
        with open('output/damage.csv', 'w') as f:
            f.write('x,y,lon,lat,date,damage\n')
    for img in tqdm(imgs, desc='Inference'):
        building_area = []
        for date in [date1, date2]:
            score = model.apply_segmentation(np.array(Image.open(img.images[date])))
            # plot_segmentation(img.images[date], score)
            building_mask_pred = (np.argmax(score, axis=0) == 1)
            building_area.append(np.sum(building_mask_pred))
            Image \
                .fromarray(building_mask_pred) \
                .save(f'output/{img.bbox[0]}_{img.bbox[1]}_{date}.png')
        damage = (building_area[1] - building_area[0]) / 65536  # Raw image size is 256x256
        lon, lat = img.tile_xy_to_lon_lat(img.bbox[0], img.bbox[1], zoom=18)
        with open('output/damage.csv', 'a') as f:
            f.write(f'{img.bbox[0]},{img.bbox[1]},{lon},{lat},{damage}\n')
    return


if __name__ == '__main__':
    # damage_estimate(36.149, 36.213, 36.162, 36.202)
    pass