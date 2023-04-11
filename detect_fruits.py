import json
from pathlib import Path
from typing import Dict

import click
import cv2
import numpy as np
from tqdm import tqdm


def detect_fruits(img_path: str) -> Dict[str, int]:
    """Fruit detection function, to implement.

    Parameters
    ----------
    img_path : str
        Path to processed image.

    Returns
    -------
    Dict[str, int]
        Dictionary with quantity of each fruit.
    """
    img_to_scale = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img_to_scale, (0, 0), fx=0.5, fy=0.5)
    blur = cv2.GaussianBlur(img, (19, 19), 0)
    img_hsv = cv2.cvtColor(blur, cv2.COLOR_RGB2HSV)

    # banana
    mask_banana1 = cv2.inRange(img_hsv, np.array([71, 95, 91]), np.array([98, 255, 255]))
    mask_banana2 = cv2.inRange(img_hsv, np.array([71, 75, 162]), np.array([98, 255, 255]))
    mask_sum_banana = cv2.bitwise_or(mask_banana1, mask_banana2)

    kernel1_banana = np.ones((15, 15), np.uint8)
    kernel2_banana = np.ones((29, 29), np.uint8)
    opening_banana = cv2.morphologyEx(mask_sum_banana, cv2.MORPH_OPEN, kernel1_banana)
    closing_banana = cv2.morphologyEx(opening_banana, cv2.MORPH_CLOSE, kernel2_banana)

    mask_finally_banana = cv2.bitwise_and(img_hsv, img_hsv, mask=closing_banana)

    image_bgr_banana = cv2.cvtColor(mask_finally_banana, cv2.COLOR_HSV2BGR)
    image_gray_banana = cv2.cvtColor(image_bgr_banana, cv2.COLOR_BGR2GRAY)
    image_res_banana, image_thresh_banana = cv2.threshold(image_gray_banana, 0, 255, cv2.THRESH_BINARY_INV)
    image_thresh_banana = cv2.bitwise_not(image_thresh_banana)
    contours_banana, _hierarchy_banana = cv2.findContours(image_thresh_banana.copy(), cv2.RETR_EXTERNAL,
                                                          cv2.CHAIN_APPROX_SIMPLE)

    bananas = 0
    cv2.drawContours(mask_finally_banana, contours_banana, -1, (255, 255, 255), 3)

    for x in contours_banana:
        # print(cv2.contourArea(x))
        if cv2.contourArea(x) > 26000:
            bananas = bananas + 1


    # orange
    mask_orange1 = cv2.inRange(img_hsv, np.array([100, 73, 210]), np.array([105, 255, 255]))
    mask_orange2 = cv2.inRange(img_hsv, np.array([103, 101, 204]), np.array([110, 255, 255]))
    mask_sum_orange = cv2.bitwise_or(mask_orange1, mask_orange2)

    kernel1_orange = np.ones((9, 9), np.uint8)
    kernel2_orange = np.ones((31, 31), np.uint8)
    opening_orange = cv2.morphologyEx(mask_sum_orange, cv2.MORPH_OPEN, kernel1_orange)
    closing_orange = cv2.morphologyEx(opening_orange, cv2.MORPH_CLOSE, kernel2_orange)

    mask_finally_orange = cv2.bitwise_and(img_hsv, img_hsv, mask=closing_orange)

    image_bgr_orange = cv2.cvtColor(mask_finally_orange, cv2.COLOR_HSV2BGR)
    image_gray_orange = cv2.cvtColor(image_bgr_orange, cv2.COLOR_BGR2GRAY)
    image_res_orange, image_thresh_orange = cv2.threshold(image_gray_orange, 0, 255, cv2.THRESH_BINARY_INV)
    image_thresh_orange = cv2.bitwise_not(image_thresh_orange)
    contours_orange, _hierarchy_orange = cv2.findContours(image_thresh_orange.copy(), cv2.RETR_EXTERNAL,
                                                          cv2.CHAIN_APPROX_SIMPLE)

    oranges = 0
    cv2.drawContours(mask_finally_orange, contours_orange, -1, (255, 255, 255), 3)

    for x in contours_orange:
        # print(cv2.contourArea(x))
        if cv2.contourArea(x) > 17500:
            oranges = oranges + 1


    # Apple
    mask_apple1 = cv2.inRange(img_hsv, np.array([111, 64, 0]), np.array([180, 220, 189]))
    mask_apple2 = cv2.inRange(img_hsv, np.array([110, 50, 0]), np.array([180, 215, 255]))
    mask_sum_apple1 = cv2.bitwise_or(mask_apple1, mask_apple2)
    mask_apple3 = cv2.inRange(img_hsv, np.array([102, 168, 30]), np.array([130, 206, 194]))
    mask_sum_apple2 = cv2.bitwise_or(mask_sum_apple1, mask_apple3)
    mask_apple4 = cv2.inRange(img_hsv, np.array([102, 95, 69]), np.array([133, 206, 153]))
    mask_sum_apple3 = cv2.bitwise_or(mask_sum_apple2, mask_apple4)

    kernel1_apple = np.ones((7, 7), np.uint8)
    kernel2_apple = np.ones((33, 33), np.uint8)
    opening_apple = cv2.morphologyEx(mask_sum_apple3, cv2.MORPH_OPEN, kernel1_apple)
    closing_apple = cv2.morphologyEx(opening_apple, cv2.MORPH_CLOSE, kernel2_apple)

    mask_finally_apple = cv2.bitwise_and(img_hsv, img_hsv, mask=closing_apple)

    image_bgr_apple = cv2.cvtColor(mask_finally_apple, cv2.COLOR_HSV2BGR)
    image_gray_apple = cv2.cvtColor(image_bgr_apple, cv2.COLOR_BGR2GRAY)
    image_res_apple, image_thresh_apple = cv2.threshold(image_gray_apple, 0, 255, cv2.THRESH_BINARY_INV)
    image_thresh_apple = cv2.bitwise_not(image_thresh_apple)
    contours_apple, _hierarchy_apple = cv2.findContours(image_thresh_apple.copy(), cv2.RETR_EXTERNAL,
                                                        cv2.CHAIN_APPROX_SIMPLE)

    apples = 0
    cv2.drawContours(mask_finally_apple, contours_apple, -1, (255, 255, 255), 3)

    for x in contours_apple:
        # print(cv2.contourArea(x))
        if cv2.contourArea(x) > 20000:
            apples = apples + 1


    
    apple = apples
    banana = bananas
    orange = oranges

    return {'apple': apple, 'banana': banana, 'orange': orange}


@click.command()
@click.option('-p', '--data_path', help='Path to data directory', type=click.Path(exists=True, file_okay=False,
                                                                                  path_type=Path), required=True)
@click.option('-o', '--output_file_path', help='Path to output file', type=click.Path(dir_okay=False, path_type=Path),
              required=True)
def main(data_path: Path, output_file_path: Path):
    img_list = data_path.glob('*.jpg')

    results = {}

    for img_path in tqdm(sorted(img_list)):
        fruits = detect_fruits(str(img_path))
        results[img_path.name] = fruits

    with open(output_file_path, 'w') as ofp:
        json.dump(results, ofp)


if __name__ == '__main__':
    main()
