import argparse
import cv2
import numpy as np
from collections import Counter


class ImagePainter:
    def __init__(self, mask='mask.png', color_mask='color_mask.png'):
        self.mask = cv2.imread(mask)
        self.color_mask = cv2.imread(color_mask)
        self.gray_background = cv2.cvtColor(self.mask, cv2.COLOR_BGR2GRAY)
        self.contours, _ = cv2.findContours(self.gray_background, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def process_contours(self):
        result_img = np.zeros_like(self.color_mask)
        contours_img = np.zeros_like(self.color_mask)
        cv2.drawContours(contours_img, self.contours, -1, (0, 0, 0), 1)

        for contour in self.contours:
            mask = np.zeros_like(self.gray_background)
            cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

            pixels = self.color_mask[mask == 255]

            if len(pixels) > 0:
                most_common_color = Counter(map(tuple, pixels)).most_common(1)[0][0]
                most_common_color = tuple(map(int, most_common_color))
            else:
                most_common_color = (0, 0, 0)

            cv2.drawContours(result_img, [contour], -1, most_common_color, -1)
        cv2.imwrite('final_result.png', result_img)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask')
    parser.add_argument('--color_mask')
    args = parser.parse_args()
    if args.mask is not None and args.color_mask is not None:
        args = parser.parse_args()
        image = ImagePainter(args.mask, args.color_mask)
    else:
        image = ImagePainter()
    image.process_contours()


