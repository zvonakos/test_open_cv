import cv2
import numpy as np
from collections import Counter


class ImagePainter:
    def __init__(self, polygons_img='mask.png', colored_img='color_mask.png'):
        self.polygons_img = cv2.imread(polygons_img)
        self.colored_img = cv2.imread(colored_img)
        self.gray_polygons = cv2.cvtColor(self.polygons_img, cv2.COLOR_BGR2GRAY)
        self.contours, _ = cv2.findContours(self.gray_polygons, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    def process_data(self):
        result_img = np.zeros_like(self.colored_img)
        contours_img = np.zeros_like(self.colored_img)
        cv2.drawContours(contours_img, self.contours, -1, (0, 0, 0), 1)

        for contour in self.contours:
            mask = np.zeros_like(self.gray_polygons)
            cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

            pixels = self.colored_img[mask == 255]

            if len(pixels) > 0:
                most_common_color = Counter(map(tuple, pixels)).most_common(1)[0][0]
                most_common_color = tuple(map(int, most_common_color))
            else:
                most_common_color = (0, 0, 0)

            cv2.drawContours(result_img, [contour], -1, most_common_color, -1)
        cv2.imwrite('final_result.png', result_img)


if __name__ == '__main__':
    image = ImagePainter()
    image.process_data()
