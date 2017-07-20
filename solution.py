import os
from collections import deque
from functools import partial
from operator import itemgetter

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import VideoFileClip

color = [255, 0, 0]


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, adjust_vertices):
    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, adjust_vertices(img), ignore_mask_color)
    return cv2.bitwise_and(img, mask)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    return cv2.HoughLinesP(img,
                           rho,
                           theta,
                           threshold,
                           np.array([]),
                           minLineLength=min_line_len,
                           maxLineGap=max_line_gap)


def weighted_img(img, initial_img, alpha=0.8, beta=1., lambdda=0.):
    return cv2.addWeighted(initial_img, alpha, img, beta, lambdda)


def plot(images, title, columns=2, cmap=None):
    rows = (len(images) + 1) // columns
    subplot = partial(plt.subplot, rows, columns)
    plt.figure(figsize=(10, 20))
    plt.suptitle(title)

    for i, image in enumerate(images, 1):
        subplot(i)
        plt.imshow(image, cmap='gray' if len(image.shape) == 2 else cmap)

    plt.show()


def convert_if_needed(image):
    if image.dtype == np.float32:
        image = np.uint8(image * 255)
    return image


def mask_bright_areas(image):
    # mask for yellow lines
    mask = cv2.inRange(image, np.array([10, 0, 100]), np.array([40, 255, 255]))
    # combine with mask for white lines
    mask = cv2.bitwise_or(mask, cv2.inRange(image, np.array([0, 200, 0]), np.array([255, 255, 255])))
    return cv2.bitwise_and(image, image, mask=mask)


def adjust_vertices(img):
    xscale = 0.001041667
    yscale = 0.001851852
    return \
        np.array([
            [(150 * xscale * img.shape[1], 540 * yscale * img.shape[0]),
             (460 * xscale * img.shape[1], 320 * yscale * img.shape[0]),
             (510 * xscale * img.shape[1], 320 * yscale * img.shape[0]),
             (900 * xscale * img.shape[1], 540 * yscale * img.shape[0])]],
            dtype=np.int32)


class MeanWithQueue:
    def __init__(self):
        self._left = deque(maxlen=10)
        self._right = deque(maxlen=10)

    def __call__(self, left, right):
        if not np.isnan(left):
            self._left.append(left)
        if not np.isnan(right):
            self._right.append(right)
        return np.mean(self._left), np.mean(self._right)


x1 = itemgetter(0)
y1 = itemgetter(1)
x2 = itemgetter(2)
y2 = itemgetter(3)
delta_y = lambda line: y2(line) - y1(line)
delta_x = lambda line: x2(line) - x1(line)


class Pipeline:
    def __init__(self,
                 kernel_size=5,
                 low_threshold=85,
                 high_threshold=170,
                 rho=1,
                 theta=np.pi / 180,
                 min_num_of_crossing_sinusoids=25,
                 min_line_len=20,
                 max_line_gap=400,
                 thickness=20,
                 slope_updater=None,
                 left_point=None,
                 right_point=None):
        """
        :update_weight: weight for the new slope in the mean slope update rule over frames
        :kernel_size: kernel size for Gaussian blur
        :low_threshold: low intensity threshold for Canny edge detection
        :high_threshold: high intensity threshold for Canny edge detection
        :rho: distance resolution of the accumulator in pixels
        :theta: angle resolution of the accumulator in radians
        :min_num_of_crossing_sinusoids: min number of sine curves required to cross at a point in Hough space
        :min_line_len: min length required for the line to be considered
        :max_line_gap: max allowed gap between parts of line for Hough transformation
        :thickness: integer indicating thickness of the line to be drawn
        :slope_updater: callable that takes two arguments and returns a tuple
        :left_point: callable that takes two arguments and returns a tuple
        :right_point: callable that takes two arguments and returns a tuple
        """
        self._kernel_size = kernel_size
        self._low_threshold = low_threshold
        self._high_threshold = high_threshold
        self._rho = rho
        self._theta = theta
        self._min_num_of_crossing_lines = min_num_of_crossing_sinusoids
        self._min_line_len = min_line_len
        self._max_line_gap = max_line_gap
        self._thickness = thickness
        self._slope_updater = slope_updater if slope_updater is not None else MeanWithQueue()
        self._left_point = left_point if left_point is not None else MeanWithQueue()
        self._right_point = right_point if right_point is not None else MeanWithQueue()

    def __call__(self, image):
        img = mask_bright_areas(cv2.cvtColor(image, cv2.COLOR_RGB2HLS))
        gray_image = grayscale(img)
        blurred = gaussian_blur(gray_image, kernel_size=self._kernel_size)
        edges_detected = canny(blurred, low_threshold=self._low_threshold, high_threshold=self._high_threshold)
        masked_edges = region_of_interest(edges_detected, adjust_vertices)
        lines = \
            hough_lines(masked_edges,
                        self._rho,
                        self._theta,
                        self._min_num_of_crossing_lines,
                        self._min_line_len,
                        self._max_line_gap)

        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        self._draw_lines(line_img, lines)
        return weighted_img(line_img, initial_img=image)

    def _draw_lines(self, image, lines):
        vertices = adjust_vertices(image)[0]

        # tangent is actually infinity in case of vertical lines, but we assign zero to filter it out
        # also don't consider horizontal lines
        tangents = \
            np.array(
                [float(delta_y(line)) / delta_x(line)
                 if delta_x(line) != 0 and delta_y(line) != 0 else 0
                 for line in lines[:, 0]])

        # since origin is at top left, signs are reversed
        left_indices = np.where(tangents < 0)[0]
        right_indices = np.where(tangents > 0)[0]

        median_left = np.median(tangents[left_indices])
        median_right = np.median(tangents[right_indices])

        left_slope, right_slope = self._slope_updater(median_left, median_right)

        self._draw_line(image,
                        left_slope,
                        '_left_right_line',
                        lines[left_indices, 0, 0],
                        lines[left_indices, 0, 1],
                        vertices[0][0],
                        vertices[1][0],
                        self._left_point)

        self._draw_line(image,
                        right_slope,
                        '_right_right_line',
                        lines[right_indices, 0, 0],
                        lines[right_indices, 0, 1],
                        vertices[3][0],
                        vertices[2][0],
                        self._right_point)

    def _draw_line(self, image, slope, which, xs, ys, x0, x2, updater):
        if not np.isnan(slope):
            x1 = np.median(xs)
            y1 = np.median(ys)
            x1, y1 = updater(x1, y1)

            if not np.isnan(x1):
                (x_start, y_start), (x_end, y_end) = _extrapolate(x1, y1, slope, x0, x2)
                setattr(self, which, ((x_start, y_start), (x_end, y_end)))
            else:  # if nothing was found
                if hasattr(self, which):
                    (x_start, y_start), (x_end, y_end) = getattr(self, which)
                else:
                    x_start = None

            if x_start:
                cv2.line(image, (x_start, y_start), (x_end, y_end), color, self._thickness)

    def slope(self, lines):
        return super().slope(lines)


def _extrapolate(x1, y1, slope, x0, x2):
    y0 = int(slope * (x0 - x1) + y1)
    y2 = int(slope * (x2 - x1) + y1)
    return (x0, y0), (x2, y2)


def videos(directory):
    for filename in os.listdir(directory):
        process_and_save_video(os.path.join(directory, filename),
                               os.path.join(directory + "_output", filename))


def process_and_save_video(input, output):
    clip1 = VideoFileClip(input)
    pipeline = \
        Pipeline(kernel_size=9,
                 low_threshold=85,
                 high_threshold=170,
                 rho=1,
                 theta=np.pi / 180,
                 min_num_of_crossing_sinusoids=25,
                 min_line_len=30,
                 max_line_gap=200,
                 thickness=10)
    white_clip = clip1.fl_image(pipeline)
    white_clip.write_videofile(output, audio=False)


def for_images(directory):
    filenames = os.listdir(directory)
    paths = [os.path.join(directory, filename) for filename in filenames]
    images = list(map(mpimg.imread, paths))
    images = list(map(lambda image: convert_if_needed(image), images))

    for image, filename in zip(images, filenames):
        pipeline = Pipeline(kernel_size=9,
                            low_threshold=85,
                            high_threshold=170,
                            rho=1,
                            theta=np.pi / 180,
                            min_num_of_crossing_sinusoids=25,
                            min_line_len=30,
                            max_line_gap=100,
                            thickness=10)
        drawn = pipeline(image)
        plt.imshow(drawn)
        plt.savefig(os.path.join(directory + '_output', filename))
        plt.xticks([])
        plt.yticks([])


if __name__ == '__main__':
    for_images("test_images")
    videos("test_videos")
