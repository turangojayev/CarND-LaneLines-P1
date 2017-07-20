import os
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
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
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


def process_and_save_video(input, output):
    clip1 = VideoFileClip(input)
    white_clip = \
        clip1.fl_image(Pipeline(update_weight=0.8,
                                kernel_size=5,
                                low_threshold=50,
                                high_threshold=150,
                                rho=1,
                                theta=np.pi / 180,
                                min_num_of_crossing_sinusoids=20,
                                min_line_len=20,
                                max_line_gap=400,
                                thickness=20))
    white_clip.write_videofile(output, audio=False)


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


def _update_rule(old_mean, new_value, step, weight=1):
    # do not update for 0 slope, we don't want to consider those
    if np.isnan(new_value) or new_value == 0.:
        return old_mean
    else:
        return old_mean + weight * (new_value - old_mean) / step


class OnlineUpdatedSlopeAndThickness:
    def __init__(self, weight):
        self._weight = weight
        self._step = 0
        # self._right_m_n_1 = 0.6
        self._right_m_n_1 = 0
        self._right_thickness = 1
        # self._left_m_n_1 = -0.72
        self._left_m_n_1 = 0
        self._left_thickness = 1

    def __call__(self, left, right):
        self._step += 1
        self._right_m_n_1 = _update_rule(self._right_m_n_1, right, self._step, self._weight)
        self._left_m_n_1 = _update_rule(self._left_m_n_1, left, self._step, self._weight)
        return self._left_m_n_1, self._right_m_n_1


x1 = itemgetter(0)
y1 = itemgetter(1)
x2 = itemgetter(2)
y2 = itemgetter(3)
delta_y = lambda line: y2(line) - y1(line)
delta_x = lambda line: x2(line) - x1(line)


class Pipeline:
    def __init__(self,
                 update_weight=0.8,
                 kernel_size=5,
                 low_threshold=85,
                 high_threshold=170,
                 rho=1,
                 theta=np.pi / 180,
                 min_num_of_crossing_sinusoids=25,
                 min_line_len=20,
                 max_line_gap=400,
                 thickness=20):
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
        self._updater = OnlineUpdatedSlopeAndThickness(update_weight)

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

        vertices = adjust_vertices(image)[0]

        median_left = np.median(tangents[left_indices])
        median_right = np.median(tangents[right_indices])

        left_slope, right_slope = self._updater(median_left, median_right)

        self._draw_line(image,
                        left_slope,
                        '_left_right_line',
                        lines[left_indices, 0, 0],
                        lines[left_indices, 0, 1],
                        vertices[0][0],
                        vertices[1][0])

        self._draw_line(image,
                        right_slope,
                        '_right_right_line',
                        lines[right_indices, 0, 0],
                        lines[right_indices, 0, 1],
                        vertices[3][0],
                        vertices[2][0])

    def _draw_line(self, image, slope, which, xs, ys, x0, x2):
        if not np.isnan(slope):
            x1 = np.median(xs)
            y1 = np.median(ys)
            if not np.isnan(x1):
                (x_start, y_start), (x_end, y_end) = _extrapolate(x1, y1, slope, x0, x2)
                setattr(self, which, ((x_start, y_start), (x_end, y_end)))
            else:  # if nothing was found
                if hasattr(self, which):
                    (x_start, y_start), (x_end, y_end) = getattr(self, which)
                else:
                    x_start = None

            if x_start:
                cv2.line(image, (x_start, y_start), (x_end, y_end), color, 20)


def _extrapolate(x1, y1, slope, x0, x2):
    y0 = int(slope * (x0 - x1) + y1)
    y2 = int(slope * (x2 - x1) + y1)
    return (x0, y0), (x2, y2)


def videos():
    process_and_save_video("test_videos/solidWhiteRight.mp4", 'test_videos_output/solidWhiteRightOutput.mp4')
    process_and_save_video('test_videos/solidYellowLeft.mp4', 'test_videos_output/solidYellowLeft.mp4')
    process_and_save_video('test_videos/challenge.mp4', 'test_videos_output/challenge.mp4')


def for_images(directory):
    filenames = os.listdir(directory)
    paths = [os.path.join(directory, filename) for filename in filenames]
    images = list(map(mpimg.imread, paths))
    images = list(map(lambda image: convert_if_needed(image), images))

    # hls_ed = list(map(lambda image: mask_bright_areas(cv2.cvtColor(image, cv2.COLOR_RGB2HLS)), images))
    # # plot(hls_ed, 'hls')
    #
    # in_gray_scale = list(map(grayscale, hls_ed))
    # plot(in_gray_scale, 'grayscale')
    #
    # blurred = list(map(lambda image: gaussian_blur(image, kernel_size=5), in_gray_scale))
    # # plot(blurred, 'blurred')
    #
    # edges = list(map(lambda image: canny(convert_if_needed(image), low_threshold=20, high_threshold=150), blurred))
    # # plot(edges, 'edges found')
    #
    # masked_edges = list(map(lambda image: region_of_interest(image, adjust_vertices), edges))
    # # plot(masked_edges, 'edges masked')
    #
    # lines = \
    #     list(map(lambda image: hough_lines(image,
    #                                        rho=1,
    #                                        theta=np.pi / 180,
    #                                        threshold=10,
    #                                        min_line_len=3,
    #                                        max_line_gap=50), masked_edges))
    #
    # line_drawn_images = \
    #     list(map(lambda image: np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8), masked_edges))
    #
    # for line_img, lines_for_img in zip(line_drawn_images, lines):
    #     draw_lines(line_img, lines_for_img)
    #
    # plot(line_drawn_images, 'hough lines')
    #
    # weighted_lines = \
    #     list(
    #         map(
    #             lambda zipped: weighted_img(zipped[0], initial_img=convert_if_needed(zipped[1])),
    #             zip(line_drawn_images, images)))
    # plot(weighted_lines, 'final')


    for image in images:
        pipeline = Pipeline(1,
                            kernel_size=5,
                            low_threshold=85,
                            high_threshold=170,
                            rho=1,
                            theta=np.pi / 180,
                            min_num_of_crossing_sinusoids=30,
                            min_line_len=30,
                            max_line_gap=400,
                            thickness=20)
        drawn = pipeline(image)
        plt.imshow(drawn)
        plt.show()


if __name__ == '__main__':
    # directory = "test_images"
    # directory = "test/test_images"
    # for_images(directory)
    videos()
