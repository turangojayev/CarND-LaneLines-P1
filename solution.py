import os
from collections import deque
from functools import partial
from operator import itemgetter

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from moviepy.editor import VideoFileClip


def plot(images, columns=3, cmap=None, title=None, directory=None, ):
    rows = len(images) / columns
    subplot = partial(plt.subplot, rows, columns)
    plt.figure(figsize=(15, 10))
    for i, image in enumerate(images, 1):
        subplot(i)
        plt.imshow(image, cmap='gray' if len(image.shape) == 2 else cmap)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    if title is not None:
        if directory:
            title = os.path.join(directory, title)
        plt.savefig(title)
    plt.show()


colors = ['Red', 'Green', 'Blue']


def plot_histogram_for_line(images,
                            cmap=None,
                            title=None,
                            line_loc_as_float=0.8,
                            directory=None,
                            colors=colors):
    rows = len(images)
    if len(images[0].shape) == 2:
        columns = len(images[0].shape)
    else:
        columns = len(images[0].shape) + 1

    subplot = partial(plt.subplot, rows, columns)
    plt.figure(figsize=(20, 10))

    for image, i in zip(images, range(1, columns * rows, columns)):
        image = convert_if_needed(image)
        subplot(i)
        plt.xticks([])
        plt.yticks([])
        line_number = int(line_loc_as_float * image.shape[0])
        plt.axhline(line_number, 0, color='red')
        plt.imshow(image, cmap='gray' if len(image.shape) == 2 else cmap)
        line = image[line_number, :] if columns == 2 else image[line_number, :, :]

        def plot_subplot(idx):
            subplot(idx)
            plt.xticks([])
            if columns == 2:
                plt.plot(range(line.shape[0]), line)
            else:
                plt.plot(range(line.shape[0]), line[:, idx - i - 1])
                plt.title(colors[idx - i - 1])

        for channel in range(columns - 1):
            plot_subplot(i + 1 + channel)

    if title is not None:
        if directory:
            title = os.path.join(directory, title)
        plt.savefig(title)
    plt.show()


def select_from_rgb(image):
    lower = np.array([160, 160, 10])
    upper = np.array([255, 255, 255])
    mask = cv2.inRange(image, lower, upper)

    lower = np.array([200, 200, 200])
    upper = np.array([255, 255, 255])
    mask = cv2.bitwise_or(mask, cv2.inRange(image, lower, upper))

    return cv2.bitwise_and(image, image, mask=mask)


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


def convert_if_needed(image):
    """
    Convenience function for testing on png files (read as float matrices)
    :param image:
    :return: image converted to uint8
    """
    if image.dtype == np.float32:
        image = np.uint8(image * 255)
    return image


def mask_dark_areas(image):
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
        """
        Keep record of last 10 element couples
        """
        self._left = deque(maxlen=10)
        self._right = deque(maxlen=10)

    def __call__(self, left, right):
        if not np.isnan(left):
            self._left.append(left)
        if not np.isnan(right):
            self._right.append(right)
        return np.mean(self._left), np.mean(self._right)


# making it convenient to calculate delta_x and delta_y's for lines
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
                 color=None,
                 thickness=20,
                 slope_updater=None,
                 left_point=None,
                 right_point=None):
        """
        Basic class that implements the pipeline of detecting the lane lines.
        :param kernel_size: kernel size for Gaussian blur
        :param low_threshold: low intensity threshold for Canny edge detection
        :param high_threshold: high intensity threshold for Canny edge detection
        :param rho: distance resolution of the accumulator in pixels
        :param theta: angle resolution of the accumulator in radians
        :param min_num_of_crossing_sinusoids: min number of sine curves required to cross at a point in Hough space
        :param min_line_len: min length required for the line to be considered
        :param max_line_gap: max allowed gap between parts of line for Hough transformation
        :param color: color of the lines to be drawn
        :param thickness: integer indicating thickness of the line to be drawn
        :param slope_updater: callable that takes two arguments and returns a tuple
        :param left_point: callable that takes two arguments and returns a tuple
        :param right_point: callable that takes two arguments and returns a tuple
        """

        self._kernel_size = kernel_size
        self._low_threshold = low_threshold
        self._high_threshold = high_threshold
        self._rho = rho
        self._theta = theta
        self._min_num_of_crossing_lines = min_num_of_crossing_sinusoids
        self._min_line_len = min_line_len
        self._max_line_gap = max_line_gap
        # just for not putting a mutable object as a keyword parameter default value
        if color is None:
            color = [255, 0, 0]
        self._color = color
        self._thickness = thickness
        self._slope_updater = slope_updater if slope_updater is not None else MeanWithQueue()
        self._left_point = left_point if left_point is not None else MeanWithQueue()
        self._right_point = right_point if right_point is not None else MeanWithQueue()

    def __call__(self, image):
        """
        Once a pipeline object created, it can be called with an image as an argument and the following will be performed
        on it:
        convert from RGB -> HLS,
        mask yellow and white lines,
        convert from HLS -> Grayscale
        apply Gaussian blur
        apply Canny edge detection
        mask a region of interest
        apply Hough transformation and find lines
        draw lines
        :param image:
        :return: lane lines plotted on image
        """
        img = mask_dark_areas(cv2.cvtColor(image, cv2.COLOR_RGB2HLS))
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

        # more stable than mean against outliers
        median_left = np.median(tangents[left_indices])
        median_right = np.median(tangents[right_indices])

        # instead of current median slope, take mean over last k frames, smoothes the changes
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
        """
        Extrapolate a line to the end points of region and draw
        :param image: image, on which the line will be drawn
        :param slope: slope of the line to be drawn
        :param which: indicates if the drawing is for right or for left line
        :param xs: x coordinates of lower points
        :param ys: y coordinates of lower points
        :param x0: x coordinate of low end point
        :param x2: x coordinate of high end point
        :param updater: updater object for pivot point
        """
        if not np.isnan(slope):
            x1 = np.median(xs)
            y1 = np.median(ys)
            x1, y1 = updater(x1, y1)

            if not np.isnan(x1):
                (x_start, y_start), (x_end, y_end) = _extrapolate(x1, y1, slope, x0, x2)
                setattr(self, which, ((x_start, y_start), (x_end, y_end)))
            else:
                if hasattr(self, which):
                    (x_start, y_start), (x_end, y_end) = getattr(self, which)
                else:
                    x_start = None

            if x_start:
                cv2.line(image, (x_start, y_start), (x_end, y_end), self._color, self._thickness)


def _extrapolate(x1, y1, slope, x0, x2):
    """
    Extrapolate from pivot point to the ends
    :param x1:
    :param y1:
    :param slope:
    :param x0:
    :param x2:
    :return: tuple containing low and high end points
    """
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
        plt.xticks([])
        plt.yticks([])
        plt.savefig(os.path.join(directory + '_output', filename))


if __name__ == '__main__':
    writeup_directory = 'writeup_images'
    test_directory = "test_images"
    challenge_directory = "challenge_images"

    paths = [os.path.join(test_directory, filename) for filename in os.listdir(test_directory)]
    images = list(map(mpimg.imread, paths))
    images = list(map(lambda image: convert_if_needed(image), images))
    plot(images, title="plot1", directory=writeup_directory)
    plot_histogram_for_line(images, title='channels_over_line1', directory=writeup_directory)

    selected = list(map(lambda image: select_from_rgb(image), images))
    plot(selected, title="plot2", directory=writeup_directory)

    paths = [os.path.join(challenge_directory, filename) for filename in os.listdir(challenge_directory)]
    challenge_images = list(map(mpimg.imread, paths))
    challenge_images = list(map(lambda image: convert_if_needed(image), challenge_images))
    plot(challenge_images, title="plot3", directory=writeup_directory)
    plot_histogram_for_line(challenge_images, title="channels_over_line2", directory=writeup_directory)
    selected = list(map(lambda image: select_from_rgb(image), challenge_images))
    plot(selected, title="plot4", directory=writeup_directory)
    plot_histogram_for_line(selected, title='channels_over_line3', directory=writeup_directory)

    hlsed = list(map(lambda image: cv2.cvtColor(image, cv2.COLOR_RGB2HLS), challenge_images))
    plot(hlsed, title="plot5", directory=writeup_directory)
    plot_histogram_for_line(hlsed, colors=['Hue', 'Lightness', 'Saturation'],
                            title="channels_over_line4", directory=writeup_directory)

    lines_visible = list(map(lambda image: mask_dark_areas(image), hlsed))
    plot(lines_visible, title="plot6", directory=writeup_directory)
    plot_histogram_for_line(lines_visible, colors=['Hue', 'Lightness', 'Saturation'], title="channels_over_line5",
                            directory=writeup_directory)

    gray_images = list(map(grayscale, lines_visible))
    plot(gray_images, title="plot7", directory=writeup_directory)

    easy_images_in_grayscale = \
        list(map(lambda image: grayscale(mask_dark_areas(cv2.cvtColor(image, cv2.COLOR_RGB2HLS))), images))
    plot(easy_images_in_grayscale, title="plot8", directory=writeup_directory)

    easy_images_in_grayscale.extend(gray_images)
    print(len(easy_images_in_grayscale))

    blurred = list(map(lambda image: gaussian_blur(image, kernel_size=9), easy_images_in_grayscale))
    plot(blurred, title="plot9", directory=writeup_directory)

    edges_detected = list(map(lambda image: canny(image, 85, 170), blurred))
    plot(edges_detected, title="plot10", directory=writeup_directory)

    masked_edges = list(map(lambda image: region_of_interest(image, adjust_vertices), edges_detected))
    plot(masked_edges, title="plot11", directory=writeup_directory)

    lines = list(map(lambda image:
                     hough_lines(image,
                                 rho=1,
                                 theta=np.pi / 180,
                                 threshold=25,
                                 min_line_len=30,
                                 max_line_gap=100), masked_edges))


    def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(img, (x1, y1), (x2, y2), color, thickness)


    line_images = \
        list(map(lambda image: np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8), masked_edges))
    for tupl in zip(line_images, lines):
        draw_lines(tupl[0], tupl[1])

    plot(line_images, title="plot12", directory=writeup_directory)

    images.extend(challenge_images)
    results = []
    for image in images:
        pipeline = Pipeline(kernel_size=9,
                            low_threshold=85,
                            high_threshold=170,
                            rho=1,
                            theta=np.pi / 180,
                            min_num_of_crossing_sinusoids=25,
                            min_line_len=30,
                            max_line_gap=100,
                            thickness=10)
        results.append(pipeline(image))
    plot(results, title="plot13", directory=writeup_directory)


    for_images("test_images")
    videos("test_videos")