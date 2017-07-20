from functools import partial

from moviepy.editor import VideoFileClip
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import os

from buildup import gaussian_blur, canny, adjust_vertices, region_of_interest, hough_lines, weighted_img


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def plot(images, columns=2, cmap=None):
    rows = (len(images) + 1) // columns
    subplot = partial(plt.subplot, rows, columns)
    plt.figure(figsize=(10, 20))
    for i, image in enumerate(images, 1):
        subplot(i)
        plt.imshow(image, cmap='gray' if len(image.shape) == 2 else cmap)
        plt.xticks([])
        plt.yticks([])
    plt.show()


def convert_if_needed(image):
    dtype = image.dtype
    if dtype == np.float32:
        image = np.uint8(image * 255)
    return image


def hls(xyz_image):
    # lower = np.array([10, 100, 100])
    # upper = np.array([40, 255, 255])
    # mask = cv2.inRange(xyz_image, lower, upper)
    #
    # lower = np.array([40, 165, 0])
    # upper = np.array([255, 255, 255])


    lower = np.array([10, 0, 100])
    upper = np.array([40, 255, 255])
    mask = cv2.inRange(xyz_image, lower, upper)

    lower = np.array([0, 200, 0])
    upper = np.array([255, 255, 255])

    mask = cv2.bitwise_or(mask, cv2.inRange(xyz_image, lower, upper))

    return cv2.bitwise_and(xyz_image, xyz_image, mask=mask)


def plot_histogram(images, cmap=None, interval=[0, 256]):
    rows = len(images)
    subplot = partial(plt.subplot, rows, 2)
    plt.figure(figsize=(10, 20))
    for image, i in zip(images, range(1, 2 * rows, 2)):
        image = convert_if_needed(image)
        subplot(i)
        plt.imshow(image, cmap='gray' if len(image.shape) == 2 else cmap)
        subplot(i + 1)
        plt.hist(x=image.flatten(), bins=256, range=interval)
        # plt.xticks([])
        # plt.yticks([])
    plt.show()


def plot_histogram_on_line(images, cmap=None, title=None, line_loc_as_float=0.8, directory=None):
    rows = len(images)
    if len(images[0].shape) == 2:
        columns = len(images[0].shape)
    else:
        columns = len(images[0].shape) + 1

    subplot = partial(plt.subplot, rows, columns)
    plt.figure(figsize=(10, 10))

    if title is not None:
        plt.suptitle(title)

    for image, i in zip(images, range(1, columns * rows, columns)):
        image = convert_if_needed(image)
        subplot(i)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(image, cmap='gray' if len(image.shape) == 2 else cmap)
        line_number = int(line_loc_as_float * image.shape[0])
        line = image[line_number, :] if columns == 2 else image[line_number, :, :]

        def plot_subplot(idx):
            subplot(idx)
            if columns == 2:
                plt.plot(range(line.shape[0]), line)
            else:
                plt.plot(range(line.shape[0]), line[:, idx - i - 1])

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

    # lower = np.array([200, 150, 150])
    lower = np.array([200, 200, 200])
    upper = np.array([255, 255, 255])
    mask = cv2.bitwise_or(mask, cv2.inRange(image, lower, upper))

    return cv2.bitwise_and(image, image, mask=mask)


if __name__ == '__main__':
    # directory = "test_images"
    directory = "challenge_images"
    filenames = os.listdir(directory)
    paths = [os.path.join(directory, filename) for filename in filenames]
    images = list(map(mpimg.imread, paths))
    images = list(map(lambda image: convert_if_needed(image), images))
    plot_histogram_on_line(images, title="images-histogram")

    selected = list(map(lambda image: select_from_rgb(image), images))
    plot(selected)

    hls = list(map(lambda image: hls(cv2.cvtColor(image, cv2.COLOR_RGB2HLS)), images))
    plot_histogram_on_line(hls)

    in_gray_scale = list(map(grayscale, hls))
    plot_histogram_on_line(in_gray_scale)

    # plot_histogram_on_line(in_gray_scale)
    # plot_histogram(in_gray_scale)


    # in_gray_scale = list(map(lambda image: hls(cv2.cvtColor(image, cv2.COLOR_RGB2HLS)), selected))
    # plot(in_gray_scale)
