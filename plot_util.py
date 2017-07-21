import os
from functools import partial

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from solution import convert_if_needed


def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)


def plot(images, columns=2, cmap=None):
    rows = len(images) / columns
    subplot = partial(plt.subplot, rows, columns)
    plt.figure(figsize=(10, 20))
    for i, image in enumerate(images, 1):
        subplot(i)
        plt.imshow(image, cmap='gray' if len(image.shape) == 2 else cmap)
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
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
    plt.figure(figsize=(10, 10))

    if title is not None:
        plt.suptitle(title)

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


if __name__ == '__main__':
    directory = "test_images"
    # directory = "challenge_images"
    filenames = os.listdir(directory)
    paths = [os.path.join(directory, filename) for filename in filenames]
    images = list(map(mpimg.imread, paths))
    images = list(map(lambda image: convert_if_needed(image), images))
    plot_histogram_for_line(images, title="images-histogram")

    selected = list(map(lambda image: select_from_rgb(image), images))
