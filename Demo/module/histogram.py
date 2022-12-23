import numpy as np
import pandas as pd
from PIL import Image, ImageTk, ImageFile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def histogram(input, output):
    img = Image.open(input)

    imgray = img.convert(mode='L')

    img_array = np.asarray(imgray)

    histogram_array = np.bincount(img_array.flatten(), minlength=256)

    num_pixels = np.sum(histogram_array)
    histogram_array = histogram_array/num_pixels

    chistogram_array = np.cumsum(histogram_array)

    transform_map = np.floor(255 * chistogram_array).astype(np.uint8)

    img_list = list(img_array.flatten())

    eq_img_list = [transform_map[p] for p in img_list]
    eq_img_array = np.reshape(np.asarray(eq_img_list), img_array.shape)

    eq_img = Image.fromarray(eq_img_array, mode='L')
    eq_img.save(output)

    print('Histogram Equalization => complete')
