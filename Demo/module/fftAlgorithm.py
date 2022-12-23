
import numpy as np
from skimage.io import imread, imshow
from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
from skimage.exposure import equalize_hist
import numpy as np
from PIL import Image, ImageTk, ImageFile
import scipy.fftpack as fp

def fftenhencement(img_filename, complete_image):
    im2freq = lambda data: fp.rfft(fp.rfft(data, axis=0),
                               axis=1)
    freq2im = lambda f: fp.irfft(fp.irfft(f, axis=1),
                                axis=0)

    ## Read in data file and transform
    data = np.array(Image.open(img_filename))
    print(data)

    freq = im2freq(data)
    back = freq2im(freq)
    # Make sure the forward and backward transforms work!
    assert(np.allclose(data, back))

    ## Helper functions to rescale a frequency-image to [0, 255] and save
    remmax = lambda x: x/x.max()
    remmin = lambda x: x - np.amin(x, axis=(0,1), keepdims=True)
    touint8 = lambda x: (remmax(remmin(x))*(256-1e-4)).astype(int)
    print(touint8(freq))

    arr2im(touint8(freq), complete_image)

def arr2im(data, fname):
    out = Image.new('RGB', data.shape[1::-1])
    data = np.delete(data, len(data-1))
    out.putdata(list(map(tuple, data.reshape(-1, 3))))
    print(list(map(tuple, data.reshape(-1, 3))))
    out.save(fname)
       
#
#dark_image_grey = rgb2gray(dark_image)
#plt.figure(num=None, figsize=(8, 6), dpi=80)
#plt.imshow(dark_image_grey, cmap='gray');
