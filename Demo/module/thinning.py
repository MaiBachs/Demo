import matplotlib.pyplot as plt
from skimage import color
import numpy as np
import cv2
from PIL import Image
from skimage.util import invert
from PIL import Image
import time
import os
import glob

_errstr = "Mode is unknown or incompatible with input array shape."


def bytescale(data, cmin=None, cmax=None, high=255, low=0):
    """
    Byte scales an array (image).
    Byte scaling means converting the input image to uint8 dtype and scaling
    the range to ``(low, high)`` (default 0-255).
    If the input image already has dtype uint8, no scaling is done.
    This function is only available if Python Imaging Library (PIL) is installed.
    Parameters
    ----------
    data : ndarray
        PIL image data array.
    cmin : scalar, optional
        Bias scaling of small values. Default is ``data.min()``.
    cmax : scalar, optional
        Bias scaling of large values. Default is ``data.max()``.
    high : scalar, optional
        Scale max value to `high`.  Default is 255.
    low : scalar, optional
        Scale min value to `low`.  Default is 0.
    Returns
    -------
    img_array : uint8 ndarray
        The byte-scaled array.
    Examples
    --------
    >>> from scipy.misc import bytescale
    >>> img = np.array([[ 91.06794177,   3.39058326,  84.4221549 ],
    ...                 [ 73.88003259,  80.91433048,   4.88878881],
    ...                 [ 51.53875334,  34.45808177,  27.5873488 ]])
    >>> bytescale(img)
    array([[255,   0, 236],
           [205, 225,   4],
           [140,  90,  70]], dtype=uint8)
    >>> bytescale(img, high=200, low=100)
    array([[200, 100, 192],
           [180, 188, 102],
           [155, 135, 128]], dtype=uint8)
    >>> bytescale(img, cmin=0, cmax=255)
    array([[91,  3, 84],
           [74, 81,  5],
           [52, 34, 28]], dtype=uint8)
    """
    if data.dtype == np.uint8:
        return data

    if high > 255:
        raise ValueError("`high` should be less than or equal to 255.")
    if low < 0:
        raise ValueError("`low` should be greater than or equal to 0.")
    if high < low:
        raise ValueError("`high` should be greater than or equal to `low`.")

    if cmin is None:
        cmin = data.min()
    if cmax is None:
        cmax = data.max()

    cscale = cmax - cmin
    if cscale < 0:
        raise ValueError("`cmax` should be larger than `cmin`.")
    elif cscale == 0:
        cscale = 1

    scale = float(high - low) / cscale
    bytedata = (data - cmin) * scale + low
    return (bytedata.clip(low, high) + 0.5).astype(np.uint8)

def toimage(arr, high=255, low=0, cmin=None, cmax=None, pal=None,
            mode=None, channel_axis=None):
    data = np.asarray(arr)
    if np.iscomplexobj(data):
        raise ValueError("Cannot convert a complex-valued array.")
    shape = list(data.shape)
    valid = len(shape) == 2 or ((len(shape) == 3) and
                                ((3 in shape) or (4 in shape)))
    if not valid:
        raise ValueError("'arr' does not have a suitable array shape for "
                         "any mode.")
    if len(shape) == 2:
        shape = (shape[1], shape[0])  # columns show up first
        if mode == 'F':
            data32 = data.astype(np.float32)
            image = Image.frombytes(mode, shape, data32.tostring())
            return image
        if mode in [None, 'L', 'P']:
            bytedata = bytescale(data, high=high, low=low,
                                 cmin=cmin, cmax=cmax)
            image = Image.frombytes('L', shape, bytedata.tostring())
            if pal is not None:
                image.putpalette(np.asarray(pal, dtype=np.uint8).tostring())
                # Becomes a mode='P' automagically.
            elif mode == 'P':  # default gray-scale
                pal = (np.arange(0, 256, 1, dtype=np.uint8)[:, np.newaxis] *
                       np.ones((3,), dtype=np.uint8)[np.newaxis, :])
                image.putpalette(np.asarray(pal, dtype=np.uint8).tostring())
            return image
        if mode == '1':  # high input gives threshold for 1
            bytedata = (data > high)
            image = Image.frombytes('1', shape, bytedata.tostring())
            return image
        if cmin is None:
            cmin = np.amin(np.ravel(data))
        if cmax is None:
            cmax = np.amax(np.ravel(data))
        data = (data*1.0 - cmin)*(high - low)/(cmax - cmin) + low
        if mode == 'I':
            data32 = data.astype(np.uint32)
            image = Image.frombytes(mode, shape, data32.tostring())
        else:
            raise ValueError(_errstr)
        return image

    # if here then 3-d array with a 3 or a 4 in the shape length.
    # Check for 3 in datacube shape --- 'RGB' or 'YCbCr'
    if channel_axis is None:
        if (3 in shape):
            ca = np.flatnonzero(np.asarray(shape) == 3)[0]
        else:
            ca = np.flatnonzero(np.asarray(shape) == 4)
            if len(ca):
                ca = ca[0]
            else:
                raise ValueError("Could not find channel dimension.")
    else:
        ca = channel_axis

    numch = shape[ca]
    if numch not in [3, 4]:
        raise ValueError("Channel axis dimension is not valid.")

    bytedata = bytescale(data, high=high, low=low, cmin=cmin, cmax=cmax)
    if ca == 2:
        strdata = bytedata.tostring()
        shape = (shape[1], shape[0])
    elif ca == 1:
        strdata = np.transpose(bytedata, (0, 2, 1)).tostring()
        shape = (shape[2], shape[0])
    elif ca == 0:
        strdata = np.transpose(bytedata, (1, 2, 0)).tostring()
        shape = (shape[2], shape[1])
    if mode is None:
        if numch == 3:
            mode = 'RGB'
        else:
            mode = 'RGBA'

    if mode not in ['RGB', 'RGBA', 'YCbCr', 'CMYK']:
        raise ValueError(_errstr)

    if mode in ['RGB', 'YCbCr']:
        if numch != 3:
            raise ValueError("Invalid array shape for mode.")
    if mode in ['RGBA', 'CMYK']:
        if numch != 4:
            raise ValueError("Invalid array shape for mode.")

    # Here we know data and mode is correct
    image = Image.frombytes(mode, shape, strdata)
    return image
    
def zeroToOne(thin_image,i,j):
	p2 = thin_image[i-1][j-1]
	p3 = thin_image[i-1][j]
	p4 = thin_image[i-1][j+1]
	p5 = thin_image[i][j+1]
	p6 = thin_image[i+1][j+1]
	p7 = thin_image[i+1][j]
	p8 = thin_image[i+1][j-1]
	p9 = thin_image[i][j-1]
	count = 0;
	if(p2==0 and p3==1):
		count = count + 1
	if(p3==0 and p4==1):
		count = count + 1
	if(p4==0 and p5==1):
		count = count + 1
	if(p5==0 and p6==1):
		count = count + 1
	if(p6==0 and p7==1):
		count = count + 1
	if(p7==0 and p8==1):
		count = count + 1
	if(p8==0 and p9==1):
		count = count + 1
	if(p9==0 and p2==1):
		count = count + 1
	return count

def neighbours(x,y,image):
	img = image
	x_1, y_1, x1, y1 = x-1, y-1, x+1, y+1
	return [ img[x_1][y], img[x_1][y1], img[x][y1], img[x1][y1], img[x1][y], img[x1][y_1], img[x][y_1], img[x_1][y_1] ]

def neighbourst(thin_image, i, j):
	p2 = thin_image[i-1][j-1]
	p3 = thin_image[i-1][j]
	p4 = thin_image[i-1][j+1]
	p5 = thin_image[i][j+1]
	p6 = thin_image[i+1][j+1]
	p7 = thin_image[i+1][j]
	p8 = thin_image[i+1][j-1]
	p9 = thin_image[i][j-1]
	return p2,p3,p4,p5,p6,p7,p8,p9

def transitions(neighbours):
	n = neighbours + neighbours[0:1]
	return sum( (n1, n2) == (0, 1) for n1, n2 in zip(n, n[1:]) )

def foregroundPixels(image):
	fgp = 0
	row, col = image.shape
	for i in range(2, row-1):
			for j in range(2, col-1):
				if(image[i][j]==1):
					fgp = fgp + 1
	return fgp


def zsAlgoIterationOne(image):
	Image_Thinned = image.copy()
	changing1 = changing2 = 1
	i = 0
	while changing1 or changing2:
		changes_occured = 0
		changing1 = []
		rows, columns = Image_Thinned.shape
		for x in range(1, rows - 1):
			for y in range(1, columns - 1):
				P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, Image_Thinned)
				if (Image_Thinned[x][y] == 1 and 3 <= sum(n) <= 6 and transitions(n) == 1 and P2 * P4 * P6 == 0  and P4 * P6 * P8 == 0):
					changing1.append((x,y))
		for x, y in changing1: 
			Image_Thinned[x][y] = 0
			changes_occured = changes_occured + 1
		
		changing2 = []
		for x in range(1, rows - 1):
			for y in range(1, columns - 1):
				P2,P3,P4,P5,P6,P7,P8,P9 = n = neighbours(x, y, Image_Thinned)
				if (Image_Thinned[x][y] == 1 and 3 <= sum(n) <= 6 and transitions(n) == 1 and P2 * P4 * P8 == 0  and P2 * P6 * P8 == 0):
					changing2.append((x,y))    
		for x, y in changing2: 
			Image_Thinned[x][y] = 0
			changes_occured = changes_occured + 1
		i = i + 1
		print("Iteration: ", i , "changes_occured: ", changes_occured)
	return Image_Thinned

def sensitivitycheck(row, col, image, fg):
	sensitivity = 0
	for i in range(2, row-1):
		for j in range(2, col-1):
			if(image[i][j]==1):
				compute = zeroToOne(image,i,j)
				sensitivity = sensitivity + compute
	sensitivity = sensitivity/fg
	sensitivity = 1 - sensitivity
	return sensitivity

def thinnesscheck(image):
	row,col = image.shape
	thinny = 0
	for i in range(2, row-1):
		for j in range(2, col-1):
			if(image[i][j]==1):
				p1 = image[i][j]
				p2,p3,p4,p5,p6,p7,p8,p9 = neighbourst(image, i, j)
				compute = (p1*p9*p2) + (p1*p9*p8) + (p1*p8*p7) + (p1*p7*p6) + (p1*p6*p5) + (p1*p5*p4) + (p1*p4*p3) + (p1*p3*p2)
				thinny = thinny + compute
	denominator = (max(row,col)-1)*(max(row,col)-1)
	denominator = denominator/4
	thinness = thinny/denominator
	return thinness

def thinning():
	count = 0
	reduction_rate = 0
	sensitivity = 0
	thinness = 0
	start_time = time.time()
	for file in glob.glob("enhancered.png"):
		count = count+1
		image = cv2.imread(file)
		image = color.rgb2gray(image)
		row,col = image.shape
		image = invert(image)
		print(image.shape)
		fgps = foregroundPixels(image)
		print("fgps: ", fgps)
		skeleton = zsAlgoIterationOne(image)
		fgpst = foregroundPixels(skeleton)
		print("fgpst: ", fgpst)
		reduction_rate = reduction_rate + (((fgps-fgpst)/fgps)*100)
		sensitivity = sensitivity + sensitivitycheck(row, col, skeleton,fgps)
		thinnesst = thinnesscheck(skeleton)
		thinnesso = thinnesscheck(image)
		thinness = thinness + (1 - (thinnesst/thinnesso))
		im = toimage(skeleton)
		im.save("thinninged.png") #str(count)
	end_time = time.time()	# Used to stop time record
	seconds=end_time - start_time
	print ("Total time: ", seconds)
	print("Average time: ", seconds/count)
	print("Reduction rate: ", reduction_rate)
	print("Average Reduction rate: ", reduction_rate/count)
	print("sensitivity: ",sensitivity)
	print("Average sensitivty: ", sensitivity/count)
	print("Average thinness: ", thinness/count)