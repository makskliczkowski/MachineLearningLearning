from skimage.io import *
from tensorflow.python.client import device_lib
from keras import backend as K
from skimage.util import img_as_ubyte
import cv2
import numpy as np
import os
import pandas as pd


def image_fromURL_toGrey(x, size, prnt, directory):
    # Here we read the file from the url, convert it to greyscale of size
    # given by the user, check if it's aviable and if it's not return nan
    # temp = np.array(x)
    dir = directory
####!!!!!!!!!!!!!!!!!! PLEASE NOTE !!! NOT GRAY ANYMORE BUT DON"T WANT TO CHANGE THE NAME!
    def applier(url, size2=size, shall_print=prnt):
        url2 = (url[:-4] + 'jpg').replace('html', 'detail')
        # img = Image.open(requests.get(url2, stream=True).raw).convert('L')
        try:
            tempname = url2.split('/')
            filename = tempname[-2] + '_' + tempname[-1]
            if os.path.isfile(dir + "/" + filename):
                return filename
            if shall_print:
                img1 = imread(url2, as_gray=False)
                img1 = cv2.resize(img1, (size2, size2), interpolation=cv2.INTER_CUBIC)
                img1 = img_as_ubyte(img1)
                imsave(dir + "/" + filename, img1)
            if os.path.isfile(dir + "/" + filename):
                return filename
            else:
                return np.nan
        except Exception as e:
            # print(e)
            return np.nan

    # print(temp)
    return np.vectorize(applier)(x)


#
def putImageToDb_CV2(x, directory):
    # @numba.jit
    def applier(url):
        url2 = (url[:-4] + 'jpg').replace('html', 'detail')
        dir = directory
        tempname = url2.split('/')
        filename = tempname[-2] + '_' + tempname[-1]
        try:
            image = cv2.imread(dir + '/' + filename)
            return image
        except:
            return np.nan

    return np.vectorize(applier)(x)
