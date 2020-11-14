from skimage.io import imread, imshow, imsave
from skimage.transform import resize
from skimage.util import img_as_ubyte


def image_fromURL_toGrey(url, size):
    # Here we read the file from the url, convert it to greyscale of size
    # given by the user, check if it's aviable and if it's not return nan
    dir = 'D:/Uni/SEMESTERS/MS/II/MonographicComputation/LAB/Paintings/Images'
    filename = 'nan'
    if url.endswith('html'):
        url2 = url[:-4]
        url2 += 'jpg'
        url2.replace('html', 'detail')
        img1 = imread(url2, as_gray=True)
        if img1.size == 0:
            url = filename
            return
        else:
            img1 = resize(img1, (size, size))
            img1 = img_as_ubyte(img1)
            tempname = url2.split('/')
            filename = '/' + tempname[-2] + '_' + tempname[-1]
            imsave(dir+filename, img1)
            url = filename
            return
    else:
        url = filename
        return

