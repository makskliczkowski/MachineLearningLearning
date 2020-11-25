import pandas as pd
import numpy as np
import os
import sys
import swifter
from ._image_handler import *

file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)


def createImageDatabase(dir, im_dir, download=False, want_whole_im=False, howmany=20000):
    db = pd.read_excel(dir, nrows=howmany)
    db['URL'] = db['URL'].where(db['URL'].str.endswith('html')).dropna()
    # db['URL'] = db.swifter.apply(lambda x: image_fromURL_toGrey(x['URL'], 128), axis=1, raw=True)
    if not want_whole_im:
        if download or len(os.listdir(im_dir)) == 0:
            # we put files into database, if the files are downloaded yet we can skip it
            print('Downloading files and saving them in ' + im_dir + '\n')
            db['URL'] = db['URL'].swifter.apply(image_fromURL_toGrey, args=[128, True, im_dir])

        else:
            # we print the files to the folder, and put them into database, if the files are downloaded yet we can
            # skip it
            print('Taking files and just putting their names in the dataframe\n')
            db['URL'] = db['URL'].swifter.apply(image_fromURL_toGrey, args=[128, False, im_dir])
        db = (db.dropna()
              .rename(columns={'URL': 'Images'}))
    else:
        print('Putting images from folder ' + im_dir + ' to dataframe\n')
        db['Image'] = (db['URL'].swifter.apply(putImageToDb_CV2, args=[im_dir])
                       .dropna())
    print('Finished checking folder\n')
    db['ID'] = db.groupby(['TYPE']).ngroup()
    # print(db)
    # db.set_index('Images', inplace=True)
    return db
