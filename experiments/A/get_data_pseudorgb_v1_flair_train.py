# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-09-09T21:54:46.895354Z","iopub.execute_input":"2021-09-09T21:54:46.895820Z","iopub.status.idle":"2021-09-09T21:54:46.940328Z","shell.execute_reply.started":"2021-09-09T21:54:46.895786Z","shell.execute_reply":"2021-09-09T21:54:46.939419Z"}}
import os
import re
import numpy as np 
import glob
import gc
import pandas as pd 
import pydicom as dicom
import seaborn as sns
import joblib
import matplotlib.pyplot as plt

from tqdm import tqdm
from scipy.ndimage import zoom


def load_source_dicom_line(path):
    t_paths = sorted(
        glob.glob(os.path.join(path, "*")), 
        key=lambda x: int(x[:-4].split("-")[-1]),
    )
    images = []
    for filename in t_paths:
        data = dicom.read_file(filename)
        #if data.pixel_array.max() == 0:
        #    continue
        images.append(data)
        
    return images


def load_dicom_xyz(path):
    _3d = load_source_dicom_line(path)
    # get metadata orientation
    x1, y1, _, x2, y2, _ = [round(j) for j in _3d[0].ImageOrientationPatient]
    cords = [x1, y1, x2, y2]
    position_first, position_last = _3d[0].ImagePositionPatient, _3d[-1].ImagePositionPatient
    # get main metadata
    spacing_z = np.abs(float(_3d[-1].SliceLocation) - float(_3d[0].SliceLocation)) / len(_3d)
    spacing_x_y = _3d[0].PixelSpacing
    assert spacing_x_y[0] == spacing_x_y[1]
    # form tensor
    _3d = [np.expand_dims(i.pixel_array, axis=0) for i in _3d]
    _3d = np.concatenate(_3d)
    # rescale
    _3d = zoom(_3d, (spacing_z/spacing_x_y[0], 1, 1))#first axis - rescaled
    
    # reorder planes if needed and rotate voxel
    if cords == [1, 0, 0, 0]:
        if position_first[1] < position_last[1]:
            _3d = _3d[::-1] 
        _3d = _3d.transpose((1, 0, 2))
        
    elif cords == [0, 1, 0, 0]:
        if position_first[0] < position_last[0]:
            _3d = _3d[::-1]
        _3d = _3d.transpose((1, 2, 0))
        _3d = np.rot90(_3d, 2, axes=(1, 2))
        
    elif cords == [1, 0, 0, 1]:
        if position_first[2] > position_last[2]:
            _3d = _3d[::-1]
        _3d = np.rot90(_3d, 2)

    return _3d
    

def get_pseudo_rgb(img):
    #img = np.clip(img, min_a, max_a)
    img_nan = np.where(img == img.min(), np.nan, img)

    pimages = []
    for i in range(3):
        # get pseudorgb shape
        shape_2_part = list(np.nanmax(img_nan, axis=i).shape)
        shape = [3] + shape_2_part
        
        prgb = np.ones(shape)
        prgb[0, :, :] = np.nanmean(img_nan, axis=i)
        prgb[0, :, :] /= np.nanmax(prgb[0, :, :])
        prgb[1, :, :] = np.nanmax(img_nan, axis=i)
        prgb[1, :, :] /= np.nanmax(prgb[1, :, :])
        prgb[2, :, :] = np.nanstd(img_nan, axis=i)
        prgb[2, :, :] /= np.nanmax(prgb[2, :, :])
        
        prgb = np.swapaxes(prgb, 0, 2)
        prgb = np.swapaxes(prgb, 0, 1)
        prgb = np.where(np.isnan(prgb), 0, prgb)
        prgb = np.clip(prgb, -1, 1)
        pimages.append(prgb)
    
    return pimages


def create_pseudorgb_set(img):
    pimgs = get_pseudo_rgb(img)
    
    new_pimgs = []
    for p in pimgs:
        # find brain bbox
        y_shape, x_shape = p.shape[0], p.shape[1]
        x_min, x_max, y_min, y_max = 1e3, -1, 1e3, -1 
        for yi in range(y_shape):
            for xi in range(x_shape):
                if not np.sum(np.abs(p[yi, xi, :])) == 0:
                    x_min = xi if xi < x_min else x_min
                    y_min = yi if yi < y_min else y_min
                    x_max = xi if xi > x_max else x_max
                    y_max = yi if yi > y_max else y_max
                    
        # place in canvas
        canvas = np.zeros((512, 512, 3))
        x_size, y_size = x_max - x_min, y_max - y_min
        start_x, start_y = (512-x_size)//2, (512-y_size)//2
        canvas[start_y:start_y+y_size, start_x:start_x+x_size, :] = p[y_min:y_max, x_min:x_max, :]

        new_pimgs.append(canvas.astype('float32'))
        
    return new_pimgs
                    
    

if __name__ == "__main__":
    DATA_PATH = "../input/rsna-miccai-brain-tumor-radiogenomic-classification/"
    df_path = DATA_PATH + "train_labels.csv"
    img_path = DATA_PATH + "train/"
    data_savepath = "./pseudorgb_v1_flair_train.joblib"
    
    df = pd.read_csv(df_path)
    pseudo_rgb_data = dict()
    for i in tqdm(df['BraTS21ID'].tolist()):
        path = img_path + str(i).zfill(5) + '/' + 'FLAIR/'

        try: 
            img = load_dicom_xyz(path)
            pimgs = create_pseudorgb_set(img)
        except:
            print(f"Bad id: {i}")
            pimgs = -1

        pseudo_rgb_data[i] = pimgs
        gc.collect()
        
    joblib.dump(pseudo_rgb_data, data_savepath)
    print(pseudo_rgb_data)
        