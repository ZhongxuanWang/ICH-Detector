import pydicom as dm
import os

def preprocess(path):
    if not os.path.isfile(path):
        print('Skipped one')

    image = dm.dcmread(path)

    if isinstance(image.WindowCenter, dm.multival.MultiValue):
        image.WindowCenter = image.WindowCenter[0]
        image.WindowWidth = image.WindowWidth[0]

    arr_hu = float(image.RescaleSlope) * image.pixel_array + float(image.RescaleIntercept)

    v_min = float(image.WindowCenter) - 0.5 * float(image.WindowWidth)
    v_max = float(image.WindowCenter) + 0.5 * float(image.WindowWidth)

    arr_hu_win = arr_hu.copy()

    #     Make the image array from 0 - 1
    arr_hu_win[arr_hu < v_min] = v_min
    arr_hu_win[arr_hu > v_max] = v_max
    arr_hu_win = (arr_hu_win - v_min) / (v_max - v_min)
    return arr_hu_win