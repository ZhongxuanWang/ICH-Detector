import pydicom as dm
import os

from models import GradCAM
import torch
import numpy as np
from main import THRESHOLD

ct_mean = 0.188
ct_std = 0.315


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


def img_tune(img):
    img = torch.tensor(img[None, None, ...], dtype=torch.float32) / 255
    img = (img - ct_mean) / ct_std
    return img


def rgb_to_grey(img):
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray


def jet(image):
    n = 4 * image[:, :1]
    r = torch.clamp(torch.min(n - 1.5, -n + 4.5), 0, 1)
    g = torch.clamp(torch.min(n - 0.5, -n + 3.5), 0, 1)
    b = torch.clamp(torch.min(n + 0.5, -n + 2.5), 0, 1)
    return torch.cat((r, g, b), 1)


def gc_(image, ind, model):
    image = image.expand(1,3,-1,-1)
    ind = torch.tensor([[ind]])
    grad_cam = GradCAM(model)
    cam = grad_cam(image, ind)
    image = image + cam
    image = image / image.max()
    return image[0].numpy()


def gc(image, model):
    if isinstance(image, torch.tensor.__class__):
        image = np.array(image)
    if image.shape == 3:
        image = image[0]
    indexes = [image]
    for ind in range(6):
        indexes.append(np.nan_to_num(gc_(torch.tensor(image), ind, model)[0]))
    indexes = torch.tensor(indexes)
    return indexes


def get_results(result):
    result = torch.sigmoid(result)[0].detach().numpy()
    flag = [None, None, None, None, None, None]
    for i, single_result in enumerate(result):
        if single_result >= THRESHOLD[3][i]:
            flag[i] = 'red'
        else:
            flag[i] = 'green'

    results = {
        "Any": result[0],
        "Epidural": result[1],
        "Intraparenchymal": result[2],
        "Intraventricular": result[3],
        "Subarachnoid": result[4],
        "Subdural": result[5],
    }

    # WHY DOING THIS IS GOOD, BUT I CAN'T DO IN THE FOR LOOP ABOVE???????
    i = 0
    for (key, item) in results.items():
        results[key] = round(result[i].item()* 100, 4)
        i += 1

    return results, flag
