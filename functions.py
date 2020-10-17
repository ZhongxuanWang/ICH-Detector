import pydicom as dm
import os

from models import GradCAM
import torch
import numpy as np

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


# def grad_cam(image, signal, model):
#     # print(f'Arch {model.arch}')
#     # load image and convert to tensor
#     ind = torch.tensor([[signal]])
#     grad_cam = GradCAM(model)
#     cam = grad_cam(image, ind)
#     # output image with cam
#     cam = jet(cam)
#     # image = torch.clamp(image * 0.315 + 0.188, 0, 1)
#     image = torch.clamp(image, 0, 1)
#     image = image + cam
#     image = np.moveaxis(image[0].cpu().numpy(), 0, 2)
#     image = image / image.max()
#     image = np.around(image * 255).astype(np.uint8)
#     return image
