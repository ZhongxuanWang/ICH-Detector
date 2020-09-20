from models import GradCAM
import torch
import numpy as np
from models import MainModel


def gc_(image, ind, path):
    if isinstance(image, np.ndarray):
        image = torch.tensor(image)
    image = image.expand(1,3,-1,-1)
    ind = torch.tensor([[ind]])
    model = MainModel('densenet201', 6)
    model.load_state_dict(torch.load(path))
    grad_cam = GradCAM(model)
    cam = grad_cam(image, ind)
    image = image + cam
    image = image / image.max()
    return image[0].numpy()

# def gc(image):
#     indexes = []
#     indexes.append(image.numpy()[0])
#     for ind in range(6):
#         indexes.append(np.nan_to_num(gc_(image, ind)[0]))
#     indexes = torch.tensor(indexes)
#     return indexes
