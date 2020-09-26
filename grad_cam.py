from models import GradCAM, MainModel
import torch
from argparse import Namespace
import numpy as np
from skimage.io import imsave, imread
import os

# grayscale to jet
def jet(image):
    n = 4 * image[:, :1]
    r = torch.clamp(torch.min(n-1.5,-n+4.5), 0, 1)
    g = torch.clamp(torch.min(n-0.5,-n+3.5), 0, 1)
    b = torch.clamp(torch.min(n+0.5,-n+2.5), 0, 1)
    return torch.cat((r,g,b), 1)


def main(image, signal, arch, model):
    # load image and convert to tensor
    ind = torch.tensor([[signal]])
    grad_cam = GradCAM(model)
    cam = grad_cam(image, ind)
    # output image with cam
    cam = jet(cam)
    image = torch.clamp(image * 0.315 + 0.188, 0, 1)
    imsave('%s.png' % opts.img_id, np.around(image[0,0].cpu().numpy()*255).astype(np.uint8))
    image = image + cam
    image = np.moveaxis(image[0].cpu().numpy(), 0, 2)
    image = image / image.max()
    image = np.around(image*255).astype(np.uint8)
    imsave('%s-cam.png' % opts.img_id, image)


if __name__ == '__main__':
    # opts = Namespace(
    #     data_dir='../../../',
    #     use_gpu=False,
    #     arch='densenet201',
    #     img_id='',
    #     ind=4,
    #     model_path='../../../trained_models/model_densenet201.pt'
    # )
