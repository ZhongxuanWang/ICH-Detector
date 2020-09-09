import os
import torch
import numpy as np  # linear algebra
from models.code.models import MainModel
from argparse import Namespace
from PIL import Image
from torchvision import transforms

ct_mean = 0.188
ct_std = 0.315


from time import time

BASE = os.path.dirname(os.path.abspath(__file__))

MODEL = BASE + "/models/model_densenet201.pt"

state_dict = torch.load(MODEL)


def get_predict(slices_0, flag):
    image_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([ct_mean], [ct_std], inplace=True),
        ]
    )
    if flag == 0:
        image = np.stack(slices_0.pixel_array)
        image = image.astype(np.float32)
        image = Image.fromarray(image)
    else:
        image = Image.open(slices_0)

    opts = Namespace(
        seed=123,
        data_dir="../",
        num_workers=0,
        use_gpu=False,
        batch_size=32,
        arch="densenet201",
        pretrained=True,
        lr=1e-4,
        epochs=30,
        lr_decay_ep=31,
        is_train=False,
        time_start=time(),
    )
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MainModel(opts.arch, 6, pretrained=opts.pretrained)
    model.load_state_dict(state_dict)
    image_tensor = image_transform(image)
    image_tensor.unsqueeze_(0)
    image_tensor = image_tensor.to(device)

    out = model(image_tensor)
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
    return percentage
