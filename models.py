import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet101
from torchvision.models.densenet import densenet201
from torchvision.models.vgg import vgg19


class MainModel(nn.Module):

    def __init__(self, arch, num_classes, pretrained=False):
        super().__init__()
        self.loaded = False
        self.arch = arch
        model = globals()[arch](pretrained=pretrained)
        if arch.startswith('resnet'):
            n_feat = model.fc.in_features
            model.fc = nn.Sequential()
        elif arch.startswith('densenet'):
            n_feat = model.classifier.in_features
            model.classifier = nn.Sequential()
        elif arch.startswith('vgg'):
            n_feat = 1000  # 曾经是1000
            # model.classifier[-1] = nn.Sequential()
        else:
            raise Exception('unkown architecture')
        self.feature = model
        self.fc = nn.Linear(n_feat, num_classes)

    def forward(self, x):
        feat = self.feature(x)
        logit = self.fc(feat)
        return logit

    def load_state_dict(self, state_dict,
                        strict: bool = True):
        self.loaded = True
        super().load_state_dict(state_dict, strict)


class GradCAM(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.model.eval()

    def forward(self, x, ind):
        feat, logit = self.model_forward(x)
        logit = torch.gather(logit, 1, ind)
        grad = torch.autograd.grad(logit.sum(), feat)[0]
        with torch.no_grad():
            weights = grad.mean((2, 3), keepdim=True)  # N x C x 1 x 1
            cam = (weights * feat).sum(1, keepdim=True)  # N x 1 x h x w
            cam = F.relu(cam)
            cam = F.interpolate(cam, x.shape[2:], mode='bilinear', align_corners=True)  # N x 1 x H x W
            cam = cam - cam.min()
            cam = cam / cam.max()
        return cam

    def model_forward(self, x):
        m = self.model.feature
        if self.model.arch.startswith('resnet'):
            with torch.no_grad():
                x = m.conv1(x)
                x = m.bn1(x)
                x = m.relu(x)
                x = m.maxpool(x)
                x = m.layer1(x)
                x = m.layer2(x)
                x = m.layer3(x)
                x = m.layer4(x)
            feat = x
            feat.requires_grad = True
            x = F.adaptive_avg_pool2d(feat, (1, 1))
            x = torch.flatten(x, 1)
            logit = self.model.fc(x)
        elif self.model.arch.startswith('densenet'):
            with torch.no_grad():
                feat = m.features(x)
                feat = F.relu(feat, inplace=True)
            feat.requires_grad = True
            out = F.adaptive_avg_pool2d(feat, (1, 1))
            out = torch.flatten(out, 1)
            logit = self.model.fc(out)
        elif self.model.arch.startswith('vgg'):
            with torch.no_grad():
                feat = m.features(x)
                feat = m.avgpool(feat)
            feat.requires_grad = True
            out = torch.flatten(feat, 1)
            out = m.classifier(out)
            logit = self.model.fc(out)
        else:
            raise Exception('unkown architecture')
        return feat, logit
