from .dnn import SimpleNet
from .resnet import ResNet
from .sResnet import SResNet

MODEL = {
    'shallow_nn': SimpleNet,
    'resnet': ResNet,
    'sResnet': SResNet
}