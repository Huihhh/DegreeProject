from .dnn import SimpleNet
from .resnet import ResNet

MODEL = {
    'shallow_dnn': SimpleNet,
    'resnet': ResNet
}