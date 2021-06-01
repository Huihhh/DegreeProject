from .dnn import SimpleNet
from .resnet import ResNet

MODEL = {
    'shallow_nn': SimpleNet,
    'resnet': ResNet
}