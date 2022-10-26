from models.dnn import SimpleNet
from models.resnet import ResNet
from models.sResnet import SResNet

MODEL = {
    'shallow_nn': SimpleNet,
    'resnet': ResNet,
    'sResnet': SResNet
}