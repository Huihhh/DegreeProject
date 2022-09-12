from nn_models.dnn import SimpleNet
from nn_models.resnet import ResNet
from nn_models.sResnet import SResNet

MODEL = {
    'shallow_nn': SimpleNet,
    'resnet': ResNet,
    'sResnet': SResNet
}