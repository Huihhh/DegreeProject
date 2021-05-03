from .eurosat import EuroSat
from .iris import Iris
from .sat import Sat
from .syntheticData import SyntheticData

DATA = {
    'eurosat': EuroSat,
    'iris': Iris,
    'sat4': Sat,
    'circles': SyntheticData,
    'moons': SyntheticData,
    'spiral': SyntheticData,
    'sphere': SyntheticData,
}