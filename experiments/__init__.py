from .experiment import Experiment
from .litExperiment import LitExperiment

EXPERIEMTS = {
    'ignite': Experiment,
    'lightning': LitExperiment
}
