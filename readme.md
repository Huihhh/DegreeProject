requirements:
`pip install tensorboardX`
`pip install hydra-core --upgrade`

## Step 1 Upload data artifacts (optional)
```shell
python 
```

train:
python train.py EXPERIMENT.log_path='./outputs/' DATASET.equal_density=False EXPERIMENT.n_epoch=500 MODEL.fc_winit='he_normal'

visualize linear regions:
python visualize.py hydra/job_logging=disabled EXPERIMENT.log_path='./outputs/' EXPERIMENT.resume_checkpoints='./checkpoints/ToyExperiment_epoch_499.pth.tar'

