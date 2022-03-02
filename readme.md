## Install requirements
```shell
pip install -r requirements.txt
```

## Train
```python
python train.py --config-name sampleEfficiency \
    dataset=spiral \
    model=shallow_nn \
    init_methods@MODEL.fc_winit=he_normal \
    DATASET.n_samples=$SLURM_ARRAY_TASK_ID \
    MODEL.h_nodes=[16,16,16,16] \
    EXPERIMENT.wandb_project=degree-project-Spiral \
    EXPERIMENT.name=sample-efficiency \
    EXPERIMENT.plot_LR=False \
    EXPERIMENT.n_epoch=500 \
    seed=$seed
```


