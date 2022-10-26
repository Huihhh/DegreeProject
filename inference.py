from distutils.command.config import config
import numpy as np
import pandas as pd
import torch

from omegaconf import DictConfig, OmegaConf
import hydra
from pathlib import Path
import logging
import os
import wandb
import yaml
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

from datasets.dataset import Dataset
from models import *
from trainer._base_trainer import Bicalssifier
from utils import flat_omegadict, set_random_seed
from utils import get_signatures, visualize_signatures

#  * set cuda
os.environ['CUDA_VISIBLE_DEVICES'] = '9'


@hydra.main(config_path='./config', config_name='test')
def main(CFG: DictConfig) -> None:
    # initial logging file
    logger = logging.getLogger(__name__)
    logger.info(OmegaConf.to_yaml(CFG))
    config = flat_omegadict(CFG)
    label_encoder = LabelEncoder()

    device = torch.device('cuda' if torch.cuda.is_available() and CFG.use_gpu else 'cpu')
    root = Path(hydra.utils.get_original_cwd())

    num_lr = pd.DataFrame(columns=['seed', '#linear_regions', 'h_nodes'])
    for name in ['circles', 'moons', 'spiral']:
        filename = name+'.yaml'
        with open(root / 'config'/'dataset'/filename, 'r') as stream:
            try:
                DATASET = yaml.safe_load(stream)['DATASET']
            except yaml.YAMLError as exc:
                print(exc)
        # get datasets
        dataset = Dataset(seed=100, n_train=0.7, n_val=0.1, n_test= 0.2, **DATASET)
        input_dim = dataset.trainset[0][0].shape[0]
        grid_points, grid_labels = dataset.grid_data
        xrange = np.unique(grid_points[:, 0])
        yrange = np.unique(grid_points[:, 1])

        set_random_seed(0)
        subplots = []
        for n, h_nodes in enumerate(CFG.archs[name]):
            for seed in range(CFG.num_seeds):
                # build model 
                model = MODEL[CFG.MODEL.name](input_dim=input_dim,seed=seed, h_nodes=h_nodes, **CFG.MODEL)
                logger.info("[Model] Building model -- input dim: {}, hidden layers: {}, out dim: {}"
                                            .format(input_dim, h_nodes, CFG.MODEL.out_dim))
                model = model.to(device=device)

                _, grid_sigs, _ = get_signatures(torch.tensor(grid_points).float().to(device), model, device)

                row = {
                    'seed': seed, 
                    'linear region density': len(torch.unique(grid_sigs, dim=0))/DATASET['input_volume'],
                    'hidden layers': str(h_nodes),
                    'data': name,
                }
                num_lr = num_lr.append(row, ignore_index=True)    

            # * visaulize linear regions from last seed
            grid_sigs = np.array([''.join(str(x) for x in s.tolist()) for s in grid_sigs])
            #  * categorial to numerical
            region_labels = label_encoder.fit_transform(grid_sigs).reshape(grid_labels.shape)
            # shuffle labels to visualy differentiate adjacent regions
            random_labels = np.random.permutation(region_labels.max() +1)
            for i, label in enumerate(random_labels):
                region_labels[np.where(region_labels==i)] = label
            subplot = visualize_signatures(region_labels, grid_labels, xrange, yrange, 
                                    showscale= n==0,
                                    colorbar_len=1,
                                    colorbary=0.5
                                    )
            subplots.append(subplot)
                
        # fig = make_subplots(
        #     cols=len(CFG.MODEL.archs), 
        #     rows=1,
        #     shared_xaxes=True, 
        #     shared_yaxes=True, 
        #     vertical_spacing = 0.01,
        #     horizontal_spacing = 0.01,
        #     subplot_titles=[str(arch) for arch in CFG.MODEL.archs],#confidence map
        #     # column_widths=[0.51, 0.49],
        #     # row_heights=[0.5, 0.5]
        #     )
        
        # for i, subplot in enumerate(subplots):
        #     for layer in subplot:
        #         fig.add_trace(layer, 1, i+1)

        # fig.update_layout(
        #     # yaxis=dict(scaleanchor='x', scaleratio=1),  # set aspect ratio to 1, actually done by width and column_widths
        #     paper_bgcolor='rgba(0,0,0,0)',
        #     # plot_bgcolor='rgba(0,0,0,0)',
        #     width=1800 if dataset.name=='moons' else 1200,
        #     height=400,
        #     showlegend=False,
        # )
        # fig.show()  
    fig2 = px.box(num_lr, x='data', y='linear region density', color='hidden layers')
    fig2.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    fig2.show()
        
    
    wandb.init(project=CFG.wandb_project, name=CFG.run_name, config=config )
    



    wandb.finish()




if __name__ == '__main__':
    # ### Error: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.###
    # import os
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    main()
