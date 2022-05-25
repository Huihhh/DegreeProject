import logging
import io

import numpy as np
import pandas as pd
import torch

from omegaconf import DictConfig, OmegaConf
import hydra
from pathlib import Path
import wandb
import yaml
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.preprocessing import LabelEncoder

from datasets.dataset import Dataset
from models import *
from utils import flat_omegadict, set_random_seed
from utils import get_signatures, visualize_signatures

logger = logging.getLogger(__name__)
label_encoder = LabelEncoder()
api = wandb.Api()

def visualize_LR_group(cfg_data: dict, cfg_model: dict, archs: list[list[int]], num_seeds: int, num_lr: pd.DataFrame, device: torch.device, use_artifact_model: bool = True):
    '''
    Generating data for visualizing linear regions of models at initialization or after training.
    Figure 1: plotly subplots of heatmaps of linear regions for different models, return html
    Figure 2: plotly boxplot of #linear regions for different models, return the pd.dataframe

    Parameters
    ------------
    * cfg_data: data config for generating LitDataModule
    * cfg_model: dict, model config used to initialize the model
    * archs: list of hidden nodes of models, used to query runs and title subplots
    * num_seeds: to generate random seeds by range(num_seeds) for random models
    * use_artifact_model: bool, default True, models are trained
    '''
    # *********************************** prapare data *********************************
    # new Dataset for circles, moons, spirals
    dataset = Dataset(seed=100, n_train=0.7, n_val=0.1,
                      n_test=0.2, **cfg_data)
    input_dim = dataset.trainset[0][0].shape[0]
    grid_points, grid_labels = dataset.grid_data
    xrange = np.unique(grid_points[:, 0])
    yrange = np.unique(grid_points[:, 1])

    stage = 'trained' if use_artifact_model else 'untrained'
    subplots = []
    for n, h_nodes in enumerate(archs):
        for seed in range(1, num_seeds+1): #
            # *********************************** build model *********************************
            if use_artifact_model:
                runs = api.runs('degree-project-decision-boundary', filters={
                    'config.seed': seed,
                    'config.DATASET_name': cfg_data['name'],
                    'config.MODEL_h_nodes': list(h_nodes),
                    })
                print(len(runs))
                model_path = runs[-1].logged_artifacts()[0].download()
                model = torch.load(Path(model_path) / 'model.h5')
                logger.info("[Model] Loading model -- input dim: {}, hidden layers: {}, out dim: {}"
                            .format(input_dim, h_nodes, cfg_model.out_dim))
            else:
                model = MODEL[cfg_model.name](
                    input_dim=input_dim, seed=seed, h_nodes=h_nodes, **cfg_model)
                logger.info("[Model] Building model -- input dim: {}, hidden layers: {}, out dim: {}"
                            .format(input_dim, h_nodes, cfg_model.out_dim))
            
            model = model.to(device=device)
            _, grid_sigs, _ = get_signatures(torch.tensor(
                grid_points).to(device), model, device)

            row = {
                'trained': use_artifact_model,
                'seed': seed,
                'linear region density': len(torch.unique(grid_sigs, dim=0))/cfg_data['input_volume'],
                'hidden layers': str(h_nodes),
                'data': cfg_data['name'],
                'data-stage': '-'.join([cfg_data['name'], stage])
            }
            num_lr = num_lr.append(row, ignore_index=True)

        # * visaulize linear regions from last seed
        grid_sigs = np.array([''.join(str(x)
                             for x in s.tolist()) for s in grid_sigs])
        #  * categorial to numerical
        region_labels = label_encoder.fit_transform(
            grid_sigs).reshape(grid_labels.shape)
        region_labels_shuffle = np.zeros_like(region_labels)
        # # shuffle labels to visualy differentiate adjacent regions
        random_labels = np.random.permutation(region_labels.max() + 1)
        for i, label in enumerate(random_labels):
            region_labels_shuffle[np.where(region_labels == i)] = label
        subplot = visualize_signatures(region_labels_shuffle, grid_labels, xrange, yrange,
                                       showscale=n == 0,
                                       colorbar_len=1.04,
                                       colorbary=0.5
                                       )
        subplot_points_stacked = visualize_signatures(region_labels_shuffle, grid_labels, xrange, yrange,
                                showscale=False,
                                colorbar_len=1,
                                colorbary=0.5,
                                data=dataset.trainset.tensors
                                )
        subplots.append([subplot, subplot_points_stacked])

    fig = make_subplots(
        cols=len(archs),
        rows=2,
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.01,
        horizontal_spacing=0.01,
        subplot_titles=[str(arch) for arch in archs],  # confidence map
        # column_widths=[0.51, 0.49],
        # row_heights=[0.5, 0.5]
    )

    for c, col in enumerate(subplots):
        for layer in col[0]:
            fig.add_trace(layer, 1, c+1)
        for layer in col[1]:
            fig.add_trace(layer, 2, c+1)

    fig.update_layout(
        # yaxis=dict(scaleanchor='x', scaleratio=1),  # set aspect ratio to 1, actually done by width and column_widths
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        width=1800 if cfg_data['name'] == 'moons' else 1600,
        height=700,
        showlegend=False,
        yaxis_title=stage,
    )
    for row in range(2):
        for col in range(len(subplots)):
            fig.update_yaxes(range=[yrange[0],yrange[-1]], row=row+1, col=col+1)
            fig.update_xaxes(range=[xrange[0],xrange[-1]], row=row+1, col=col+1)
    # fig.show()
    buffer = io.StringIO()
    fig.write_html(buffer)
    html = buffer.getvalue()
    return html, num_lr


@hydra.main(config_path='./config', config_name='decision_boundary')
def main(CFG: DictConfig) -> None:
    # initial logging file
    logger.info(OmegaConf.to_yaml(CFG))
    config = flat_omegadict(CFG)
    wandb.init(project=CFG.wandb_project, name=CFG.run_name, config=config) 

    set_random_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available()
                        and CFG.use_gpu else 'cpu')
    root = Path(hydra.utils.get_original_cwd())

    num_lr = pd.DataFrame(columns=['seed', 'linear region density', 'hidden layers'])
    for name in ['circles', 'moons', 'spiral']:
        filename = name+'.yaml'
        with open(root / 'config'/'dataset'/filename, 'r') as stream:
            try:
                DATASET = yaml.safe_load(stream)['DATASET']
            except yaml.YAMLError as exc:
                print(exc)

        html, num_lr = visualize_LR_group(
            DATASET, CFG.MODEL, CFG.archs[name], CFG.num_seeds, num_lr, device, False) # results at initialization
        wandb.log({f"LinearRegions/{DATASET['name']}_untrained": wandb.Html(html)})
        
        html, num_lr = visualize_LR_group(
            DATASET, CFG.MODEL, CFG.archs[name], CFG.num_seeds, num_lr, device, True) # results after train
        wandb.log({f"LinearRegions/{DATASET['name']}_trained": wandb.Html(html)})

    # for stage in ['untrained', 'trained']:
    #     trained = stage == 'trained'
    #     fig = px.box(num_lr[num_lr['trained']==trained], x='data-stage', y='linear region density',
    #                 color='hidden layers')
    #     # fig.update_layout(
    #     #     paper_bgcolor='rgba(0,0,0,0)',
    #     #     plot_bgcolor='rgba(0,0,0,0)',
    #     # )
    #     # fig2.show()
    #     buffer = io.StringIO()
    #     fig.write_html(buffer)
    #     html = buffer.getvalue()
    #     wandb.log({f'linear region density - {stage}': wandb.Html(html)})
    fig = px.box(num_lr, x='data-stage', y='linear region density',
                color='hidden layers')
    # fig.update_layout(
    #     paper_bgcolor='rgba(0,0,0,0)',
    #     plot_bgcolor='rgba(0,0,0,0)',
    # )
    # fig2.show()
    buffer = io.StringIO()
    fig.write_html(buffer)
    html = buffer.getvalue()
    wandb.log({f'linear region density': wandb.Html(html)})
    wandb.finish()


if __name__ == '__main__':
    # ### Error: Initializing libiomp5.dylib, but found libiomp5.dylib already initialized.###
    # import os
    # os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    main()
