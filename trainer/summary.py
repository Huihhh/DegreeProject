import enum
import io
import logging
from typing import Any
from unicodedata import decimal
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
import hydra
from plotly.subplots import make_subplots
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from utils import get_signatures, visualize_signatures
from models import *

from datasets.dataset import Dataset

api = wandb.Api()
logger = logging.getLogger(__name__)
label_encoder = LabelEncoder()

def fig2html(fig) -> Any:
    '''
    write plotly figure to buffer to get html

    Parameter
    ----------
    fig: plotly figure

    Return
    ---------
    html that can be passed to wandb.Html(html)
    '''
    buffer = io.StringIO()
    fig.write_html(buffer)
    return buffer.getvalue()

class Summary:
    '''
    Use wandb.api to revisit trained models or wandb run summaries.
    '''

    def __init__(self, wandb_project: str, run_name: str, num_seeds: int = 6, use_gpu: bool = True) -> None:
        '''
        Parameters
        ---------------
        * wandb_project: wandb project name, used to init wandb run
        * run_name: current run name, used to init wandb run
        * num_seeds: to generate random seeds by range(num_seeds) for random models
        * use_gpu: bool, used to init device
        '''
        self.root = Path(hydra.utils.get_original_cwd())
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        self.num_seeds = num_seeds
        self.wandb_project = wandb_project
        self.run_name = run_name

    def plot_confusion_matrix(self, testloader: DataLoader, nclass: int=10) -> None:
        '''
        Predict on testset and calculate confusion matrix on the result using trained models from wandb artifact
        '''
        from torchmetrics import ConfusionMatrix

        blues = px.colors.sequential.Blues
        confmat = ConfusionMatrix(num_classes=nclass)
        classes = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop',
        'Residential', 'River', 'SeaLake']
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        wandb.init(project=self.wandb_project, name=self.run_name)
        for name in ['Pre-trained', 'Pre-trained + He uniform', 'Pre-trained + custom']:
            cfmatrix = torch.zeros((nclass, nclass))
            for seed in range(self.num_seeds):
                model = load_model_artifact(self.wandb_project, filters={
                                        'config.seed': seed,
                                        'display_name': name,
                })
                model = model.eval() #!
                for x, y in testloader:
                    x = x.to(device)
                    out = model(x)
                    pred = torch.argmax(torch.softmax(out, dim=-1), dim=-1)
                    cfmatrix += confmat(y, pred.cpu())
            cfmatrix /= cfmatrix.sum(1, True)
            cfmatrix *= 100
            cfmatrix = cfmatrix.numpy()

            fig = px.imshow(cfmatrix, 
                    x=classes, y=classes, 
                    aspect="equal", 
                    text_auto=".2f",
                    # hoverinfo='z', 
                    # annotation_text=np.around(cfmatrix, 2),
                    color_continuous_scale=[
                        [0, blues[0]],
                        [1./1000, blues[2]],
                        [1./100, blues[4]],
                        [1./10, blues[6]],
                        [1., blues[8]],
                    ],)
            # fig.update_traces(text=cfmatrix, texttemplate="%{text}")
            fig.update_layout(
                width=1000,
                height=900,
                title={
                    'text': name,
                    'y':0.95,
                    'x':0.5,
                    'xanchor': 'center',
                    'yanchor': 'top'},
                xaxis_title="Predicted label",
                yaxis_title="True label",
                legend_title="Legend Title",
            )
            html = fig2html(fig)
            wandb.log({f'{name}': wandb.Html(html)})
        wandb.finish()

    
    def vis_decision_boundary(self, cfg_model: dict, archs: dict) -> None:
        '''
        1. wandb log visualizations of linear regions for dataset: Circles, Moons, Spirals. 
        On each dataset, generate a plotly subplot.
        * 1st row: plots from models at init state
        * 2nd row: plots from models after training
        2. wandb log plotly boxplot of linear regions for the three datasets on different models
        * xaxis: dataset-stage, e.g., Circles-untrained
        * yaxis: #linear regions

        Parameters
        --------------
        * cfg_model: dict, model config used to initialize the model
        * archs: dict, network configs for the three datasets, key=data name, value=list of net archs
        '''
        wandb.init(project=self.wandb_project, name=self.run_name)
        num_lr = pd.DataFrame(columns=['seed', 'linear region density', 'hidden layers'])
        for name in ['circles', 'moons', 'spiral']:
            filename = name + '.yaml'
            with open(self.root / 'config' / 'dataset' / filename, 'r') as stream:
                try:
                    DATASET = yaml.safe_load(stream)['DATASET']
                except yaml.YAMLError as exc:
                    print(exc)

            html, num_lr = visualize_LR_group(DATASET, cfg_model, archs[name], self.num_seeds, num_lr, self.device,
                                              False)  # results at initialization
            wandb.log({f"LinearRegions/{DATASET['name']}_untrained": wandb.Html(html)})

            html, num_lr = visualize_LR_group(DATASET, cfg_model, archs[name], self.num_seeds, num_lr, self.device,
                                              True)  # results after train
            wandb.log({f"LinearRegions/{DATASET['name']}_trained": wandb.Html(html)})

        fig = px.box(num_lr, x='data-stage', y='linear region density', color='hidden layers')
        fig.update_layout(
            font=dict(
                size=28
            )
        )
        html = fig2html(fig)
        wandb.log({f'linear region density': wandb.Html(html)})
        wandb.finish()

    def metric_line_plot(self, cfg: dict, filters: dict) -> None:
        '''
        plot from summaries of runs, metric 1 versus metic 2
        e.g., plot the number of training samples versus train loss

        Parameters
        ----------
        * cfg: config to init wandb run
        * filters: used to query runs
    
        '''
        for seed in range(self.num_seeds):
            cfg['seed'] = seed
            wandb.init(project=cfg['wandb_project'], name=cfg['run_name'], config=cfg)
            # runs =  api.runs(f'ahui/degree-project-Spiral', filters={'config.name': {"$regex": f'sample-efficiency-{init_method}-*'}, 'config.seed': seed})
            runs = api.runs(
                f"ahui/{cfg['wandb_project']}",
                filters={
                    'config.seed': seed,
                    # 'createdAt':{"$gt": "2022-03-01"} # * filter for runs after specific date
                    **filters
                })

            for run in runs:
                summary = run.summary
                test = run.history(keys=['test.acc', 'test.total_loss'])
                train_size = run.config['DATASET_n_samples']
                wandb.log({
                    'train_size': train_size,
                    'train.acc': summary['train.acc'],
                    'val.acc': summary['val.acc'],
                    'test.acc': test["test.acc"][0],
                    'tain.total_loss': summary['train.total_loss'],
                    'val.total_loss': summary['val.total_loss'],
                    'test.total_loss': test['test.total_loss'][0],
                    '#linear regions': summary['#Linear regions'],
                })
            wandb.finish()

    def metric_boxplot(self, cfg: dict, filters: dict) -> None:
        '''
        Generate boxplot across runs
        xaxis: descrete metric/param, e.g., wdecay
        yaxis: accuracy or #linear regions

        Parameters
        ----------
        * cfg: config to init wandb run
        * filters: used to query runs
        '''
        
        df = pd.DataFrame(columns=['weight std', 'seed', 'accuracy', '#Linear regions', 'net arch'])
        runs = api.runs(f"ahui/{cfg['wandb_project']}", filters=filters)
        for i, run in enumerate(runs):
            summary = run.summary
            test = run.history(keys=['test.acc'])
            row = {
                'seed': run.config['seed'],
                'net arch': run.config['MODEL_h_nodes'],
                'weight std': run.config['params_std'],
                'accuracy': test["test.acc"][0],
                '#Linear regions': summary['#Linear regions']
            }
            df = df.append(row, ignore_index=True)
        acc_fig = px.scatter(df, x='weight std', y='accuracy')
        lr_fig = px.scatter(df, x='weight std', y='#Linear regions')
        acc_fig.show()
        lr_fig.show()

        wandb.init(project=cfg['wandb_project'], name=cfg['run_name'])
        for i, fig in enumerate([acc_fig, lr_fig]):
            buffer = io.StringIO()
            fig.write_html(buffer)
            html = buffer.getvalue()
            wandb.log({f'{i}': wandb.Html(html)})
        wandb.finish()

def load_model_artifact(project: str, filters: dict) -> nn.Module:
    '''
    load torch.nn.Module from wandb artifact with given filters

    Parameters
    ----------
    * project: wandb project name
    * filters: dict, used to filter runs

    Return
    ----------
    * model: nn model
    '''
    runs = api.runs(project, filters=filters)
    model_path = runs[0].logged_artifacts()[0].download() #* use the first run
    model = torch.load(Path(model_path) / 'model.h5')
    return model

def visualize_LR_group(data_cfg: dict,
                       model_cfg: dict,
                       archs: list[list[int]],
                       num_seeds: int,
                       num_lr: pd.DataFrame,
                       device: torch.device,
                       use_artifact_model: bool = True):
    '''
    Generating data for visualizing linear regions of models at initialization or after training.
    Figure 1: plotly subplots of heatmaps of linear regions for different models, return html
    Figure 2: plotly boxplot of #linear regions for different models, return the pd.dataframe

    Parameters
    ------------
    * data_cfg: data config for generating LitDataModule
    * model_cfg: str, model config used to initialize the model
    * archs: list of hidden nodes of models, used to query runs and title subplots
    * num_seeds: range(num_seeds) random models
    * use_artifact_model: bool, default True, models are trained
    '''
    # *********************************** prapare data *********************************
    # new Dataset for circles, moons, spirals
    dataset = Dataset(seed=100, n_train=0.7, n_val=0.1, n_test=0.2, **data_cfg)
    input_dim = dataset.trainset[0][0].shape[0]
    grid_points, grid_labels = dataset.grid_data
    xrange = np.unique(grid_points[:, 0])
    yrange = np.unique(grid_points[:, 1])

    stage = 'trained' if use_artifact_model else 'untrained'
    subplots = []
    for n, h_nodes in enumerate(archs):
        for seed in range(1, num_seeds + 1):  #
            # *********************************** build model *********************************
            if use_artifact_model:
                load_model_artifact('degree-project-decision-boundary', 
                                filters={
                                    'config.seed': seed,
                                    'config.DATASET_name': data_cfg['name'],
                                    'config.MODEL_h_nodes': list(h_nodes),
                                })
                logger.info("[Model] Loading model -- input dim: {}, hidden layers: {}, out dim: {}".format(
                    input_dim, h_nodes, model_cfg.out_dim))
            else:
                model = MODEL[model_cfg.name](input_dim=input_dim, seed=seed, h_nodes=h_nodes, **model_cfg)
                logger.info("[Model] Building model -- input dim: {}, hidden layers: {}, out dim: {}".format(
                    input_dim, h_nodes, model_cfg.out_dim))

            model = model.to(device=device)
            _, grid_sigs, _ = get_signatures(torch.tensor(grid_points).to(device), model, device)

            row = {
                'trained': use_artifact_model,
                'seed': seed,
                'linear region density': len(torch.unique(grid_sigs, dim=0)) / data_cfg['input_volume'],
                'hidden layers': str(h_nodes),
                'data': data_cfg['name'],
                'data-stage': '-'.join([data_cfg['name'], stage])
            }
            num_lr = num_lr.append(row, ignore_index=True)

        # * visaulize linear regions from last seed
        grid_sigs = np.array([''.join(str(x) for x in s.tolist()) for s in grid_sigs])
        #  * categorial to numerical
        region_labels = label_encoder.fit_transform(grid_sigs).reshape(grid_labels.shape)
        region_labels_shuffle = np.zeros_like(region_labels)
        # # shuffle labels to visualy differentiate adjacent regions
        random_labels = np.random.permutation(region_labels.max() + 1)
        for i, label in enumerate(random_labels):
            region_labels_shuffle[np.where(region_labels == i)] = label
        subplot = visualize_signatures(region_labels_shuffle,
                                       grid_labels,
                                       xrange,
                                       yrange,
                                       showscale=n == 0,
                                       colorbar_len=1.04,
                                       colorbary=0.5)
        subplot_points_stacked = visualize_signatures(region_labels_shuffle,
                                                      grid_labels,
                                                      xrange,
                                                      yrange,
                                                      showscale=False,
                                                      colorbar_len=1,
                                                      colorbary=0.5,
                                                      data=dataset.trainset.tensors)
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
            fig.add_trace(layer, 1, c + 1)
        for layer in col[1]:
            fig.add_trace(layer, 2, c + 1)

    fig.update_layout(
        # yaxis=dict(scaleanchor='x', scaleratio=1),  # set aspect ratio to 1, actually done by width and column_widths
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        width=1800 if data_cfg['name'] == 'moons' else 1600,
        height=700,
        showlegend=False,
        yaxis_title=stage,
        font=dict(
            size=28
        )
    )
    for row in range(2):
        for col in range(len(subplots)):
            fig.update_yaxes(range=[yrange[0], yrange[-1]], row=row + 1, col=col + 1)
            fig.update_xaxes(range=[xrange[0], xrange[-1]], row=row + 1, col=col + 1)
    # fig.show()
    buffer = io.StringIO()
    fig.write_html(buffer)
    html = buffer.getvalue()
    return html, num_lr