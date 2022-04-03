from matplotlib.pyplot import autoscale
from pytorch_lightning.callbacks import Callback
import torch
import logging
import numpy as np
from sklearn.preprocessing import LabelEncoder
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from utils import get_signatures, visualize_signatures

logger = logging.getLogger(__name__)

class LinearRegionLogger(Callback):
    LOCAL_MODEL_DIR = '/outputs/checkpoints'

    def __init__(self, log_every: int, grid_data:list[np.ndarray], log_stage:str='end', data: list[np.ndarray]=None, pred_th:float=0.5) -> None:
        """ 
        Log the number of linear regions & visualize linear regions

        Parameters
        ----------
        log_every: int or list of int
        grid_data: [grid_points, grid_labels]
        log_stage: log after train epoch if 'end', else log before train epoch
        data: [train_points, labels]
        pred_th: threshold to predict on sigmoid output
        """
        super().__init__()
        self.log_every = log_every
        assert isinstance(log_every, int) or isinstance(log_every, list) or isinstance(self.log_every, np.ndarray), 'invalid type of log_every, must be list or int'
        self.label_encoder = LabelEncoder()
        self.grid_points, self.grid_labels = grid_data
        self.xrange = np.unique(self.grid_points[:, 0])
        self.yrange = np.unique(self.grid_points[:, 1])
        self.data = data
        self.PRED_TH = pred_th
        
        if log_stage == 'start':
            self.on_train_epoch_start = self.plot_on_train_epoch_start
        else:
            self.on_train_epoch_end = self.plot_on_train_epoch_end
        


    def plot_on_train_epoch_start(self, trainer, plmodel) -> None:
        epoch = trainer.current_epoch
        plot = epoch in range(10) or (epoch + 1) % self.log_every == 0
        if plot:
            net_out, grid_sigs, _ = get_signatures(torch.tensor(self.grid_points).float().to(plmodel.device), plmodel.model, plmodel.device)
            grid_sigs = np.array([''.join(str(x) for x in s.tolist()) for s in grid_sigs])
            #  * categorial to numerical
            region_labels = self.label_encoder.fit_transform(grid_sigs).reshape(self.grid_labels.shape)
            # shuffle labels to visualy differentiate adjacent regions
            random_labels = np.random.permutation(region_labels.max() +1)
            for i, label in enumerate(random_labels):
                region_labels[np.where(region_labels==i)] = label

            net_out = torch.sigmoid(net_out)
            pseudo_label = torch.where(net_out.cpu() > self.PRED_TH, 1.0, 0.0).numpy()
            pseudo_label = pseudo_label.reshape(self.grid_labels.shape)
            subplot1 = visualize_signatures(region_labels, self.grid_labels, self.xrange, self.yrange, showscale=True)
            subplot2 = visualize_signatures(region_labels, pseudo_label, self.xrange, self.yrange)
            subplot3 = visualize_signatures(region_labels, self.grid_labels, self.xrange, self.yrange, self.data)
            
            confidence = net_out.reshape(self.grid_labels.shape).detach().cpu().numpy()
            layer4 = go.Heatmap(
                x=self.xrange, y=self.yrange, z=confidence, 
                zmin=0, zmax=1,
                colorscale='Viridis', 
                colorbar={'title': 'confidence', 'titleside': 'right', 'len': 0.51, 'y':0.235, 'nticks': 5,},
                # showscale=False
            )
            
            fig = make_subplots(
                cols=2, 
                rows=2,
                shared_xaxes=True, 
                shared_yaxes=True, 
                vertical_spacing = 0.05,
                horizontal_spacing = 0.05,
                subplot_titles=('true label', '', 'true label', ''),#confidence map
                column_widths=[0.51, 0.49],
                row_heights=[0.5, 0.5]
                )
            for layer in subplot1:
                fig.add_trace(layer, 1,1)
            
            for layer in subplot2:
                fig.add_trace(layer, 1,2)

            for layer in subplot3:
                fig.add_trace(layer, 2,1)

            fig.add_trace(layer4, 2, 2)   
            fig.update_layout(
                # yaxis=dict(scaleanchor='x', scaleratio=1),  # set aspect ratio to 1, actually done by width and column_widths
                paper_bgcolor='rgba(0,0,0,0)',
                # plot_bgcolor='rgba(0,0,0,0)',
                width=1180 if trainer.datamodule.name=='moons' else 840,
                height=800,
                showlegend=False,
            )
            # update_yaxes and update_xaxes are needed for removing extra range caused by scatterplots, 
            # thus leaving gap between the heatmap and xy axis, resulting different sizes among subplots
            fig.update_yaxes(range=[self.yrange[0], self.yrange[-1]], nticks=5, row=1, col=1)
            fig.update_yaxes(range=[self.yrange[0], self.yrange[-1]], row=2, col=1)
            fig.update_xaxes(range=[self.xrange[0], self.xrange[-1]], row=2, col=1)
            fig.update_xaxes(range=[self.xrange[0], self.xrange[-1]], row=2, col=2)


            fig.show()  
            num_lr = len(np.unique(grid_sigs))
            self.log(f'#linear regions', num_lr)
            del grid_points
            del sigs_grid


    def plot_on_train_epoch_end(self, trainer, plmodel) -> None:
        epoch = trainer.current_epoch
        plot = epoch in range(10) or (epoch + 1) % self.log_every == 0
        if plot:     
            _, grid_sigs, _ = get_signatures(torch.tensor(self.grid_points).float().to(plmodel.device), plmodel.model, plmodel.device)
            grid_sigs = np.array([''.join(str(x) for x in s.tolist()) for s in grid_sigs])
            num_lr = len(np.unique(grid_sigs)) 
            self.log(f'#linear regions', num_lr)
            del grid_points
            del grid_sigs


        

    