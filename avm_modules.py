import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import torch.nn.functional as F


import pytorch_lightning as pl 
from torch_geometric.data import DataLoader 

from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from avm_etl_module import get_geometric_data

from torchmetrics import MeanSquaredError
from torchmetrics import MeanAbsoluteError

test_mse = MeanSquaredError(compute_on_step=False)
test_mae = MeanAbsoluteError(compute_on_step=False)

valid_mse = MeanSquaredError(compute_on_step=False)
valid_mae = MeanAbsoluteError(compute_on_step=False)

class AVMProjModule(pl.LightningModule):
    def __init__(self, hidden_dim = 5, batch_size = 1, lr = 0.001):
        super(AVMProjModule, self).__init__()
        
        # Step 1: 載入模型參數
        self.hidden_dim = hidden_dim
        
        # Step 2: 載入data-dependent參數
        self.num_node_features = 2 
        self.output_dim = 1 
        
        # Step 3: 載入訓練參數
        self.batch_size = batch_size
        self.lr = lr 
        
        # Step 4: 建立模型 
        self.model = GNN(self.num_node_features, hidden_dim = self.hidden_dim, output_dim = self.output_dim)
        
        self._batch_cnt = 0
        self._architecture_logged = False
        self._best_val_loss = 1e13
        self._best_val_mae = 1e13
        
    def forward(self, x, edge_index):
        y = self.model(x, edge_index)
        return y 
    
    def on_train_start(self):
        self._batch_cnt = 0
        self._architecture_logged = False
        self._best_val_loss = 1e13
        self._best_val_mae = 1e13
        self.log('best_val_loss', self._best_val_loss)
        self.log('best_val_mae', self._best_val_mae)
    
    def on_fit_start(self): 
        self._batch_cnt = 0
        self.logger.log_hyperparams(
            params = {
                "hidden_dim": self.hidden_dim,
                "num_node_features": self.num_node_features, 
                "output_dim": self.output_dim,
                "batch_size": self.batch_size, 
                "lr": self.lr
            }, 
            metrics = {
                "best_val_loss": self._best_val_loss
            }
        )
    
    def training_step(self, batch, batch_idx):
        
        # Step 1: calculate training loss 
        x, y, edge_index = batch.x, batch.y, batch.edge_index 
        y_hat = self(x, edge_index)
        loss = F.l1_loss(y_hat, y)
        
        # Step 2: log values 
        self.logger.experiment.add_scalar(f'Train/loss', loss, self._batch_cnt)
        
        # Step 3: log neural architecture 
        if not self._architecture_logged: 
            self.logger.experiment.add_graph(self, [x, edge_index])
            print("Model Architecture Logged")
            self._architecture_logged = True
        
        # Step 4: increase batch count 
        self._batch_cnt += 1 
        return loss
    
    def validation_step(self, batch, batch_idx):
        # Step 1: calculate validation loss 
        x, y, edge_index = batch.x, batch.y, batch.edge_index 
        y_hat = self(x, edge_index)
        loss = F.l1_loss(y_hat, y)
        
        valid_mse(y_hat, y)
        valid_mae(y_hat, y)
        
        # Step 2: log value
        self.log('val_loss', loss)
        return {'val_loss': loss}
    
    def validation_epoch_end(self, outputs):
        if self.current_epoch > 0:
            # Step 1: calculate average loss and metric values 
            avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
            
            mse = valid_mse.compute()
            mae = valid_mae.compute()
            
            # Step 2: log total loss  
            self.logger.experiment.add_scalar("Val/total_loss", avg_loss, self.current_epoch)
            self.logger.experiment.add_scalar("Val/mse", mse, self.current_epoch)
            self.logger.experiment.add_scalar("Val/mae", mae, self.current_epoch)
            
            # Step 3: replace and log best values
            if avg_loss < self._best_val_loss:
                self._best_val_loss = avg_loss 
                # loss the best_val_loss 
                self.log('best_val_loss', self._best_val_loss)
            if mae < self._best_val_mae:
                self._best_val_mae = mae 
                # loss the best_val_loss 
                self.log('best_val_mae', self._best_val_mae)
                
    def test_step(self, batch, batch_idx):
        # Step 1: calculate output 
        x, y, edge_index = batch.x, batch.y, batch.edge_index 
        y_hat = self(x, edge_index)
        
        # Step 2: apply metrics 
        test_mse(y_hat, y)
        test_mae(y_hat, y)
        return {'result': 'test'}
    def on_test_epoch_end(self):
        mse = test_mse.compute().item()
        mae = test_mae.compute().item()
        result = {
            'mse': mse, 
            'mae': mae
            }
        print(result)
        return result
    
    def prepare_data(self):
        self.geometric_data = get_geometric_data()
    
    def train_dataloader(self):
        return DataLoader(
            [self.geometric_data], 
            shuffle=True, 
            batch_size=self.batch_size, 
            num_workers=4)
    
    def val_dataloader(self):
        return DataLoader(
            [self.geometric_data], 
            batch_size=self.batch_size, 
            num_workers=4,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            [self.geometric_data], 
            batch_size=self.batch_size, 
            num_workers=4,
            pin_memory=True
        )
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=5e-4)
        lr_scheduler = LinearWarmupCosineAnnealingLR(optimizer, warmup_epochs=5, max_epochs=40)
        return {
            "optimizer": optimizer, 
            "lr_scheduler": lr_scheduler
        }

class GNN(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim=2, output_dim=1):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)
    def forward(self, x, edge_index):
        # x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        return x
    
class GCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv, self).__init__(aggr='add')
        self.lin = torch.nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # Step 1: Add self-loops
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Multiply with weights
        x = self.lin(x)

        # Step 3: Calculate the normalization
        row, col = edge_index
        deg = degree(row, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4: Propagate the embeddings to the next layer
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x,
                              norm=norm)

    def message(self, x_j, norm):
        # Normalize node features.
        return norm.view(-1, 1) * x_j