import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
import torch.nn.functional as F

import pytorch_lightning as pl 
from torch_geometric.data import DataLoader 
from get_geometric_data import get_geometric_data

class AVMProjModule(pl.LightningModule):
    def __init__(self, hidden_dim = 5, output_dim=1, lr = 0.001, batch_size = 1):
        super(AVMProjModule, self).__init__()
        self.geometric_data = get_geometric_data()
        num_node_features = self.geometric_data.num_node_features
        self.model = GraphNeuralNet(num_node_features, hidden_dim = hidden_dim, output_dim = output_dim)
        self.lr = lr 
        self.batch_size = batch_size
    def forward(self, x, edge_index):
        y = self.model(x, edge_index)
        return y 
    def training_step(self, batch, batch_idx):
        x, y, edge_index = batch.x, batch.y, batch.edge_index 
        y_hat = self(x, edge_index)
        loss = F.l1_loss(y_hat, y)
        return {'loss': loss}
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        # print(self.current_epoch, " - loss:", avg_loss.item() / self.batch_size)
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=5e-4) 
    def train_dataloader(self):
        return DataLoader([self.geometric_data], batch_size=self.batch_size)


class GraphNeuralNet(torch.nn.Module):
    def __init__(self, num_node_features, hidden_dim=2, output_dim=1):
        super(GraphNeuralNet, self).__init__()
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