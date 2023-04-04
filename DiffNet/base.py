import torch
from torch.utils.data import DataLoader
from pytorch_lightning.core import LightningModule


class PDE(LightningModule):
    """
    PDE Base Class
    """

    def __init__(self, network, **kwargs):
        super().__init__()
        # self.save_hyperparameters(kwargs)
        self.kwargs = kwargs
        self.network = network
        self.nsd = kwargs.get('nsd', 2)
        self.batch_size =  kwargs.get('batch_size', 64)
        self.n_workers = kwargs.get('n_workers', 1)
        self.learning_rate = kwargs.get('learning_rate', 3e-4)

        self.domain_length = kwargs.get('domain_length', 1.) # for backward compatibility
        self.domain_size = kwargs.get('domain_size', 64) # for backward compatibility
        self.domain_lengths_nd = kwargs.get('domain_lengths', (self.domain_length, self.domain_length, self.domain_length))
        self.domain_sizes_nd = kwargs.get('domain_sizes', (self.domain_size, self.domain_size, self.domain_size))
        if self.nsd >= 2:
            self.domain_lengthX = self.domain_lengths_nd[0]
            self.domain_lengthY = self.domain_lengths_nd[1]
            self.domain_sizeX = self.domain_sizes_nd[0]
            self.domain_sizeY = self.domain_sizes_nd[1]
            if self.nsd >= 3:
                self.domain_lengthZ = self.domain_lengths_nd[2]
                self.domain_sizeZ = self.domain_sizes_nd[2]

    def loss(self, u, inputs_tensor, forcing_tensor):
        raise NotImplementedError

    def forward(self, batch):
        inputs_tensor, forcing_tensor = batch
        u = self.network(inputs_tensor)
        return u, inputs_tensor, forcing_tensor

    def training_step(self, batch, batch_idx):
        u, inputs_tensor, forcing_tensor = self.forward(batch)
        loss_val = self.loss(u, inputs_tensor, forcing_tensor).mean()
        self.log('PDE_loss', loss_val.item())
        self.log('loss', loss_val.item())
        return loss_val

    def configure_optimizers(self):
        """
        Configure optimizer for network parameters
        """
        lr = self.learning_rate
        opts = [torch.optim.Adam(self.network.parameters(), lr=lr)]
        return opts, []