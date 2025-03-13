from torch import nn
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_std=1.0):
        super().__init__()
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.zeros(out_features, in_features))
        self.bias_mu   = nn.Parameter(torch.zeros(out_features))
        self.bias_rho  = nn.Parameter(torch.zeros(out_features))
        self.prior_mean = 0.0
        self.prior_std  = prior_std

    def forward(self, x):
        weight_sigma = torch.log1p(torch.exp(self.weight_rho))
        bias_sigma   = torch.log1p(torch.exp(self.bias_rho))
        eps_w = torch.randn_like(self.weight_mu)
        eps_b = torch.randn_like(self.bias_mu)
        weight = self.weight_mu + weight_sigma * eps_w
        bias   = self.bias_mu   + bias_sigma * eps_b
        out = torch.matmul(x, weight.t()) + bias
        prior_var = self.prior_std ** 2
        weight_var = weight_sigma ** 2
        kl_weight = 0.5 * torch.sum(
            (weight_var + (self.weight_mu - self.prior_mean)**2) / prior_var 
            - 1.0 + 2.0 * torch.log(self.prior_std / weight_sigma)
        )
        bias_var = bias_sigma ** 2
        kl_bias = 0.5 * torch.sum(
            (bias_var + (self.bias_mu - self.prior_mean)**2) / prior_var 
            - 1.0 + 2.0 * torch.log(self.prior_std / bias_sigma)
        )
        kl = kl_weight + kl_bias
        return out, kl

class BayesianNN(nn.Module):
    def __init__(self, layer_sizes, prior_std=1.0):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            in_dim, out_dim = layer_sizes[i], layer_sizes[i+1]
            layers.append(BayesianLinear(in_dim, out_dim, prior_std))
            if i < len(layer_sizes) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        total_kl = 0.0
        out = x
        for layer in self.net:
            if isinstance(layer, BayesianLinear):
                out, kl = layer(out)
                total_kl += kl
            else:
                out = layer(out)
        return out, total_kl

# Modified Lightning module that auto-configures based on task type
class HMNNLightning(pl.LightningModule):
    def __init__(self, input_dim, output_dim, prior_std=1.0, lr=1e-3, task='regression'):
        """
        Parameters:
          - input_dim: number of input features.
          - output_dim: number of output features (for regression, typically 1; for classification, number of classes).
          - task: 'regression' or 'classification'.
        """
        super().__init__()
        self.task = task
        self.lr = lr
        self.bnn = BayesianNN([input_dim, 20, output_dim], prior_std)

    def forward(self, x):
        return self.bnn(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, total_kl = self(x)
        if self.task == 'classification':
            # Ensure target is of integer type and 1D (class indices)
            y = y.long().squeeze()  # adjust shape if necessary
            nll_loss = F.cross_entropy(y_hat, y)
        else:
            # For regression, ensure y_hat and y have the same shape.
            # Here we assume y_hat is [batch, output_dim] and y is [batch] for scalar regression.
            # If output_dim==1, we can squeeze y_hat.
            nll_loss = F.mse_loss(y_hat.squeeze(-1), y)
        loss = nll_loss + total_kl / x.size(0)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, total_kl = self(x)
        if self.task == 'classification':
            y = y.long().squeeze()
            val_loss = F.cross_entropy(y_hat, y)
        else:
            val_loss = F.mse_loss(y_hat.squeeze(-1), y)
        val_loss += total_kl / x.size(0)
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
