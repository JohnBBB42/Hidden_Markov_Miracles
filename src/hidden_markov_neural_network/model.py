import torch
from torch import nn
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



##########################################################################
# New model
##########################################################################

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class BayesLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, 
                 prior_initial_sigma: float = 1.0, prior_large_sigma: float = 1.0, drop_rate: float = 0.1, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.drop_rate = drop_rate  # probability of *dropping* (jump/reset)
        # Variational posterior parameters for weights
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), -3.0))
        # Initialize mu near 0, rho to a negative value so initial sigma (via softplus) is small
        nn.init.xavier_normal_(self.weight_mu)  # small random init for means
        # Variational posterior for bias
        self.has_bias = bias
        if bias:
            self.bias_mu = nn.Parameter(torch.zeros(out_features))
            self.bias_rho = nn.Parameter(torch.full((out_features,), -3.0))
            nn.init.uniform_(self.bias_mu, -0.1, 0.1)
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)
        # Prior parameters (to be updated as weights evolve)
        # prior_mu_old: previous weight means (initially 0), prior_sigma_small: small variance around old weights
        self.register_buffer('prior_mu_old', torch.zeros(out_features, in_features))
        self.register_buffer('prior_sigma_small', torch.full((out_features, in_features), prior_initial_sigma))
        # We use scalar prior_large_sigma for the "jump" component (assumed same for all weights for simplicity)
        self.prior_sigma_large = prior_large_sigma
        # For numerical stability in log-prob calculations, define a small variance for the spike component of q
        self.posterior_sigma2 = 1e-6  # effectively a point-mass at 0 for DropConnect posterior

    def forward(self, input: torch.Tensor):
        # Sample weights from variational posterior
        if self.training:
            # Reparameterize to get weight sample: w = mu + sigma * eps
            weight_sigma = torch.log1p(torch.exp(self.weight_rho))  # softplus to ensure positivity
            eps_w = torch.randn_like(weight_sigma)
            w = self.weight_mu + weight_sigma * eps_w
            # DropConnect mask: Bernoulli(1 - drop_rate) for each weight element
            mask = (torch.rand_like(w) < (1 - self.drop_rate)).float()
            w_sample = w * mask  # apply mask (dropped weights become 0)
            # Sample bias similarly
            if self.has_bias:
                bias_sigma = torch.log1p(torch.exp(self.bias_rho))
                eps_b = torch.randn_like(bias_sigma)
                b = self.bias_mu + bias_sigma * eps_b
                # For bias, we could apply DropConnect as well (here using same drop rate per bias element)
                mask_b = (torch.rand_like(b) < (1 - self.drop_rate)).float()
                b_sample = b * mask_b
            else:
                b_sample = None
            # Compute log posterior (q) and log prior (p) for this weight sample
            # Variational posterior q: mixture of N(w; mu, sigma^2) and a point mass at 0 (approximated by small variance Gaussian)
            # Prior p: mixture of N(w; old_weight_mean, small_var^2) and N(w; 0, large_var^2)
            # Calculate element-wise log-probabilities:
            # Posterior components
            weight_sigma_val = torch.log1p(torch.exp(self.weight_rho))  # actual sigma values
            # Posterior comp1: N(w_sample; weight_mu, weight_sigma^2), comp2: N(w_sample; 0, (posterior_sigma2)^2)
            log_q1 = -0.5 * ((w_sample - self.weight_mu) / weight_sigma_val)**2 - torch.log(weight_sigma_val * math.sqrt(2 * math.pi))
            log_q2 = -0.5 * (w_sample / self.posterior_sigma2)**2 - math.log(self.posterior_sigma2 * math.sqrt(2 * math.pi))
            # Mix weights: (1-p) for comp1, p for comp2
            log_q = torch.logaddexp(log_q1 + math.log(1 - self.drop_rate), log_q2 + math.log(self.drop_rate))
            # Sum over all weight elements
            log_q_weight = log_q.sum()
            log_q_bias = 0.0
            if self.has_bias:
                bias_sigma_val = torch.log1p(torch.exp(self.bias_rho))
                log_q1_b = -0.5 * ((b_sample - self.bias_mu) / bias_sigma_val)**2 - torch.log(bias_sigma_val * math.sqrt(2 * math.pi))
                log_q2_b = -0.5 * (b_sample / self.posterior_sigma2)**2 - math.log(self.posterior_sigma2 * math.sqrt(2 * math.pi))
                log_q_b = torch.logaddexp(log_q1_b + math.log(1 - self.drop_rate), log_q2_b + math.log(self.drop_rate))
                log_q_bias = log_q_b.sum()
            total_log_q = log_q_weight + log_q_bias
            # Prior components
            prior_sigma_small = self.prior_sigma_small  # tensor of small variances per weight (from previous posterior)
            log_p1 = -0.5 * ((w_sample - self.prior_mu_old) / prior_sigma_small)**2 - torch.log(prior_sigma_small * math.sqrt(2 * math.pi))
            # prior_sigma_large is scalar; broadcast to w_sample shape
            sigma_large = torch.tensor(self.prior_sigma_large, device=w_sample.device)
            log_p2 = -0.5 * (w_sample / sigma_large)**2 - math.log(sigma_large * math.sqrt(2 * math.pi))
            log_p = torch.logaddexp(log_p1 + math.log(1 - self.drop_rate), log_p2 + math.log(self.drop_rate))
            log_p_weight = log_p.sum()
            log_p_bias = 0.0
            if self.has_bias:
                # For bias prior, assume prior_mu_old = 0 initially and small variance = prior_initial_sigma, can update similarly
                prior_sigma_small_b = self.prior_sigma_small.mean(dim=1) if self.prior_sigma_small.ndim > 1 else self.prior_sigma_small  # average or same for bias
                prior_sigma_small_b = prior_sigma_small_b.squeeze()
                log_p1_b = -0.5 * ((b_sample - 0) / prior_sigma_small_b)**2 - torch.log(prior_sigma_small_b * math.sqrt(2 * math.pi))
                log_p2_b = -0.5 * (b_sample / sigma_large)**2 - math.log(sigma_large * math.sqrt(2 * math.pi))
                log_p_b = torch.logaddexp(log_p1_b + math.log(1 - self.drop_rate), log_p2_b + math.log(self.drop_rate))
                log_p_bias = log_p_b.sum()
            total_log_p = log_p_weight + log_p_bias
        else:
            # In evaluation mode, use deterministic weights (posterior mean) for predictions
            w_sample = self.weight_mu
            b_sample = self.bias_mu if self.has_bias else None
            total_log_q = torch.tensor(0.0, device=w_sample.device)
            total_log_p = torch.tensor(0.0, device=w_sample.device)
        # Compute linear layer output
        output = input.matmul(w_sample.t())
        if self.has_bias:
            output = output + b_sample
        return output, total_log_q, total_log_p

    def update_prior(self):
        """Update the prior distribution parameters to the current posterior (to be called after training on a time segment)."""
        # Set prior mean to current posterior mean (weights), and small variance to current posterior std
        weight_sigma = torch.log1p(torch.exp(self.weight_rho)).detach()
        self.prior_mu_old.copy_(self.weight_mu.detach())            # previous mean becomes current mean
        self.prior_sigma_small.copy_(weight_sigma.detach())         # small variance = posterior std dev
        if self.has_bias:
            # For bias, we can update similarly (set previous bias mean and small var).
            # Ensure shapes match: prior_mu_old for bias could be stored separately or reuse part of tensor.
            # Simplicity: we won't maintain separate bias prior, assume bias prior always N(0, prior_initial_sigma).
            pass

class HMNNModel(pl.LightningModule):
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 1,
                 prior_initial_sigma: float = 1.0, prior_large_sigma: float = 1.0, drop_rate: float = 0.1,
                 lr: float = 1e-3, n_train: int = 1):
        """
        ... (existing parameters)
        target_mean: mean of training targets (for logging normalization)
        target_std: std of training targets (for logging normalization)
        """
        super().__init__()
        self.save_hyperparameters()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.drop_rate = drop_rate
        self.lr = lr
        self.n_train = n_train  # for KL scaling
        self.bayes_fc1 = BayesLinear(input_dim, hidden_dim,
                                     prior_initial_sigma=prior_initial_sigma, 
                                     prior_large_sigma=prior_large_sigma, drop_rate=drop_rate)
        self.bayes_fc2 = BayesLinear(hidden_dim, output_dim,
                                     prior_initial_sigma=prior_initial_sigma, 
                                     prior_large_sigma=prior_large_sigma, drop_rate=drop_rate)

    def forward(self, x):
        out1, log_q1, log_p1 = self.bayes_fc1(x)
        out1 = F.relu(out1)
        out2, log_q2, log_p2 = self.bayes_fc2(out1)
        total_log_q = log_q1 + log_q2
        total_log_p = log_p1 + log_p2
        return out2, total_log_q, total_log_p

    def training_step(self, batch, batch_idx):
        inputs, targets, _ = batch
        preds, log_q, log_p = self(inputs)
        nll = F.mse_loss(preds.squeeze(), targets.squeeze(), reduction='mean')
        kl_div = log_q - log_p
        kl_weight = 1.0 / self.n_train if self.n_train else 1.0
        loss = nll + kl_weight * kl_div
        # Compute loss for logging:
        self.log("train_nll", nll, on_epoch=True, prog_bar=True)
        self.log("train_kl", kl_div, on_epoch=True, prog_bar=True)
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets, _ = batch
        preds, _, _ = self(inputs)
        val_loss = F.mse_loss(preds.squeeze(), targets.squeeze(), reduction='mean')
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def update_prior_weights(self):
        self.bayes_fc1.update_prior()
        self.bayes_fc2.update_prior()
