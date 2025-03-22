import csv
import os
from datetime import datetime
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import pytorch_lightning as pl
import math

# try without dropconnect

class BayesLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int,
                 prior_initial_sigma: float = 1.0, prior_large_sigma: float = 1.0,
                 drop_rate: float = 0.2, bias: bool = True, alpha_k: float = 0.5, sigma_k: float = np.exp(-2),
                 c: float = np.exp(5), pi: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha_k = alpha_k
        self.sigma_k = sigma_k
        self.c = c
        self.pi = pi
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
            # Register bias prior buffers
            self.register_buffer('prior_mu_old_bias', torch.zeros(out_features))
            self.register_buffer('prior_sigma_small_bias', torch.full((out_features,), prior_initial_sigma))
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

    def hmm_update_weights(self, alpha_k, sigma_k, c, pi):
        weight_sigma = torch.log1p(torch.exp(self.weight_rho)).detach()

        # Correct Bayesian HMNN update logic
        new_prior_mu = (1 - alpha_k) * self.weight_mu.detach() + alpha_k * self.prior_mu_old
        new_prior_sigma = torch.sqrt(sigma_k ** 2 + (alpha_k ** 2) * (self.prior_sigma_small ** 2))

        self.prior_mu_old.copy_(new_prior_mu)
        self.prior_sigma_small.copy_(new_prior_sigma)

        if self.has_bias:
            bias_sigma = torch.log1p(torch.exp(self.bias_rho)).detach()
            new_prior_mu_bias = (1 - alpha_k) * self.bias_mu.detach() + alpha_k * self.prior_mu_old_bias
            new_prior_sigma_bias = torch.sqrt(sigma_k ** 2 + (alpha_k ** 2) * (self.prior_sigma_small_bias ** 2))

            self.prior_mu_old_bias.copy_(new_prior_mu_bias)
            self.prior_sigma_small_bias.copy_(new_prior_sigma_bias)


class HMNNModel(pl.LightningModule):
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 1,
                 num_hidden_layers: int = 5,  # new parameter for number of hidden layers
                 prior_initial_sigma: float = 1.0, prior_large_sigma: float = 1.0,
                 drop_rate: float = 0.0,  # disable dropconnect by default
                 lr: float = 1e-3, alpha_k: float = 0.5,
                 sigma_k: float = math.exp(-2), c: float = math.exp(5),
                 pi: float = 0.5, start_year: int = 1980, n_train: int = 1, 
                 pos_weight=1.0):
        super().__init__()
        self.save_hyperparameters()
        self.pos_weight = pos_weight
        self.n_train = n_train
        # Create a list of hidden layers
        self.hidden_layers = nn.ModuleList()
        # First layer: input to first hidden layer
        self.hidden_layers.append(
            BayesLinear(input_dim, hidden_dim,
                        prior_initial_sigma=prior_initial_sigma,
                        prior_large_sigma=prior_large_sigma, drop_rate=drop_rate,
                        alpha_k=alpha_k, sigma_k=sigma_k, c=c, pi=pi)
        )
        # Additional hidden layers (if any)
        for _ in range(num_hidden_layers - 1):
            self.hidden_layers.append(
                BayesLinear(hidden_dim, hidden_dim,
                            prior_initial_sigma=prior_initial_sigma,
                            prior_large_sigma=prior_large_sigma, drop_rate=drop_rate,
                            alpha_k=alpha_k, sigma_k=sigma_k, c=c, pi=pi)
            )
        # Output layer
        self.out_layer = BayesLinear(hidden_dim, output_dim,
                                     prior_initial_sigma=prior_initial_sigma,
                                     prior_large_sigma=prior_large_sigma, drop_rate=drop_rate,
                                     alpha_k=alpha_k, sigma_k=sigma_k, c=c, pi=pi)
        self.lr = lr
        self.start_year = start_year

    def forward(self, x):
        total_log_q, total_log_p = 0.0, 0.0
        # Pass input through hidden layers with ReLU activation
        for layer in self.hidden_layers:
            x, log_q, log_p = layer(x)
            total_log_q += log_q
            total_log_p += log_p
            x = F.relu(x)
        # Output layer (no activation, as BCEWithLogitsLoss expects raw logits)
        x, log_q, log_p = self.out_layer(x)
        total_log_q += log_q
        total_log_p += log_p
        return x, total_log_q, total_log_p
    
    def write_metrics_to_csv(self, metrics: dict, phase: str, window_info: str = ""):
        """Writes training/validation metrics to a CSV file."""
        date_str = datetime.now().strftime("%Y%m%d")
        filename = f"hmnn_metrics_{phase}_{self.start_year}_{window_info}_{date_str}.csv"
        os.makedirs("metrics", exist_ok=True)
        filepath = os.path.join("metrics", filename)
        file_exists = os.path.isfile(filepath)
        with open(filepath, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=metrics.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics)

    def training_step(self, batch, batch_idx):
        inputs, targets, _ = batch
        preds, log_q, log_p = self(inputs)
        current_epoch = self.current_epoch
        max_epochs = self.trainer.max_epochs
        anneal_factor = min(1.0, current_epoch / (max_epochs * 0.5))
        pos_weight_tensor = torch.tensor([self.pos_weight], device=preds.device)
        nll = F.binary_cross_entropy_with_logits(preds.squeeze(), targets.squeeze(), pos_weight=pos_weight_tensor, reduction='mean')
        kl_div = log_q - log_p
        kl_weight = anneal_factor / self.n_train if self.n_train else anneal_factor
        loss = nll + kl_weight * kl_div
        # Compute classification accuracy
        probs = torch.sigmoid(preds.squeeze())
        pred_labels = (probs >= 0.5).float()
        acc = (pred_labels == targets.squeeze()).float().mean()
        # Compute loss for logging:
        self.log("train_nll", nll, on_epoch=True, prog_bar=True)
        self.log("train_kl", kl_div, on_epoch=True, prog_bar=True)
        self.log("train_loss", loss, on_epoch=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True)

        # Save metrics to CSV
        metrics = {
            "epoch": self.current_epoch,
            "train_nll": nll.item(),
            "train_kl": kl_div.item(),
            "train_loss": loss.item(),
            "train_acc": acc.item(),
            "lr": self.hparams.lr
        }

        window_info = getattr(self, 'window_info', "")
        self.write_metrics_to_csv(metrics, phase="train", window_info=window_info)

        return loss

    def validation_step(self, batch, batch_idx):
        inputs, targets, _ = batch
        preds, _, _ = self(inputs)
        val_loss = F.binary_cross_entropy_with_logits(preds.squeeze(), targets.squeeze(), reduction='mean')
        probs = torch.sigmoid(preds.squeeze())
        pred_labels = (probs >= 0.5).float()
        acc = (pred_labels == targets.squeeze()).float().mean()
        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

        # Save validation metrics to CSV
        metrics = {
            "epoch": self.current_epoch,
            "val_loss": val_loss.item(),
            "val_acc": acc.item(),
            "lr": self.hparams.lr
        }

        window_info = getattr(self, 'window_info', "")
        self.write_metrics_to_csv(metrics, phase="val", window_info=window_info)

        return val_loss

    def test_step(self, batch, batch_idx):
        inputs, targets, _ = batch
        preds, _, _ = self(inputs)
        test_loss = F.binary_cross_entropy_with_logits(preds.squeeze(), targets.squeeze(), reduction='mean')
        probs = torch.sigmoid(preds.squeeze())
        pred_labels = (probs >= 0.5).float()
        acc = (pred_labels == targets.squeeze()).float().mean()

        # Log metrics
        self.log("test_loss", test_loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

        return {"test_loss": test_loss, "test_acc": acc}


    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def hmm_update_model_weights(self):
        alpha_k = self.hparams.alpha_k
        sigma_k = self.hparams.sigma_k
        c = self.hparams.c
        pi = self.hparams.pi

        # Update each hidden layer
        for layer in self.hidden_layers:
            layer.hmm_update_weights(alpha_k, sigma_k, c, pi)
        # Update the output layer
        self.out_layer.hmm_update_weights(alpha_k, sigma_k, c, pi)


class NeuralNetwork(pl.LightningModule):
    def __init__(self, input_size: int, lr: float = 0.001, start_year: int = 2024):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 1)  # Output layer
        
        self.relu = nn.ReLU()
        self.lr = lr
        self.start_year = start_year
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        x = self.relu(self.fc5(x))
        x = self.fc6(x)  # No sigmoid here, as loss function expects logits
        return x.squeeze(-1)

    def compute_accuracy(self, y_pred, y):
        # Convert logits to probabilities
        probs = torch.sigmoid(y_pred)
        # Threshold to get binary predictions (0 or 1)
        predictions = (probs > 0.5).float()
        # Calculate accuracy
        accuracy = (predictions == y).float().mean()
        return accuracy

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y = y.squeeze(-1)
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        accuracy = self.compute_accuracy(y_pred, y)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train_accuracy", accuracy, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y = y.squeeze(-1)
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        accuracy = self.compute_accuracy(y_pred, y)
        self.log("validation_loss", loss, prog_bar=True, on_epoch=True)
        self.log("validation_accuracy", accuracy, prog_bar=True, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        y = y.squeeze(-1)
        y_pred = self(x)
        loss = self.loss_fn(y_pred, y)
        accuracy = self.compute_accuracy(y_pred, y)
        self.log("test_loss", loss, prog_bar=True, on_epoch=True)
        self.log("test_accuracy", accuracy, prog_bar=True, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def write_metrics_to_csv(self, metrics: dict, phase: str, window_info: str = ""):
        date_str = datetime.now().strftime("%Y%m%d")
        filename = f"nn_metrics_{phase}_{self.start_year}_{window_info}_{date_str}.csv"
        os.makedirs("metrics", exist_ok=True)
        filepath = os.path.join("metrics", filename)
        file_exists = os.path.isfile(filepath)
        with open(filepath, "a", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=metrics.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics)

    def on_train_epoch_end(self):
        metrics = {
            'epoch': self.current_epoch,
            'train_loss': self.trainer.callback_metrics['train_loss'].item(),
            'train_accuracy': self.trainer.callback_metrics['train_accuracy'].item()
        }
        self.write_metrics_to_csv(metrics, phase='train')

    def on_validation_epoch_end(self):
        metrics = {
            'epoch': self.current_epoch,
            'validation_loss': self.trainer.callback_metrics['validation_loss'].item(),
            'validation_accuracy': self.trainer.callback_metrics['validation_accuracy'].item()
        }
        self.write_metrics_to_csv(metrics, phase='validation')

    def on_test_epoch_end(self):
        metrics = {
            'epoch': self.current_epoch,
            'test_loss': self.trainer.callback_metrics['test_loss'].item(),
            'test_accuracy': self.trainer.callback_metrics['test_accuracy'].item()
        }
        self.write_metrics_to_csv(metrics, phase='test')
