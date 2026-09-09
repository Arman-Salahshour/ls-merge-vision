import torch
from torch import nn
from torch.nn import functional as F
from .transformer import TransformerStack


class TVAE(nn.Module):
    def __init__(self, input_dim=144, model_dim=256, latent_dim=144, depth=4, n_heads=4, n_layers_total=19, n_stages=4):
        super(TVAE, self).__init__()
        self.input_dim = input_dim
        '''model_dim is the transformer width, latent_dim is the bottleneck d_z'''
        self.model_dim = model_dim
        self.latent_dim = latent_dim

        '''Encoder block'''
        self.chunk_projection = nn.Linear(input_dim, model_dim)
        self.enc_depth_embedding = nn.Embedding(n_layers_total, model_dim)
        self.enc_stage_embedding = nn.Embedding(n_stages, model_dim)
        self.encoder = TransformerStack(dim=model_dim, depth=depth, n_heads=n_heads)
        self.fc_mu = nn.Linear(model_dim, latent_dim)
        self.fc_logvar = nn.Linear(model_dim, latent_dim)

        '''Decoder block'''
        self.latent_projection = nn.Linear(latent_dim, model_dim)
        self.dec_depth_embedding = nn.Embedding(n_layers_total, model_dim)
        self.dec_stage_embedding = nn.Embedding(n_stages, model_dim)
        self.decoder = TransformerStack(dim=model_dim, depth=depth, n_heads=n_heads)
        self.out_projection = nn.Linear(model_dim, input_dim)

    def encode(self, x, depth, stage):
        x = self.chunk_projection(x)
        d = self.enc_depth_embedding(depth)
        s = self.enc_stage_embedding(stage)
        '''conditioning is per sequence, broadcast across all tokens'''
        x = x + d[:, None, :] + s[:, None, :]
        x = self.encoder(x)
        return self.fc_mu(x), self.fc_logvar(x)

    def reparameterize(self, mu, logvar):
        '''sample at train time, use the mean at eval time so reconstruction is deterministic'''
        if not self.training:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z, depth, stage):
        x = self.latent_projection(z)
        d = self.dec_depth_embedding(depth)
        s = self.dec_stage_embedding(stage)
        x = x + d[:, None, :] + s[:, None, :]
        x = self.decoder(x)
        return self.out_projection(x)

    def forward(self, x, depth, stage):
        mu, logvar = self.encode(x, depth, stage)
        z = self.reparameterize(mu, logvar)
        xhat = self.decode(z, depth, stage)
        return xhat, mu, logvar


def vae_loss(xhat, x, mu, logvar, mask=None, beta=0.0):
    '''masked mse, normalize by the count of REAL values not the total'''
    if mask is None:
        recon = F.mse_loss(xhat, x)
    else:
        m = mask.float()
        recon = ((xhat - x) ** 2 * m).sum() / m.sum().clamp(min=1.0)

    '''kl per latent element, averaged over the batch, beta=0 disables it for stage B'''
    kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1))
    return recon + beta * kl, recon.detach(), kl.detach()