import torch
from torch import nn
from torch.nn import functional as F
from transformer import TransformerStack


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


    def encode(self, x, depth, stage):
        x = self.chunk_projection(x)
        d = self.enc_depth_embedding(depth)
        s = self.enc_stage_embedding(stage)
        '''conditioning is per sequence, broadcast across all tokens'''
        x = x + d[:, None, :] + s[:, None, :]
        x = self.encoder(x)
        return self.fc_mu(x), self.fc_logvar(x)



