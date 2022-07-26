from math import sqrt

import torch
from torch import nn


class SlotAttentionModule(nn.Module):

    def __init__(self, num_slots: int, channels_enc: int, latent_size: int, iters=3, eps=1e-8, hidden_dim=128):
        super().__init__()
        self.num_slots = num_slots
        self.iters = iters
        self.eps = eps
        self.scale = latent_size ** -0.5
        self.latent_size = latent_size

        self.slots_mu = nn.Parameter(torch.rand(1, 1, latent_size))
        self.slots_log_sigma = nn.Parameter(torch.randn(1, 1, latent_size))
        with torch.no_grad():
            limit = sqrt(6.0 / (1 + latent_size))
            torch.nn.init.uniform_(self.slots_mu, -limit, limit)
            torch.nn.init.uniform_(self.slots_log_sigma, -limit, limit)
        self.to_q = nn.Linear(latent_size, latent_size, bias=False)
        self.to_k = nn.Linear(channels_enc, latent_size, bias=False)
        self.to_v = nn.Linear(channels_enc, latent_size, bias=False)

        self.gru = nn.GRUCell(latent_size, latent_size)

        hidden_dim = max(latent_size, hidden_dim)

        self.mlp = nn.Sequential(
            nn.Linear(latent_size, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_size)
        )

        self.norm_input = nn.LayerNorm(channels_enc, eps=0.001)
        self.norm_slots = nn.LayerNorm(latent_size, eps=0.001)
        self.norm_pre_ff = nn.LayerNorm(latent_size, eps=0.001)

    def forward(self, inputs, num_slots=None):
        b, n, _ = inputs.shape
        n_s = num_slots if num_slots is not None else self.num_slots

        mu = self.slots_mu.expand(b, n_s, -1)
        sigma = self.slots_log_sigma.expand(b, n_s, -1).exp()
        slots = torch.normal(mu, sigma)

        inputs = self.norm_input(inputs)
        k, v = self.to_k(inputs), self.to_v(inputs)

        for _ in range(self.iters):
            slots_prev = slots

            slots = self.norm_slots(slots)
            q = self.to_q(slots)

            dots = torch.einsum('bid,bjd->bij', q, k) * self.scale
            attn = dots.softmax(dim=1) + self.eps
            attn = attn / attn.sum(dim=-1, keepdim=True)

            updates = torch.einsum('bjd,bij->bid', v, attn)

            slots = self.gru(
                updates.reshape(-1, self.dim),
                slots_prev.reshape(-1, self.dim)
            )

            slots = slots.reshape(b, -1, self.dim)
            slots = slots + self.mlp(self.norm_pre_ff(slots))

        return slots