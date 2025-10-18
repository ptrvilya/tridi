import torch
from torch import nn
import numpy as np


def get_timestep_embedding(embed_dim, timesteps, device):
    """
    Timestep embedding function. Note that this should work just as well for
    continuous values as for discrete values.
    """

    assert len(timesteps.shape) == 1
    half_dim = embed_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = torch.from_numpy(np.exp(np.arange(0, half_dim) * -emb)).float().to(device)
    emb = timesteps[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embed_dim % 2 == 1:  # zero pad
        emb = nn.functional.pad(emb, (0, 1), "constant", 0)
    assert emb.shape == torch.Size([timesteps.shape[0], embed_dim])
    return emb


class Projection(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_out: int
    ):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(d_in, d_out),
            nn.SiLU(),
            nn.Linear(d_out, d_out),
        )

    def forward(self, x):
        return self.projection(x)


class TransformertUni3WayModel(nn.Module):
    def __init__(
        self,
        dim_sbj: int,
        dim_obj: int,
        dim_contact: int,
        dim_cond: int,
        dim_hidden: int,
        dim_timestep_embed: int,
        dim_output: int,
        num_layers: int = 4,
        nhead: int = 8,
        dim_feedforward: int = 512,
    ):
        super().__init__()
        self.dim_sbj = dim_sbj
        self.dim_obj = dim_obj
        self.dim_contact = dim_contact
        self.dim_cond = dim_cond
        self.dim_hidden = dim_hidden
        self.dim_output = dim_output
        self.dim_timestep_embed = dim_timestep_embed
        self.num_layers = num_layers

        # Time projection function
        self.projection_T = Projection(self.dim_timestep_embed, self.dim_hidden)

        # Input projection (sbj, obj, contact, cond)
        self.projection_S_shape = Projection(10, dim_hidden)
        self.projection_S_orient = Projection(6, dim_hidden)
        self.projection_S_pose = Projection(dim_sbj - 19, dim_hidden)
        self.projection_S_transl = Projection(3, dim_hidden)
        self.projection_O_orient = Projection(6, dim_hidden)
        self.projection_O_transl = Projection(3, dim_hidden)
        self.projection_CNT = Projection(dim_contact, dim_hidden)
        self.projection_C = Projection(dim_cond, dim_hidden)

        # Modality embeddings
        self.sbj_embedding = nn.Parameter(torch.randn(dim_hidden))
        self.obj_embedding = nn.Parameter(torch.randn(dim_hidden))
        self.cnt_embedding = nn.Parameter(torch.randn(dim_hidden))
        self.cond_embedding = nn.Parameter(torch.randn(dim_hidden))

        # Param embeddings
        self.sbj_pose_embedding = nn.Parameter(torch.randn(dim_hidden))
        self.sbj_shape_embedding = nn.Parameter(torch.randn(dim_hidden))
        self.global_orient_embedding = nn.Parameter(torch.randn(dim_hidden))
        self.global_transl_embedding = nn.Parameter(torch.randn(dim_hidden))

        # Normalization
        self.layernorm = nn.LayerNorm(dim_hidden)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_hidden, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # Decoders
        self.decoder_S_shape = Projection(dim_hidden, 10)
        self.decoder_S_orient = Projection(dim_hidden, 6)
        self.decoder_S_pose = Projection(dim_hidden, dim_sbj - 19)
        self.decoder_S_transl = Projection(dim_hidden, 3)
        self.decoder_O_orient = Projection(dim_hidden, 6)
        self.decoder_O_transl = Projection(dim_hidden, 3)
        self.decoder_CNT = Projection(dim_hidden, dim_contact)

    def prepare_time(self, t_1, t_2, t_3, device):
        # Embed and project timesteps
        t_emb_1 = get_timestep_embedding(self.dim_timestep_embed, t_1, device)
        t_emb_1 = self.projection_T(t_emb_1) + self.sbj_embedding  # B x D_t_emb
        t_emb_2 = get_timestep_embedding(self.dim_timestep_embed, t_2, device)
        t_emb_2 = self.projection_T(t_emb_2) +self.obj_embedding # B x D_t_emb
        t_emb_3 = get_timestep_embedding(self.dim_timestep_embed, t_3, device)
        t_emb_3 = self.projection_T(t_emb_3) + self.cnt_embedding # B x D_t_emb

        return t_emb_1.unsqueeze(1), t_emb_2.unsqueeze(1), t_emb_3.unsqueeze(1)

    def prepare_sbj(self, sbj):
        shape, global_orient, pose, global_transl = torch.split(sbj, [10, 6, 51*6, 3], dim=1)

        shape = self.projection_S_shape(shape)
        shape = shape + self.sbj_embedding + self.sbj_shape_embedding

        global_orient = self.projection_S_orient(global_orient)
        global_orient = global_orient + self.sbj_embedding + self.global_orient_embedding

        pose = self.projection_S_pose(pose)
        pose = pose + self.sbj_embedding + self.sbj_pose_embedding

        global_transl = self.projection_S_transl(global_transl)
        global_transl = global_transl + self.sbj_embedding + self.global_transl_embedding

        return torch.stack([shape, global_orient, pose, global_transl], dim=1)  # B x 4 x D

    def prepare_obj(self, obj):
        global_orient, global_transl = torch.split(obj, [6, 3], dim=1)

        global_orient = self.projection_O_orient(global_orient)
        global_orient = global_orient + self.obj_embedding + self.global_orient_embedding

        global_transl = self.projection_O_transl(global_transl)
        global_transl = global_transl + self.obj_embedding + self.global_transl_embedding

        return torch.stack([global_orient, global_transl], dim=1)  # B x 2 x D

    def unembed_prediction(self, x):
        # sbj is shape, global_orient, pose, transl
        sbj = torch.cat([
            self.decoder_S_shape(x[:, 0]),
            self.decoder_S_orient(x[:, 1]),
            self.decoder_S_pose(x[:, 2]),
            self.decoder_S_transl(x[:, 3]),
        ], dim=1)

        # obj is global_orient, transl
        obj = torch.cat([
            self.decoder_O_orient(x[:, 4]),
            self.decoder_O_transl(x[:, 5]),
        ], dim=1)

        cnt = self.decoder_CNT(x[:, 6])

        return sbj, obj, cnt

    def forward(self, sbj, obj, contact, cond, t1, t2, t3):
        t_1, t_2, t_3 = self.prepare_time(t1, t2, t3, sbj.device)
        sbj = self.prepare_sbj(sbj)
        obj = self.prepare_obj(obj)
        cnt = (self.projection_CNT(contact) + self.cnt_embedding).unsqueeze(1)
        cond = (self.projection_C(cond) + self.cond_embedding).unsqueeze(1)

        x = torch.cat([sbj, obj, cnt, cond, t_1, t_2, t_3], dim=1)
        x = self.layernorm(x)
        x = self.transformer_encoder(x)

        sbj, obj, contact = self.unembed_prediction(x)

        return torch.cat([sbj, obj, contact], dim=1)
