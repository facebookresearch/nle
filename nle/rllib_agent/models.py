from typing import Optional, Any, Dict, Tuple

import gym
import torch
from nle import nethack
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from torch import nn
from torch.nn import functional as F
from typeguard import typechecked


def _step_to_range(delta, num_steps):
    """Range of `num_steps` integers with distance `delta` centered around zero."""
    return delta * torch.arange(-num_steps // 2, num_steps // 2)


@typechecked
class Crop(nn.Module):
    """Helper class for NetHackNet below."""

    def __init__(self, height: int, width: int, height_target: int, width_target: int):
        super(Crop, self).__init__()
        self.width = width
        self.height = height
        self.width_target = width_target
        self.height_target = height_target
        width_grid = _step_to_range(2 / (self.width - 1), self.width_target)[
            None, :
        ].expand(self.height_target, -1)
        height_grid = _step_to_range(2 / (self.height - 1), height_target)[
            :, None
        ].expand(-1, self.width_target)

        # "clone" necessary, https://github.com/pytorch/pytorch/issues/34880
        self.register_buffer("width_grid", width_grid.clone())
        self.register_buffer("height_grid", height_grid.clone())

    def forward(self, inputs: torch.Tensor, coordinates: torch.Tensor) -> torch.Tensor:
        """Calculates centered crop around given x,y coordinates.
        Args:
           inputs [B x H x W]
           coordinates [B x 2] x,y coordinates
        Returns:
           [B x H' x W'] inputs cropped and centered around x,y coordinates.
        """
        assert inputs.shape[1] == self.height
        assert inputs.shape[2] == self.width

        inputs = inputs[:, None, :, :].float()

        x = coordinates[:, 0]
        y = coordinates[:, 1]

        x_shift = 2 / (self.width - 1) * (x.float() - self.width // 2)
        y_shift = 2 / (self.height - 1) * (y.float() - self.height // 2)

        grid = torch.stack(
            [
                self.width_grid[None, :, :] + x_shift[:, None, None],
                self.height_grid[None, :, :] + y_shift[:, None, None],
            ],
            dim=3,
        )

        # TODO: only cast to int if original tensor was int
        return (
            torch.round(F.grid_sample(inputs, grid, align_corners=True))
            .squeeze(1)
            .long()
        )


@typechecked
class NetHackNet(nn.Module):
    def __init__(
        self,
        processed_observation_space: gym.spaces.Space,
        embedding_dim: int = 32,
        crop_dim: int = 9,
        num_layers: int = 5,
        hidden_dim: int = 512,
    ):
        super(NetHackNet, self).__init__()

        self.observation_space = processed_observation_space.original_space

        self.glyph_shape = self.observation_space["glyphs"].shape
        self.blstats_size = self.observation_space["blstats"].shape[0]

        self.H = self.glyph_shape[0]
        self.W = self.glyph_shape[1]

        self.k_dim = embedding_dim
        self.h_dim = hidden_dim

        self.crop_dim = crop_dim

        self.crop = Crop(self.H, self.W, self.crop_dim, self.crop_dim)

        self.embed = nn.Embedding(nethack.MAX_GLYPH, self.k_dim)

        K = embedding_dim  # number of input filters
        F = 3  # filter dimensions
        S = 1  # stride
        P = 1  # padding
        M = 16  # number of intermediate filters
        Y = 8  # number of output filters
        L = num_layers  # number of convnet layers

        in_channels = [K] + [M] * (L - 1)
        out_channels = [M] * (L - 1) + [Y]

        def interleave(xs, ys):
            return [val for pair in zip(xs, ys) for val in pair]

        conv_extract = [
            nn.Conv2d(
                in_channels=in_channels[i],
                out_channels=out_channels[i],
                kernel_size=(F, F),
                stride=S,
                padding=P,
            )
            for i in range(L)
        ]

        self.extract_representation = nn.Sequential(
            *interleave(conv_extract, [nn.ELU()] * len(conv_extract))
        )

        # CNN crop model.
        conv_extract_crop = [
            nn.Conv2d(
                in_channels=in_channels[i],
                out_channels=out_channels[i],
                kernel_size=(F, F),
                stride=S,
                padding=P,
            )
            for i in range(L)
        ]

        self.extract_crop_representation = nn.Sequential(
            *interleave(conv_extract_crop, [nn.ELU()] * len(conv_extract))
        )

        out_dim = self.k_dim
        # CNN over full glyph map
        out_dim += self.H * self.W * Y

        # CNN crop model.
        out_dim += self.crop_dim ** 2 * Y

        self.embed_blstats = nn.Sequential(
            nn.Linear(self.blstats_size, self.k_dim),
            nn.ReLU(),
            nn.Linear(self.k_dim, self.k_dim),
            nn.ReLU(),
        )

        self.fc = nn.Sequential(
            nn.Linear(out_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
        )

    def _select(self, embed: nn.Embedding, x: torch.Tensor) -> torch.Tensor:
        # Work around slow backward pass of nn.Embedding, see
        # https://github.com/pytorch/pytorch/issues/24912
        out = embed.weight.index_select(0, x.reshape(-1))
        return out.reshape(x.shape + (-1,))

    def forward(self, env_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # -- [B x H x W]
        glyphs = env_outputs["glyphs"]

        # -- [B x F]
        blstats = env_outputs["blstats"]

        B, *_ = glyphs.shape

        # -- [B x H x W]
        glyphs = glyphs.long()
        # -- [B x 2] x,y coordinates
        coordinates = blstats[:, :2]

        # -- [B x K]
        blstats_emb = self.embed_blstats(blstats)

        assert blstats_emb.shape[0] == B

        reps = [blstats_emb]

        # -- [B x H' x W']
        crop = self.crop(glyphs, coordinates)

        # -- [B x H' x W' x K]
        crop_emb = self._select(self.embed, crop)

        # CNN crop model.
        # -- [B x K x W' x H']
        crop_emb = crop_emb.transpose(1, 3)
        # -- [B x W' x H' x K]
        crop_rep = self.extract_crop_representation(crop_emb)

        # -- [B x K']
        crop_rep = crop_rep.view(B, -1)
        assert crop_rep.shape[0] == B

        reps.append(crop_rep)

        # -- [B x H x W x K]
        glyphs_emb = self._select(self.embed, glyphs)
        # glyphs_emb = self.embed(glyphs)
        # -- [B x K x W x H]
        glyphs_emb = glyphs_emb.transpose(1, 3)
        # -- [B x W x H x K]
        glyphs_rep = self.extract_representation(glyphs_emb)

        # -- [B x K']
        glyphs_rep = glyphs_rep.view(B, -1)

        assert glyphs_rep.shape[0] == B

        # -- [B x K'']
        reps.append(glyphs_rep)

        st = torch.cat(reps, dim=1)

        # -- [B x K]
        st = self.fc(st)

        return st


@typechecked
class RLLibNLENetwork(TorchModelV2, nn.Module):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        num_outputs: Optional[int],
        model_config: dict,
        name: str,
        **kwargs: dict,
    ):
        TorchModelV2.__init__(
            self, observation_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        custom_model_config = dict(
            kwargs, **model_config.get("custom_model_config", {})
        )
        self.num_outputs = custom_model_config["hidden_dim"]

        self.base = NetHackNet(
            observation_space,
            embedding_dim=custom_model_config.get("embedding_dim"),
            crop_dim=custom_model_config.get("crop_dim"),
            num_layers=custom_model_config.get("num_layers"),
            hidden_dim=custom_model_config.get("hidden_dim"),
        )

    @override(TorchModelV2)
    def forward(self, x: Dict[str, Any], *_: Any) -> Tuple[torch.Tensor, None]:
        return self.base(x["obs"]), None


ModelCatalog.register_custom_model("rllib_nle_model", RLLibNLENetwork)
