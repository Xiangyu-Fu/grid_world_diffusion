# Hydra app
# import hydra
from omegaconf import DictConfig, OmegaConf

# Torch
import torch
from torch import nn, Tensor
import torch.nn.functional as F
# from PIL import Image
import torchvision
from torchviz import make_dot
# from torchvision.transforms import ToPILImage

# Python
import math
import numpy as np
import einops
from pathlib import Path
from collections import deque

from grid_world.conf.configuration_diffusion import DiffusionConfig
from grid_world.utils.utils import _make_noise_scheduler, _replace_submodules

class DiffusionPolicy(
    nn.Module,
):
    def __init__(
            self, 
            config: DiffusionConfig | None = None, 
            dataset_stats: dict[str, dict[str, Tensor]] | None = None,  
            ):
        super().__init__()
        if config is None:
            config = DiffusionConfig()  # Here we use the default config
        self.config = config

        # # TODO: replace with my own implementation
        # self.normalize_inputs = Normalize(
        #     config.input_shapes, config.input_normalization_modes, dataset_stats
        # )
        # self.normalize_targets = Normalize(
        #     config.output_shapes, config.output_normalization_modes, dataset_stats
        # )
        # self.unnormalize_outputs = Unnormalize(
        #     config.output_shapes, config.output_normalization_modes, dataset_stats
        # )

        self._queues = None

        self.diffusion = DiffusionModel(config)

        self.expected_image_keys = [k for k in config.input_shapes if k.startswith("observation.image")]

        self.reset()

    def reset(self):
        """Clear observation and action queues. Should be called on `env.reset()`"""
        self._queues = {
            "observation.state": deque(maxlen=self.config.n_obs_steps),
            "action": deque(maxlen=self.config.n_action_steps),
        }
        if len(self.expected_image_keys) > 0:
            self._queues["observation.images"] = deque(maxlen=self.config.n_obs_steps)
        # if self.use_env_state:
        #     self._queues["observation.environment_state"] = deque(maxlen=self.config.n_obs_steps)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        # Do something with the input batch
        # input shape: (36864, 96) = (64*2*3*96, 96)
        # batch = self.normalize_inputs(batch)
        if len(self.expected_image_keys) > 0:
            batch = dict(batch)  # Copy the batch to avoid modifying the original
            batch["observation.images"] = torch.stack([batch[k] for k in self.expected_image_keys], dim=-4)
        # batch = self.normalize_targets(batch) 

        loss = self.diffusion.compute_loss(batch)  # TODO
        # Return the output dictionary
        return {"loss": loss}


class DiffusionModel(nn.Module):
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config

        # Build observation encoders (depending on which observations are provided).
        # config.input_shapes = {'observation.image': [3, 96, 96], 'observation.state': [2]}
        global_cond_dim = config.input_shapes["observation.state"][0] 
        num_images = len([k for k in config.input_shapes if k.startswith("observation.image")])
        self._use_images = False
        self._use_env_state = False
        if num_images > 0:
            self._use_images = True
            self.rgb_encoder = DiffusionRgbEncoder(config)
            global_cond_dim += config.diffusion_step_embed_dim  # 2+128

        # U-Net
        self.unet = DiffusionConditionalUnet1d(config, global_cond_dim=global_cond_dim * config.n_obs_steps)  # TODO

        # Build the noise scheduler 
        # TODO: check the code
        self.noise_scheduler = _make_noise_scheduler(
            config.noise_scheduler_type,
            num_train_timesteps=config.num_train_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            clip_sample_range=config.clip_sample_range,
            prediction_type=config.prediction_type,
        )

        if config.num_inference_steps is not None:
            self.num_inference_steps = self.noise_scheduler.config.num_train_timesteps
        else:
            self.num_inference_steps = config.num_inference_steps

    def _prepare_global_conditioning(self, batch: dict[str, Tensor]) -> Tensor:
        '''全局条件特征的编码, 从不同来源的特征（如状态、图像、环境状态）中提取有用信息，
        并将这些信息整合到一个统一的向量中。
        '''
        batch_size, n_obs_steps = batch["observation.state"].shape[:2]
        global_cond_feats = [batch["observation.state"]]

        if self._use_images:
            img_features = self.rgb_encoder(
                einops.rearrange(batch["observation.images"], "b s n ... -> (b s n ) ...")
            )

            img_features = einops.rearrange(
                img_features, 
                "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
            )

            global_cond_feats.append(img_features)

        if self._use_env_state:
            global_cond_feats.append(batch["observation.environment_state"])
        
        return torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)

        
    def compute_loss(self, batch: dict[str, Tensor]) -> Tensor:
        """
        This function expects `batch` to have (at least):
        {

            "observation.state": (B, n_obs_steps, state_dim)

            "observation.images": (B, n_obs_steps, num_cameras, C, H, W)
                AND/OR
            "observation.environment_state": (B, environment_dim)

            "action": (B, horizon, action_dim)
            "action_is_pad": (B, horizon)
        }
        """
        # 1. input validation
        # assert set(batch).issuperset({"observation.state", "action", "action_is_pad"})
        # assert "observation.images" in batch or "observation.environment_state" in batch
        n_obs_steps = batch["observation.state"].shape[1] # ？
        horizon = batch["action"].shape[1] 
        assert n_obs_steps == self.config.n_obs_steps
        assert horizon == self.config.horizon

        # 2. prepare global conditioning
        global_cond = self._prepare_global_conditioning(batch)

        # 3. Forward diffusion
        trajectory = batch["action"]
        # sample the noise
        eps = torch.randn(trajectory.shape, device=trajectory.device)
        timesteps = torch.randint(
            low=0,
            high=self.noise_scheduler.config.num_train_timesteps,
            size=(trajectory.shape[0],),
            device=trajectory.device,
        )
        # TODO: 
        noise_trajecotry = self.noise_scheduler.add_noise(trajectory, eps, timesteps)

        # 4. run the denoising network
        pred = self.unet(noise_trajecotry, timesteps, global_cond)

        # 5.compute the loss 
        if self.config.prediction_type == "epsilon":
            target = eps  # 学习预测噪声
        elif self.config.prediction_type == "sample":
            target = batch["action"]  # 学习恢复原始数据
        else:
            raise ValueError(f"Unknown prediction type: {self.config.prediction_type}")
        
        loss = F.mse_loss(pred, target, reduction="none")

        return loss.mean()

    
# ================== Encoder ==================
class DiffusionRgbEncoder(nn.Module):
    '''Encode the image observation into a 1D feature vector.
    '''
    def __init__(self, config:DiffusionConfig):
        super().__init__()
        self.config = config
        if config.crop_shape is not None:
            self.do_crop = True
            self.center_crop = torchvision.transforms.CenterCrop(config.crop_shape)
            if config.crop_is_random:
                self.maybe_random_crop = torchvision.transforms.RandomCrop(config.crop_shape)
            else:
                self.maybe_random_crop = self.center_crop
        else:
            self.do_crop = False

        # set up the vision backbone
        # getattr 获取动态属性， 等效于
        # backbone_model = torchvision.models.resnet18(pretrained=True)
        backbone_model = getattr(torchvision.models, config.vision_backbone)(
            pretrained=config.pretrained_backbone_weights
        )
        self.backbone = nn.Sequential(*(list(backbone_model.children())[:-2]))
        # 替换批归一化层为组归一化层
        if config.use_group_norm:
            if config.pretrained_backbone_weights:
                raise ValueError(
                    "You can't replace BatchNorm in a pretrained model without ruining the weights!"
                )
            self.backbone = _replace_submodules(
                root_module=self.backbone,
                predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                func=lambda x: nn.GroupNorm(num_groups=x.num_features // 16, num_channels=x.num_features),
            )

        # set up pooling and final layers
        image_keys = [k for k in config.input_shapes if k.startswith("observation.image")]
        image_key = image_keys[0]
        dummy_input_h_w = (
            config.crop_shape if config.crop_shape is not None else config.input_shapes[image_key][1:]
            )
        dummy_input = torch.zeros(size=(1, config.input_shapes[image_key][0], *dummy_input_h_w))
        with torch.inference_mode():
            dummy_feature_map = self.backbone(dummy_input)
        feature_map_shape = tuple(dummy_feature_map.shape[1:])
        self.pool = SptialSoftmax(feature_map_shape, config.spatial_softmax_num_keypoints)
        self.feature_dim = config.spatial_softmax_num_keypoints * 2  # 2 for x and y
        self.out = nn.Linear(self.feature_dim, config.diffusion_step_embed_dim)
        self.relu = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (batch_size, C, H, W) image tensor with pixel values in [0, 1]
        Returns:
            (batch_size, diffusion_step_embed_dim) tensor
        """
        if self.do_crop:
            if self.training:
                x = self.maybe_random_crop(x)
            else:
                x = self.center_crop(x)  # Center crop during evaluation
            
        # Extract features
        x = torch.flatten(self.pool(self.backbone(x)), start_dim=1)
        x = self.relu(self.out(x))
        return x
    

class SptialSoftmax(nn.Module):
    def __init__(self, input_shape, num_keypoints=None):
        super().__init__()
        
        assert len(input_shape) == 3
        self._in_c, self._in_h, self._in_w = input_shape

        if num_keypoints is not None:
            self.nets = torch.nn.Conv2d(self._in_c, num_keypoints, kernel_size=1)
            self._out_c = num_keypoints
        else:
            self.nets = None
            self._out_c = self._in_c

        # 1. Create a meshgrid of the input shape
        pos_x, pos_y = np.meshgrid(np.linspace(-1, 1, self._in_w), np.linspace(-1, 1, self._in_h))
        pos_x = torch.from_numpy(pos_x.reshape(self._in_h * self._in_w, 1)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self._in_h * self._in_w, 1)).float()
        self.register_buffer("pos_grid", torch.cat([pos_x, pos_y], dim=1))

    def forward(self, features:Tensor) -> Tensor:
        if self.nets is not None:
            features = self.nets(features)
        
        # [B, K, H, W] -> [B * K, H * W]
        features = features.reshape(-1, self._in_h * self._in_w)
        # 2. Apply softmax to get the weights
        attention = torch.nn.functional.softmax(features, dim=1)
        # 3. Compute the weighted sum of the positions
        excepted_xy = attention @ self.pos_grid

        feature_keypoints = excepted_xy.reshape(-1, self._out_c, 2)

        return feature_keypoints

# ================== Unet ==================
class DiffusionConditionalUnet1d(nn.Module):
    '''实现了一个一维卷积的条件 U-Net 模型，并结合了 FiLM(Feature-wise Linear Modulation)调节,
    用于扩散模型的应用场景。U-Net 是一种常见的网络结构，通常用于图像分割任务，但这里使用的是一维卷积，
    因此适用于时间序列或类似的任务。
    该模型采用编码器-解码器结构，利用 skip connections来保留输入的高分辨率特征,并结合扩散过程的条件信息进行建模。
    '''
    def __init__(self, config:DiffusionConfig, global_cond_dim:int):
        super().__init__()
        self.config = config

        # 1. 扩散步骤编码器, 将时间步长转化为一个与输入特征兼容的嵌入
        self.diffusion_step_encoder = nn.Sequential(
            DiffusionSinusoidalPosEmb(config.diffusion_step_embed_dim),
            nn.Linear(config.diffusion_step_embed_dim, config.diffusion_step_embed_dim * 4),
            nn.Mish(),
            nn.Linear(config.diffusion_step_embed_dim * 4, config.diffusion_step_embed_dim),
        )

        # 2. FiLM条件维度 388 = 128 + 260
        cond_dim = config.diffusion_step_embed_dim + global_cond_dim  # 这里的conditional dimension是什么？

        # 3. 下采样编码器, len(config.down_dims) = 3
        in_out = [(config.output_shapes["action"][0], config.down_dims[0])] + list(
            zip(config.down_dims[:-1], config.down_dims[1:], strict=True)
        )

        # 4. unet 编码器模块
        common_res_block_kwargs = {
            "cond_dim": cond_dim,
            "kernel_size": config.kernel_size,
            "n_groups": config.n_groups,
            "use_film_scale_modulation": config.use_film_scale_modulation,
        }
        self.down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList(
                    [
                        DiffusionConditionalResidualBlock1d(dim_in, dim_out, **common_res_block_kwargs),
                        DiffusionConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                        # Downsample as long as it is not the last block.
                        nn.Conv1d(dim_out, dim_out, 3, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )

        # 5. 中间处理模块
        self.mid_modules = nn.ModuleList(
            [
                DiffusionConditionalResidualBlock1d(
                    config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs
                ),
                DiffusionConditionalResidualBlock1d(
                    config.down_dims[-1], config.down_dims[-1], **common_res_block_kwargs
                ),
            ]
        )

        # 6. 上采样解码器
        self.up_modules = nn.ModuleList([])
        for ind, (dim_out, dim_in) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList(
                    [
                        # dim_in * 2, because it takes the encoder's skip connection as well
                        DiffusionConditionalResidualBlock1d(dim_in * 2, dim_out, **common_res_block_kwargs),
                        DiffusionConditionalResidualBlock1d(dim_out, dim_out, **common_res_block_kwargs),
                        # Upsample as long as it is not the last block.
                        nn.ConvTranspose1d(dim_out, dim_out, 4, 2, 1) if not is_last else nn.Identity(),
                    ]
                )
            )
        
        # 7. 最终卷积层
        self.final_conv = nn.Sequential(
            DiffusionConv1dBlock(config.down_dims[0], config.down_dims[0], kernel_size=config.kernel_size),
            nn.Conv1d(config.down_dims[0], config.output_shapes["action"][0], 1),
        )
    
    def forward(self, x:Tensor, timestep:Tensor|int, global_cond=None) -> Tensor:
        """
        """
        # For 1D convolutions we'll need feature dimension first.
        x = einops.rearrange(x, "b t d -> b d t")

        # Encode the diffusion step.
        timesteps_embed = self.diffusion_step_encoder(timestep)

        # If there is a global conditioning feature, concatenate it to the timestep embedding.
        if global_cond is not None:
            global_feature = torch.cat([timesteps_embed, global_cond], axis=-1)
        else:
            global_feature = timesteps_embed

        # Run encoder, keeping track of skip features to pass to the decoder.
        encoder_skip_features: list[Tensor] = []
        for resnet, resnet2, downsample in self.down_modules:
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            encoder_skip_features.append(x)
            x = downsample(x)

        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)

        # Run decoder, using the skip features from the encoder.
        for resnet, resnet2, upsample in self.up_modules:
            x = torch.cat((x, encoder_skip_features.pop()), dim=1)
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)

        x = self.final_conv(x)

        x = einops.rearrange(x, "b d t -> b t d")
        return x


class DiffusionSinusoidalPosEmb(nn.Module):
    """这个类实现了 1D 正弦位置嵌入，用于为时间序列数据提供位置信息。
    通过正弦和余弦函数生成嵌入向量，并结合不同频率的缩放因子来区分不同的位置。
    嵌入维度 dim 被分为两部分：一部分用于正弦嵌入，另一部分用于余弦嵌入。
    TODO: look in detail about how the embedding is generated.
    Args:
        dim (int): 嵌入向量的维度
    Output:
        emb (Tensor): 位置嵌入向量, shape 为 (B, T, dim)"""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class DiffusionConv1dBlock(nn.Module):
    """Conv1d --> GroupNorm --> Mish
    这个模块适用于时间序列数据或一维特征的处理场景，包含了一个卷积层、组归一化层和 Mish 激活函数。"""

    def __init__(self, inp_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x):
        return self.block(x)


class DiffusionConditionalResidualBlock1d(nn.Module):
    """DiffusionConditionalResidualBlock1d 实现了一个带有条件调节(FiLM)的 1D 卷积残差块。
    该结构适用于处理时间序列或一维数据，允许通过条件张量对特征进行灵活的调节。
    - FiLM 调节：可以调节每个通道的缩放(scale)和偏移(bias)，使得模型能够根据条件信息动态调整特征的分布。
    - 残差连接：保证了梯度的稳定传递，并提供了更好的训练表现。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        # Set to True to do scale modulation with FiLM as well as bias modulation (defaults to False meaning
        # FiLM just modulates bias).
        use_film_scale_modulation: bool = False,
    ):
        super().__init__()

        self.use_film_scale_modulation = use_film_scale_modulation
        self.out_channels = out_channels

        self.conv1 = DiffusionConv1dBlock(in_channels, out_channels, kernel_size, n_groups=n_groups)

        # FiLM modulation (https://arxiv.org/abs/1709.07871) outputs per-channel bias and (maybe) scale.
        cond_channels = out_channels * 2 if use_film_scale_modulation else out_channels # if Film, then 2*out_channels
        self.cond_encoder = nn.Sequential(nn.Mish(), nn.Linear(cond_dim, cond_channels))

        self.conv2 = DiffusionConv1dBlock(out_channels, out_channels, kernel_size, n_groups=n_groups)

        # A final convolution for dimension matching the residual (if needed).
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        """
        Args:
            x: (B, in_channels, T)
            cond: (B, cond_dim)
        Returns:
            (B, out_channels, T)
        """
        out = self.conv1(x)

        # Get condition embedding. Unsqueeze for broadcasting to `out`, resulting in (B, out_channels, 1).
        cond_embed = self.cond_encoder(cond).unsqueeze(-1)
        if self.use_film_scale_modulation:
            # Treat the embedding as a list of scales and biases.
            scale = cond_embed[:, : self.out_channels]
            bias = cond_embed[:, self.out_channels :]
            out = scale * out + bias
        else:
            # Treat the embedding as biases.
            out = out + cond_embed

        out = self.conv2(out)
        out = out + self.residual_conv(x)
        return out
