# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

# Code adapted from:
# https://github.com/huggingface/pytorch-image-models/blob/v0.5.4/timm/models/vision_transformer.py
# Copyright 2021, Ross Wightman, Licensed under the Apache License, Version 2.0
# License is provided for attribution purposes only, not a contribution.


import logging
from functools import partial

import torch
from torch import nn
from timm.models.helpers import build_model_with_cfg
from timm.models.layers import PatchEmbed
from timm.models.registry import register_model
from timm.models.vision_transformer import (
    Block,
    VisionTransformer,
    _cfg,
    checkpoint_filter_fn,
)

from model.uformer_utils import LeFF

_logger = logging.getLogger(__name__)

default_cfgs = {
    "skipat_tiny_patch16_224": _cfg(url=""),
    "skipat_small_patch16_224": _cfg(url=""),
    "skipat_base_patch16_224": _cfg(url=""),
}


class BlockSkipAt(Block):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__(
            dim, num_heads, mlp_ratio, qkv_bias, drop, attn_drop, drop_path, act_layer, norm_layer
        )

    def forward(self, x, feat_val=None):
        res = x
        if feat_val is None:
            val = self.attn(self.norm1(x))
        else:
            val = feat_val

        x = res + self.drop_path(val)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, val


class VisionTransformerSkipAt(VisionTransformer):
    """SkipAt with Vision Transformer

    A PyTorch impl of : `Skip-attention: Improving vision transformers by paying less attention`
        - https://arxiv.org/abs/2301.02240
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        representation_size=None,
        distilled=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        embed_layer=PatchEmbed,
        norm_layer=None,
        act_layer=None,
        weight_init="",
        skip_indices=(3, 4, 5, 6, 7, 8),
    ):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]):
              enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer (nn.Module): normalization layer
            weight_init (str): weight init scheme
            skip_indices (tuple): block indices to get skipped for SkipAt
        """
        super().__init__(
            img_size,
            patch_size,
            in_chans,
            num_classes,
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            qkv_bias,
            representation_size,
            distilled,
            drop_rate,
            attn_drop_rate,
            drop_path_rate,
            embed_layer,
            norm_layer,
            act_layer,
            weight_init,
        )
        self.skip = skip_indices

        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, depth)
        ]  # stochastic depth decay rule
        self.blocks = nn.Sequential(
            *[
                BlockSkipAt(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    act_layer=act_layer,
                )
                for i in range(depth)
            ]
        )

        self.param_1 = LeFF(dim=embed_dim, hidden_dim=2 * embed_dim)
        self.param_2 = LeFF(dim=embed_dim, hidden_dim=2 * embed_dim)
        self.param_3 = LeFF(dim=embed_dim, hidden_dim=2 * embed_dim)
        self.param_4 = LeFF(dim=embed_dim, hidden_dim=2 * embed_dim)
        self.param_5 = LeFF(dim=embed_dim, hidden_dim=2 * embed_dim)
        self.param_6 = LeFF(dim=embed_dim, hidden_dim=2 * embed_dim)
        self.parametric = [
            self.param_1,
            self.param_2,
            self.param_3,
            self.param_4,
            self.param_5,
            self.param_6,
        ]

    def forward_features(self, x):
        """
         A modified version from timm's vision_transformer.
        """
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(
            x.shape[0], -1, -1
        )  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        i_param = 0
        features_rest, features_prev = None, None
        for cnt, blk in enumerate(self.blocks):
            if cnt not in self.skip:
                x, features = blk(x, feat_val=None)
                cls_token = features[:, 0, :].unsqueeze(1)
                features_rest = features[:, 1:, :]
            else:
                if i_param == 0:
                    features_prev = features_rest
                features_cur = self.parametric[i_param](features_prev)
                x, _ = blk(x, feat_val=torch.cat((cls_token, features_cur), dim=1))
                features_prev = features_cur
                i_param = i_param + 1

        x = self.norm(x)
        return self.pre_logits(x[:, 0])


def _create_vision_transformer(variant, pretrained=False, default_cfg=None, **kwargs):
    """ create SkipAt model from config file.
    Modified version of _create_vision_transformer() from timm's vision_transformer.
    """
    default_cfg = default_cfg or default_cfgs[variant]
    if kwargs.get("features_only", None):
        raise RuntimeError("features_only not implemented for Vision Transformer models.")

    # NOTE this extra code to support handling of repr size for in21k pretrained models
    default_num_classes = default_cfg["num_classes"]
    num_classes = kwargs.get("num_classes", default_num_classes)
    repr_size = kwargs.pop("representation_size", None)
    if repr_size is not None and num_classes != default_num_classes:
        # Remove representation layer if fine-tuning. This may not always be the desired
        # action, but I feel better than doing nothing by default for fine-tuning. Perhaps
	# a better interface?
        _logger.warning("Removing representation layer for fine-tuning.")
        repr_size = None

    model = build_model_with_cfg(
        VisionTransformerSkipAt,
        variant,
        pretrained,
        default_cfg=default_cfg,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load="npz" in default_cfg["url"],
        **kwargs
    )
    return model


@register_model
def skipat_tiny_patch16_224(pretrained=False, **kwargs):
    """ SkipAt-Tiny (SkipAt-Ti/16)
    A modified version from timm's vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = _create_vision_transformer(
        "skipat_tiny_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def skipat_small_patch16_224(pretrained=False, **kwargs):
    """ SkipAt-Small (SkipAt-S/16)
     A modified version from timm's vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_vision_transformer(
        "skipat_small_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model


@register_model
def skipat_base_patch16_224(pretrained=False, **kwargs):
    """ SkipAt-Base (SkipAt-B/16)
     A modified version from timm's vision_transformer.
    """
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        "skipat_base_patch16_224", pretrained=pretrained, **model_kwargs
    )
    return model
