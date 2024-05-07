# Copyright (c) 2024 Qualcomm Technologies, Inc.
# All Rights Reserved.

import torch


def create_model(
    model_name,
    pretrained=False,
    checkpoint_path="",
    scriptable=None,
    exportable=None,
    no_jit=None,
    **kwargs
):
    """
    overriding timm's create_model() function to make it work with skipat.
    the signature is deliberately kept identical to timm function, and therefore some arguments are not used.
    """
    num_classes = kwargs["num_classes"]

    # Baseline models
    if model_name == "vit_tiny_patch16_224":
        from timm.models.vision_transformer import vit_tiny_patch16_224

        model = vit_tiny_patch16_224(num_classes=num_classes)
    elif model_name == "vit_small_patch16_224":
        from timm.models.vision_transformer import vit_small_patch16_224

        model = vit_small_patch16_224(num_classes=num_classes)
    elif model_name == "vit_base_patch16_224":
        from timm.models.vision_transformer import vit_base_patch16_224

        model = vit_base_patch16_224(num_classes=num_classes)
    # SkipAt
    elif model_name == "skipat_tiny_patch16_224":
        from model.vit_skipat import skipat_tiny_patch16_224

        model = skipat_tiny_patch16_224(num_classes=num_classes)
    elif model_name == "skipat_small_patch16_224":
        from model.vit_skipat import skipat_small_patch16_224

        model = skipat_small_patch16_224(num_classes=num_classes)
    elif model_name == "skipat_base_patch16_224":
        from model.vit_skipat import skipat_base_patch16_224

        model = skipat_base_patch16_224(num_classes=num_classes)
    else:
        raise NotImplementedError(
            "Supported model types are: vit_tiny_patch16_224, vit_small_patch16_224, "
            "skipat_tiny_patch16_224, skipat_small_patch16_224"
        )

    return model


def load_checkpoint(model, checkpoint_path, use_ema=False, strict=True):
    """
    overriding timm's load_checkpoint() function to make it work with skipat.
    the signature is deliberately kept identical to timm function, and therefore some arguments are not used.
    """
    checkpoint = torch.load(checkpoint_path)
    if "model" in checkpoint:
        checkpoint = checkpoint["model"]
    elif "state_dict_ema" in checkpoint:
        checkpoint = checkpoint["state_dict_ema"]
    elif "state_dict" in checkpoint:
        checkpoint = checkpoint["state_dict"]
    else:
        raise NotImplementedError("%s is not a valid checkpoint" % checkpoint_path)

    checkpoint_state = {}
    for k, v in checkpoint.items():
        if "module" in k:
            checkpoint_state[k[7:]] = v
        else:
            checkpoint_state[k] = v

    model.load_state_dict(checkpoint_state)
