"""Converts a DeepCompressor state dict to a Nunchaku state dict."""

import argparse
import os

import safetensors.torch
import torch
import tqdm

from deepcompressor.backend.nunchaku.utils import convert_to_nunchaku_w4x4y16_linear_weight, convert_to_nunchaku_w4x16_linear_weight


def convert_to_nunchaku_w4x4y16_linear_state_dict(
    weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor | None = None,
    smooth: torch.Tensor | None = None,
    lora: tuple[torch.Tensor, torch.Tensor] | None = None,
    shift: torch.Tensor | None = None,
    smooth_fused: bool = False,
    float_point: bool = False,
    subscale: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if weight.ndim > 2:  # pointwise conv
        assert weight.numel() == weight.shape[0] * weight.shape[1]
        weight = weight.view(weight.shape[0], weight.shape[1])
    if scale.numel() > 1:
        assert scale.ndim == weight.ndim * 2
        assert scale.numel() == scale.shape[0] * scale.shape[2]
        scale = scale.view(scale.shape[0], 1, scale.shape[2], 1)
        scale_key = "wcscales" if scale.shape[2] == 1 else "wscales"
    else:
        scale_key = "wtscale"
    if subscale is None:
        subscale_key = ""
    else:
        assert subscale.ndim == weight.ndim * 2
        assert subscale.numel() == subscale.shape[0] * subscale.shape[2]
        assert subscale.numel() > 1
        subscale = subscale.view(subscale.shape[0], 1, subscale.shape[2], 1)
        subscale_key = "wcscales" if subscale.shape[2] == 1 else "wscales"
    if lora is not None and (smooth is not None or shift is not None):
        # unsmooth lora down projection
        dtype = weight.dtype
        lora_down, lora_up = lora
        lora_down = lora_down.to(dtype=torch.float64)
        if smooth is not None and not smooth_fused:
            lora_down = lora_down.div_(smooth.to(torch.float64).unsqueeze(0))
        if shift is not None:
            bias = torch.zeros([lora_up.shape[0]], dtype=torch.float64) if bias is None else bias.to(torch.float64)
            if shift.numel() == 1:
                shift = shift.view(1, 1).expand(lora_down.shape[1], 1).to(torch.float64)
            else:
                shift = shift.view(-1, 1).to(torch.float64)
            bias = bias.add_((lora_up.to(dtype=torch.float64) @ lora_down @ shift).view(-1))
            bias = bias.to(dtype=dtype)
        lora = (lora_down.to(dtype=dtype), lora_up)
    weight, scale, bias, smooth, lora, subscale = convert_to_nunchaku_w4x4y16_linear_weight(
        weight, scale=scale, bias=bias, smooth=smooth, lora=lora, float_point=float_point, subscale=subscale
    )
    state_dict: dict[str, torch.Tensor] = {}
    state_dict["qweight"] = weight
    state_dict[scale_key] = scale
    if subscale is not None:
        state_dict[subscale_key] = subscale
    state_dict["bias"] = bias
    state_dict["smooth_factor_orig"] = smooth
    state_dict["smooth_factor"] = torch.ones_like(smooth) if smooth_fused else smooth.clone()
    if lora is not None:
        state_dict["proj_down"] = lora[0]
        state_dict["proj_up"] = lora[1]
    return state_dict


def convert_to_nunchaku_w4x16_adanorm_single_state_dict(
    weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
) -> dict[str, torch.Tensor]:
    weight, scale, zero, bias = convert_to_nunchaku_w4x16_linear_weight(
        weight, scale=scale, bias=bias, adanorm_splits=3
    )
    state_dict: dict[str, torch.Tensor] = {}
    state_dict = {}
    state_dict["qweight"] = weight
    state_dict["wscales"] = scale
    state_dict["wzeros"] = zero
    state_dict["bias"] = bias
    return state_dict


def convert_to_nunchaku_w4x16_adanorm_zero_state_dict(
    weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
) -> dict[str, torch.Tensor]:
    weight, scale, zero, bias = convert_to_nunchaku_w4x16_linear_weight(
        weight, scale=scale, bias=bias, adanorm_splits=6
    )
    state_dict: dict[str, torch.Tensor] = {}
    state_dict = {}
    state_dict["qweight"] = weight
    state_dict["wscales"] = scale
    state_dict["wzeros"] = zero
    state_dict["bias"] = bias
    return state_dict


def convert_to_nunchaku_w4x16_adanorm_mod_state_dict(
    weight: torch.Tensor,
    scale: torch.Tensor,
    bias: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Convert QwenImage modulation layer (img_mod[1] or txt_mod[1]) to Nunchaku format.

    QwenImage uses 6 modulation parameters (shift1, scale1, gate1, shift2, scale2, gate2)
    which is the same split count as adanorm_zero.
    """
    weight, scale, zero, bias = convert_to_nunchaku_w4x16_linear_weight(
        weight, scale=scale, bias=bias, adanorm_splits=6
    )
    state_dict: dict[str, torch.Tensor] = {}
    state_dict["qweight"] = weight
    state_dict["wscales"] = scale
    state_dict["wzeros"] = zero
    state_dict["bias"] = bias
    return state_dict


def update_state_dict(
    lhs: dict[str, torch.Tensor], rhs: dict[str, torch.Tensor], prefix: str = ""
) -> dict[str, torch.Tensor]:
    for rkey, value in rhs.items():
        lkey = f"{prefix}.{rkey}" if prefix else rkey
        assert lkey not in lhs, f"Key {lkey} already exists in the state dict."
        lhs[lkey] = value
    return lhs


def convert_to_nunchaku_transformer_block_state_dict(
    state_dict: dict[str, torch.Tensor],
    scale_dict: dict[str, torch.Tensor],
    smooth_dict: dict[str, torch.Tensor],
    branch_dict: dict[str, torch.Tensor],
    block_name: str,
    local_name_map: dict[str, str | list[str]],
    smooth_name_map: dict[str, str],
    branch_name_map: dict[str, str],
    convert_map: dict[str, str],
    float_point: bool = False,
) -> dict[str, torch.Tensor]:
    print(f"Converting block {block_name}...")
    converted: dict[str, torch.Tensor] = {}
    candidates: dict[str, torch.Tensor] = {
        param_name: param for param_name, param in state_dict.items() if param_name.startswith(block_name)
    }
    for converted_local_name, candidate_local_names in tqdm.tqdm(
        local_name_map.items(), desc=f"Converting {block_name}", dynamic_ncols=True
    ):
        if isinstance(candidate_local_names, str):
            candidate_local_names = [candidate_local_names]
        candidate_names = [f"{block_name}.{candidate_local_name}" for candidate_local_name in candidate_local_names]
        weight = [candidates[f"{candidate_name}.weight"] for candidate_name in candidate_names]
        bias = [candidates.get(f"{candidate_name}.bias", None) for candidate_name in candidate_names]
        scale = [scale_dict.get(f"{candidate_name}.weight.scale.0", None) for candidate_name in candidate_names]
        subscale = [scale_dict.get(f"{candidate_name}.weight.scale.1", None) for candidate_name in candidate_names]
        if len(weight) > 1:
            bias = None if all(b is None for b in bias) else torch.concat(bias, dim=0)
            if all(s is None for s in scale):
                scale = None
            else:
                if scale[0].numel() == 1:  # switch from per-tensor to per-channel scale
                    assert all(s.numel() == 1 for s in scale)
                    scale = torch.concat(
                        [
                            s.view(-1).expand(weight[i].shape[0]).reshape(weight[i].shape[0], 1, 1, 1)
                            for i, s in enumerate(scale)
                        ],
                        dim=0,
                    )
                else:
                    scale = torch.concat(scale, dim=0)
            subscale = None if all(s is None for s in subscale) else torch.concat(subscale, dim=0)
            weight = torch.concat(weight, dim=0)
        else:
            weight, bias, scale, subscale = weight[0], bias[0], scale[0], subscale[0]
        smooth = smooth_dict.get(f"{block_name}.{smooth_name_map.get(converted_local_name, '')}", None)
        branch = branch_dict.get(f"{block_name}.{branch_name_map.get(converted_local_name, '')}", None)
        if branch is not None:
            branch = (branch["a.weight"], branch["b.weight"])
        if scale is None:
            assert smooth is None and branch is None and subscale is None
            print(f"  - Copying {block_name} weights of {candidate_local_names} as {converted_local_name}.weight")
            converted[f"{converted_local_name}.weight"] = weight.clone().cpu()
            if bias is not None:
                print(f"  - Copying {block_name} biases of {candidate_local_names} as {converted_local_name}.bias")
                converted[f"{converted_local_name}.bias"] = bias.clone().cpu()
            continue
        if convert_map[converted_local_name] == "adanorm_single":
            print(f"  - Converting {block_name} weights of {candidate_local_names} to {converted_local_name}.")
            update_state_dict(
                converted,
                convert_to_nunchaku_w4x16_adanorm_single_state_dict(weight=weight, scale=scale, bias=bias),
                prefix=converted_local_name,
            )
        elif convert_map[converted_local_name] == "adanorm_zero":
            print(f"  - Converting {block_name} weights of {candidate_local_names} to {converted_local_name}.")
            update_state_dict(
                converted,
                convert_to_nunchaku_w4x16_adanorm_zero_state_dict(weight=weight, scale=scale, bias=bias),
                prefix=converted_local_name,
            )
        elif convert_map[converted_local_name] == "adanorm_mod":
            print(f"  - Converting {block_name} weights of {candidate_local_names} to {converted_local_name}.")
            update_state_dict(
                converted,
                convert_to_nunchaku_w4x16_adanorm_mod_state_dict(weight=weight, scale=scale, bias=bias),
                prefix=converted_local_name,
            )
        elif convert_map[converted_local_name] == "linear":
            smooth_fused = "out_proj" in converted_local_name and smooth_dict.get("proj.fuse_when_possible", True)
            # Try to find shift key - handle cases like "img_mlp.net.2.linear.weight" where shift is at "img_mlp.net.2.shift"
            shift_list = []
            for candidate_name in candidate_names:
                base_name = candidate_name[:-7]  # Remove ".weight"
                shift_val = candidates.get(f"{base_name}.shift", None)
                if shift_val is None:
                    # Try removing ".linear" suffix if present (e.g., for QwenImage MLP down proj)
                    if base_name.endswith(".linear"):
                        alt_base = base_name[:-7]  # Remove ".linear"
                        shift_val = candidates.get(f"{alt_base}.shift", None)
                shift_list.append(shift_val)
            assert all(s == shift_list[0] for s in shift_list), f"Inconsistent shift values: {shift_list}"
            shift = shift_list[0]
            print(
                f"  - Converting {block_name} weights of {candidate_local_names} to {converted_local_name}."
                f" (smooth_fused={smooth_fused}, shifted={shift is not None}, float_point={float_point})"
            )
            update_state_dict(
                converted,
                convert_to_nunchaku_w4x4y16_linear_state_dict(
                    weight=weight,
                    scale=scale,
                    bias=bias,
                    smooth=smooth,
                    lora=branch,
                    shift=shift,
                    smooth_fused=smooth_fused,
                    float_point=float_point,
                    subscale=subscale,
                ),
                prefix=converted_local_name,
            )
        else:
            raise NotImplementedError(f"Conversion of {convert_map[converted_local_name]} is not implemented.")
    return converted


def convert_to_nunchaku_flux_single_transformer_block_state_dict(
    state_dict: dict[str, torch.Tensor],
    scale_dict: dict[str, torch.Tensor],
    smooth_dict: dict[str, torch.Tensor],
    branch_dict: dict[str, torch.Tensor],
    block_name: str,
    float_point: bool = False,
) -> dict[str, torch.Tensor]:
    down_proj_local_name = "proj_out.linears.1.linear"
    if f"{block_name}.{down_proj_local_name}.weight" not in state_dict:
        down_proj_local_name = "proj_out.linears.1"
        assert f"{block_name}.{down_proj_local_name}.weight" in state_dict

    return convert_to_nunchaku_transformer_block_state_dict(
        state_dict=state_dict,
        scale_dict=scale_dict,
        smooth_dict=smooth_dict,
        branch_dict=branch_dict,
        block_name=block_name,
        local_name_map={
            "norm.linear": "norm.linear",
            "qkv_proj": ["attn.to_q", "attn.to_k", "attn.to_v"],
            "norm_q": "attn.norm_q",
            "norm_k": "attn.norm_k",
            "out_proj": "proj_out.linears.0",
            "mlp_fc1": "proj_mlp",
            "mlp_fc2": down_proj_local_name,
        },
        smooth_name_map={
            "qkv_proj": "attn.to_q",
            "out_proj": "proj_out.linears.0",
            "mlp_fc1": "attn.to_q",
            "mlp_fc2": down_proj_local_name,
        },
        branch_name_map={
            "qkv_proj": "attn.to_q",
            "out_proj": "proj_out.linears.0",
            "mlp_fc1": "proj_mlp",
            "mlp_fc2": down_proj_local_name,
        },
        convert_map={
            "norm.linear": "adanorm_single",
            "qkv_proj": "linear",
            "out_proj": "linear",
            "mlp_fc1": "linear",
            "mlp_fc2": "linear",
        },
        float_point=float_point,
    )


def convert_to_nunchaku_flux_transformer_block_state_dict(
    state_dict: dict[str, torch.Tensor],
    scale_dict: dict[str, torch.Tensor],
    smooth_dict: dict[str, torch.Tensor],
    branch_dict: dict[str, torch.Tensor],
    block_name: str,
    float_point: bool = False,
) -> dict[str, torch.Tensor]:
    down_proj_local_name = "ff.net.2.linear"
    if f"{block_name}.{down_proj_local_name}.weight" not in state_dict:
        down_proj_local_name = "ff.net.2"
        assert f"{block_name}.{down_proj_local_name}.weight" in state_dict
    context_down_proj_local_name = "ff_context.net.2.linear"
    if f"{block_name}.{context_down_proj_local_name}.weight" not in state_dict:
        context_down_proj_local_name = "ff_context.net.2"
        assert f"{block_name}.{context_down_proj_local_name}.weight" in state_dict

    return convert_to_nunchaku_transformer_block_state_dict(
        state_dict=state_dict,
        scale_dict=scale_dict,
        smooth_dict=smooth_dict,
        branch_dict=branch_dict,
        block_name=block_name,
        local_name_map={
            "norm1.linear": "norm1.linear",
            "norm1_context.linear": "norm1_context.linear",
            "qkv_proj": ["attn.to_q", "attn.to_k", "attn.to_v"],
            "qkv_proj_context": ["attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"],
            "norm_q": "attn.norm_q",
            "norm_k": "attn.norm_k",
            "norm_added_q": "attn.norm_added_q",
            "norm_added_k": "attn.norm_added_k",
            "out_proj": "attn.to_out.0",
            "out_proj_context": "attn.to_add_out",
            "mlp_fc1": "ff.net.0.proj",
            "mlp_fc2": down_proj_local_name,
            "mlp_context_fc1": "ff_context.net.0.proj",
            "mlp_context_fc2": context_down_proj_local_name,
        },
        smooth_name_map={
            "qkv_proj": "attn.to_q",
            "qkv_proj_context": "attn.add_k_proj",
            "out_proj": "attn.to_out.0",
            "out_proj_context": "attn.to_out.0",
            "mlp_fc1": "ff.net.0.proj",
            "mlp_fc2": down_proj_local_name,
            "mlp_context_fc1": "ff_context.net.0.proj",
            "mlp_context_fc2": context_down_proj_local_name,
        },
        branch_name_map={
            "qkv_proj": "attn.to_q",
            "qkv_proj_context": "attn.add_k_proj",
            "out_proj": "attn.to_out.0",
            "out_proj_context": "attn.to_add_out",
            "mlp_fc1": "ff.net.0.proj",
            "mlp_fc2": down_proj_local_name,
            "mlp_context_fc1": "ff_context.net.0.proj",
            "mlp_context_fc2": context_down_proj_local_name,
        },
        convert_map={
            "norm1.linear": "adanorm_zero",
            "norm1_context.linear": "adanorm_zero",
            "qkv_proj": "linear",
            "qkv_proj_context": "linear",
            "out_proj": "linear",
            "out_proj_context": "linear",
            "mlp_fc1": "linear",
            "mlp_fc2": "linear",
            "mlp_context_fc1": "linear",
            "mlp_context_fc2": "linear",
        },
        float_point=float_point,
    )


def convert_to_nunchaku_flux_state_dicts(
    state_dict: dict[str, torch.Tensor],
    scale_dict: dict[str, torch.Tensor],
    smooth_dict: dict[str, torch.Tensor],
    branch_dict: dict[str, torch.Tensor],
    float_point: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    block_names: set[str] = set()
    other: dict[str, torch.Tensor] = {}
    for param_name in state_dict.keys():
        if param_name.startswith(("transformer_blocks.", "single_transformer_blocks.")):
            block_names.add(".".join(param_name.split(".")[:2]))
        else:
            other[param_name] = state_dict[param_name]
    block_names = sorted(block_names, key=lambda x: (x.split(".")[0], int(x.split(".")[-1])))
    print(f"Converting {len(block_names)} transformer blocks...")
    converted: dict[str, torch.Tensor] = {}
    for block_name in block_names:
        convert_fn = convert_to_nunchaku_flux_single_transformer_block_state_dict
        if block_name.startswith("transformer_blocks"):
            convert_fn = convert_to_nunchaku_flux_transformer_block_state_dict
        update_state_dict(
            converted,
            convert_fn(
                state_dict=state_dict,
                scale_dict=scale_dict,
                smooth_dict=smooth_dict,
                branch_dict=branch_dict,
                block_name=block_name,
                float_point=float_point,
            ),
            prefix=block_name,
        )
    return converted, other


def convert_to_nunchaku_qwenimage_transformer_block_state_dict(
    state_dict: dict[str, torch.Tensor],
    scale_dict: dict[str, torch.Tensor],
    smooth_dict: dict[str, torch.Tensor],
    branch_dict: dict[str, torch.Tensor],
    block_name: str,
    float_point: bool = False,
) -> dict[str, torch.Tensor]:
    """Convert a QwenImage transformer block to Nunchaku format.

    QwenImage dual-stream architecture layer mapping:
    - img_mod[1] → mod (W4A16 modulation)
    - txt_mod[1] → add_mod (W4A16 modulation)
    - attn.to_q/to_k/to_v → qkv_proj (image stream)
    - attn.add_q_proj/add_k_proj/add_v_proj → qkv_proj_context (text stream)
    - attn.to_out[0] → out_proj (image output)
    - attn.to_add_out → out_proj_context (text output)
    - img_mlp.net.0.proj → mlp_fc1 (image MLP)
    - img_mlp.net.2 → mlp_fc2 (image MLP)
    - txt_mlp.net.0.proj → mlp_context_fc1 (text MLP)
    - txt_mlp.net.2 → mlp_context_fc2 (text MLP)
    """
    # Handle potential .linear suffix variations in MLP down projections
    img_down_proj_local_name = "img_mlp.net.2.linear"
    if f"{block_name}.{img_down_proj_local_name}.weight" not in state_dict:
        img_down_proj_local_name = "img_mlp.net.2"
        assert f"{block_name}.{img_down_proj_local_name}.weight" in state_dict

    txt_down_proj_local_name = "txt_mlp.net.2.linear"
    if f"{block_name}.{txt_down_proj_local_name}.weight" not in state_dict:
        txt_down_proj_local_name = "txt_mlp.net.2"
        assert f"{block_name}.{txt_down_proj_local_name}.weight" in state_dict

    return convert_to_nunchaku_transformer_block_state_dict(
        state_dict=state_dict,
        scale_dict=scale_dict,
        smooth_dict=smooth_dict,
        branch_dict=branch_dict,
        block_name=block_name,
        local_name_map={
            # Modulation layers (W4A16) - keep original names for Nunchaku compatibility
            "img_mod.1": "img_mod.1",
            "txt_mod.1": "txt_mod.1",
            # Attention QKV projections - merged to single layer
            "attn.to_qkv": ["attn.to_q", "attn.to_k", "attn.to_v"],
            "attn.add_qkv_proj": ["attn.add_q_proj", "attn.add_k_proj", "attn.add_v_proj"],
            # Attention normalization layers (non-quantized, direct copy)
            "attn.norm_q": "attn.norm_q",
            "attn.norm_k": "attn.norm_k",
            "attn.norm_added_q": "attn.norm_added_q",
            "attn.norm_added_k": "attn.norm_added_k",
            # Attention output projections
            "attn.to_out.0": "attn.to_out.0",
            "attn.to_add_out": "attn.to_add_out",
            # Image MLP - keep original structure
            "img_mlp.net.0.proj": "img_mlp.net.0.proj",
            "img_mlp.net.2": img_down_proj_local_name,
            # Text MLP - keep original structure
            "txt_mlp.net.0.proj": "txt_mlp.net.0.proj",
            "txt_mlp.net.2": txt_down_proj_local_name,
        },
        smooth_name_map={
            "attn.to_qkv": "attn.to_q",
            "attn.add_qkv_proj": "attn.add_k_proj",
            "attn.to_out.0": "attn.to_out.0",
            "attn.to_add_out": "attn.to_out.0",
            "img_mlp.net.0.proj": "img_mlp.net.0.proj",
            "img_mlp.net.2": img_down_proj_local_name,
            "txt_mlp.net.0.proj": "txt_mlp.net.0.proj",
            "txt_mlp.net.2": txt_down_proj_local_name,
            "img_mod.1": "img_mod.1",
            "txt_mod.1": "txt_mod.1",
        },
        branch_name_map={
            "attn.to_qkv": "attn.to_q",
            "attn.add_qkv_proj": "attn.add_k_proj",
            "attn.to_out.0": "attn.to_out.0",
            "attn.to_add_out": "attn.to_add_out",
            "img_mlp.net.0.proj": "img_mlp.net.0.proj",
            "img_mlp.net.2": img_down_proj_local_name,
            "txt_mlp.net.0.proj": "txt_mlp.net.0.proj",
            "txt_mlp.net.2": txt_down_proj_local_name,
        },
        convert_map={
            # Modulation layers use adanorm_mod (W4A16 with 6 splits)
            "img_mod.1": "adanorm_mod",
            "txt_mod.1": "adanorm_mod",
            # All other quantized layers use linear (W4A4)
            "attn.to_qkv": "linear",
            "attn.add_qkv_proj": "linear",
            "attn.to_out.0": "linear",
            "attn.to_add_out": "linear",
            "img_mlp.net.0.proj": "linear",
            "img_mlp.net.2": "linear",
            "txt_mlp.net.0.proj": "linear",
            "txt_mlp.net.2": "linear",
        },
        float_point=float_point,
    )


def convert_to_nunchaku_qwenimage_state_dicts(
    state_dict: dict[str, torch.Tensor],
    scale_dict: dict[str, torch.Tensor],
    smooth_dict: dict[str, torch.Tensor],
    branch_dict: dict[str, torch.Tensor],
    float_point: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Convert QwenImage model state dict to Nunchaku format.

    QwenImage does NOT have single_transformer_blocks (unlike Flux).
    All transformer blocks are dual-stream blocks.

    Non-quantized layers (directly copied):
    - img_in, txt_in: Input projections
    - txt_norm: Text normalization (RMSNorm)
    - time_text_embed: Timestep embedding
    - pos_embed: Position encoding (RoPE)
    - norm_out, proj_out: Output layers
    - img_norm1, img_norm2, txt_norm1, txt_norm2: Block layer norms
    - norm_q, norm_k, norm_added_q, norm_added_k: Attention QK norms
    """
    block_names: set[str] = set()
    other: dict[str, torch.Tensor] = {}
    for param_name in state_dict.keys():
        # QwenImage only has transformer_blocks, no single_transformer_blocks
        if param_name.startswith("transformer_blocks."):
            block_names.add(".".join(param_name.split(".")[:2]))
        else:
            other[param_name] = state_dict[param_name]
    block_names = sorted(block_names, key=lambda x: int(x.split(".")[-1]))
    print(f"Converting {len(block_names)} QwenImage transformer blocks...")
    converted: dict[str, torch.Tensor] = {}
    for block_name in block_names:
        update_state_dict(
            converted,
            convert_to_nunchaku_qwenimage_transformer_block_state_dict(
                state_dict=state_dict,
                scale_dict=scale_dict,
                smooth_dict=smooth_dict,
                branch_dict=branch_dict,
                block_name=block_name,
                float_point=float_point,
            ),
            prefix=block_name,
        )
    return converted, other


def convert_to_nunchaku_wan2_transformer_block_state_dict(
    state_dict: dict[str, torch.Tensor],
    scale_dict: dict[str, torch.Tensor],
    smooth_dict: dict[str, torch.Tensor],
    branch_dict: dict[str, torch.Tensor],
    block_name: str,
    float_point: bool = False,
) -> dict[str, torch.Tensor]:
    """Convert a WAN2 transformer block to Nunchaku format.

    WAN2 I2V architecture layer mapping:
    - blocks.N.attn1: Self-attention (q/k/v from same input)
      * attn1.to_q/to_k/to_v → attn1.to_qkv (merged)
      * attn1.to_out.0 → attn1.to_out.0
      * attn1.norm_q/norm_k → copied directly (non-quantized)
    - blocks.N.attn2: Cross-attention (q from hidden_states, k/v from encoder_hidden_states)
      * attn2.to_q → attn2.to_q (alone, different input source)
      * attn2.to_k/to_v → attn2.to_kv (merged, share encoder_hidden_states input)
      * attn2.to_out.0 → attn2.to_out.0
      * attn2.norm_q/norm_k → copied directly (non-quantized)
    - blocks.N.ffn: Feed-forward
      * ffn.net.0.proj → ffn.net.0.proj (up projection)
      * ffn.net.2.linear → ffn.net.2.linear (down projection, has shift)
    - blocks.N.norm2: Layer norm (non-quantized, copied directly)
    - blocks.N.scale_shift_table: Modulation parameters (non-quantized)
    """
    # Handle potential .linear suffix variations in FFN down projection
    ffn_down_proj_local_name = "ffn.net.2.linear"
    if f"{block_name}.{ffn_down_proj_local_name}.weight" not in state_dict:
        ffn_down_proj_local_name = "ffn.net.2"
        assert f"{block_name}.{ffn_down_proj_local_name}.weight" in state_dict

    # First, handle non-quantized parameters that don't follow .weight/.bias convention
    converted: dict[str, torch.Tensor] = {}

    # Copy scale_shift_table directly (AdaLN modulation parameter)
    scale_shift_table_key = f"{block_name}.scale_shift_table"
    if scale_shift_table_key in state_dict:
        converted["scale_shift_table"] = state_dict[scale_shift_table_key].clone().cpu()

    # Copy norm2 (layer norm with weight and bias)
    norm2_weight_key = f"{block_name}.norm2.weight"
    norm2_bias_key = f"{block_name}.norm2.bias"
    if norm2_weight_key in state_dict:
        converted["norm2.weight"] = state_dict[norm2_weight_key].clone().cpu()
    if norm2_bias_key in state_dict:
        converted["norm2.bias"] = state_dict[norm2_bias_key].clone().cpu()

    # Now convert quantized layers using the standard function
    quantized_converted = convert_to_nunchaku_transformer_block_state_dict(
        state_dict=state_dict,
        scale_dict=scale_dict,
        smooth_dict=smooth_dict,
        branch_dict=branch_dict,
        block_name=block_name,
        local_name_map={
            # Self-attention (attn1): Q/K/V merged together
            "attn1.to_qkv": ["attn1.to_q", "attn1.to_k", "attn1.to_v"],
            "attn1.norm_q": "attn1.norm_q",
            "attn1.norm_k": "attn1.norm_k",
            "attn1.to_out.0": "attn1.to_out.0",
            # Cross-attention (attn2): Q alone, K/V merged
            "attn2.to_q": "attn2.to_q",
            "attn2.to_kv": ["attn2.to_k", "attn2.to_v"],
            "attn2.norm_q": "attn2.norm_q",
            "attn2.norm_k": "attn2.norm_k",
            "attn2.to_out.0": "attn2.to_out.0",
            # Feed-forward
            "ffn.net.0.proj": "ffn.net.0.proj",
            "ffn.net.2.linear": ffn_down_proj_local_name,
        },
        smooth_name_map={
            # Self-attention smooth: Q is the reference for QKV
            "attn1.to_qkv": "attn1.to_q",
            "attn1.to_out.0": "attn1.to_out.0",
            # Cross-attention smooth: Q alone, K is the reference for KV
            "attn2.to_q": "attn2.to_q",
            "attn2.to_kv": "attn2.to_k",
            "attn2.to_out.0": "attn2.to_out.0",
            # FFN smooth
            "ffn.net.0.proj": "ffn.net.0.proj",
            "ffn.net.2.linear": ffn_down_proj_local_name,
        },
        branch_name_map={
            # Self-attention branch
            "attn1.to_qkv": "attn1.to_q",
            "attn1.to_out.0": "attn1.to_out.0",
            # Cross-attention branch
            "attn2.to_q": "attn2.to_q",
            "attn2.to_kv": "attn2.to_k",
            "attn2.to_out.0": "attn2.to_out.0",
            # FFN branch
            "ffn.net.0.proj": "ffn.net.0.proj",
            "ffn.net.2.linear": ffn_down_proj_local_name,
        },
        convert_map={
            # All quantized layers use W4A4 linear
            "attn1.to_qkv": "linear",
            "attn1.to_out.0": "linear",
            "attn2.to_q": "linear",
            "attn2.to_kv": "linear",
            "attn2.to_out.0": "linear",
            "ffn.net.0.proj": "linear",
            "ffn.net.2.linear": "linear",
        },
        float_point=float_point,
    )

    # Merge the non-quantized parameters with the quantized ones
    converted.update(quantized_converted)
    return converted


def convert_to_nunchaku_wan2_state_dicts(
    state_dict: dict[str, torch.Tensor],
    scale_dict: dict[str, torch.Tensor],
    smooth_dict: dict[str, torch.Tensor],
    branch_dict: dict[str, torch.Tensor],
    float_point: bool = False,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Convert WAN2 model state dict to Nunchaku format.

    WAN2 I2V structure:
    - blocks.N: WanTransformerBlock (attn1, attn2, ffn, norms)

    Non-quantized layers (directly copied):
    - time_embed, time_proj, text_embed: Time and text embeddings
    - img_embed: Image embedding (for I2V)
    - patch_embed: Patch embedding
    - head: Output head
    - norm1, norm2, norm3: Block layer norms
    - norm_q, norm_k: Attention QK norms
    - scale_shift_table: Modulation parameters
    """
    block_names: set[str] = set()
    other: dict[str, torch.Tensor] = {}
    for param_name in state_dict.keys():
        if param_name.startswith("blocks."):
            block_names.add(".".join(param_name.split(".")[:2]))
        else:
            other[param_name] = state_dict[param_name]
    block_names = sorted(block_names, key=lambda x: int(x.split(".")[-1]))
    print(f"Converting {len(block_names)} WAN2 transformer blocks...")
    converted: dict[str, torch.Tensor] = {}
    for block_name in block_names:
        update_state_dict(
            converted,
            convert_to_nunchaku_wan2_transformer_block_state_dict(
                state_dict=state_dict,
                scale_dict=scale_dict,
                smooth_dict=smooth_dict,
                branch_dict=branch_dict,
                block_name=block_name,
                float_point=float_point,
            ),
            prefix=block_name,
        )
    return converted, other


def _detect_model_type(model_name: str) -> str:
    """Detect the model type from the model name.

    Args:
        model_name: The model name string.

    Returns:
        The detected model type: "flux", "qwenimage", or "wan2".

    Raises:
        ValueError: If the model type cannot be detected.
    """
    model_name_lower = model_name.lower().replace("-", "").replace("_", "")
    if "qwenimage" in model_name_lower or "qwen2vl" in model_name_lower:
        return "qwenimage"
    elif "flux" in model_name_lower:
        return "flux"
    elif "wan" in model_name_lower:
        return "wan2"
    else:
        raise ValueError(
            f"Cannot detect model type from model name '{model_name}'. "
            "Supported model types: 'flux', 'qwenimage' (or 'qwen-image', 'qwen2vl'), 'wan2' (or 'wan')."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant-path", type=str, default="qwen-image1/", help="path to the quantization checkpoint directory.")
    parser.add_argument("--output-root", type=str, default="save_model/", help="root to the output checkpoint directory.")
    parser.add_argument("--model-name", type=str, default="qwen-image2/", help="name of the model.")
    parser.add_argument("--float-point", action="store_true", help="use float-point 4-bit quantization.")
    args = parser.parse_args()
    if not args.output_root:
        args.output_root = args.quant_path
    if args.model_name is None:
        assert args.model_path is not None, "model name or path is required."
        model_name = args.model_path.rstrip(os.sep).split(os.sep)[-1]
        print(f"Model name not provided, using {model_name} as the model name.")
    else:
        model_name = args.model_name
    assert model_name, "Model name must be provided."

    # Detect model type and select appropriate conversion function
    model_type = _detect_model_type(model_name)
    print(f"Detected model type: {model_type}")

    state_dict_path = os.path.join(args.quant_path, "model.pt")
    scale_dict_path = os.path.join(args.quant_path, "scale.pt")
    smooth_dict_path = os.path.join(args.quant_path, "smooth.pt")
    branch_dict_path = os.path.join(args.quant_path, "branch.pt")
    map_location = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
    state_dict = torch.load(state_dict_path, map_location=map_location)
    scale_dict = torch.load(scale_dict_path, map_location="cpu")
    smooth_dict = torch.load(smooth_dict_path, map_location=map_location) if os.path.exists(smooth_dict_path) else {}
    branch_dict = torch.load(branch_dict_path, map_location=map_location) if os.path.exists(branch_dict_path) else {}

    # Select conversion function based on model type
    if model_type == "qwenimage":
        convert_fn = convert_to_nunchaku_qwenimage_state_dicts
    elif model_type == "wan2":
        convert_fn = convert_to_nunchaku_wan2_state_dicts
    else:
        convert_fn = convert_to_nunchaku_flux_state_dicts

    converted_state_dict, other_state_dict = convert_fn(
        state_dict=state_dict,
        scale_dict=scale_dict,
        smooth_dict=smooth_dict,
        branch_dict=branch_dict,
        float_point=args.float_point,
    )
    output_dirpath = os.path.join(args.output_root, model_name)
    os.makedirs(output_dirpath, exist_ok=True)
    safetensors.torch.save_file(converted_state_dict, os.path.join(output_dirpath, "transformer_blocks.safetensors"))
    safetensors.torch.save_file(other_state_dict, os.path.join(output_dirpath, "unquantized_layers.safetensors"))
    print(f"Quantized model saved to {output_dirpath}.")
