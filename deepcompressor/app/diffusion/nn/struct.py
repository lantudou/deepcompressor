# -*- coding: utf-8 -*-
"""Utility functions for Diffusion Models."""

import enum
import typing as tp
from abc import abstractmethod
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

# region imports

import torch.nn as nn
from diffusers.models.activations import GEGLU, GELU, ApproximateGELU, SwiGLU
from diffusers.models.attention import BasicTransformerBlock, FeedForward, JointTransformerBlock
from diffusers.models.attention_processor import Attention, SanaLinearAttnProcessor2_0
from diffusers.models.embeddings import (
    CombinedTimestepGuidanceTextProjEmbeddings,
    CombinedTimestepTextProjEmbeddings,
    ImageHintTimeEmbedding,
    ImageProjection,
    ImageTimeEmbedding,
    PatchEmbed,
    PixArtAlphaTextProjection,
    TextImageProjection,
    TextImageTimeEmbedding,
    TextTimeEmbedding,
    TimestepEmbedding,
)
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormSingle, AdaLayerNormZero
from diffusers.models.resnet import Downsample2D, ResnetBlock2D, Upsample2D
from diffusers.models.transformers.pixart_transformer_2d import PixArtTransformer2DModel
from diffusers.models.transformers.sana_transformer import GLUMBConv, SanaTransformer2DModel, SanaTransformerBlock
from diffusers.models.transformers.transformer_2d import Transformer2DModel
from diffusers.models.transformers.transformer_flux import (
    FluxAttention,
    FluxSingleTransformerBlock,
    FluxTransformer2DModel,
    FluxTransformerBlock,
)
from diffusers.models.transformers.transformer_qwenimage import (
    QwenImageTransformer2DModel,
    QwenImageTransformerBlock,
)
from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
from diffusers.models.transformers.transformer_wan import (
    WanAttention,
    WanTransformer3DModel,
    WanTransformerBlock,
)
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.models.unets.unet_2d_blocks import (
    CrossAttnDownBlock2D,
    CrossAttnUpBlock2D,
    DownBlock2D,
    UNetMidBlock2D,
    UNetMidBlock2DCrossAttn,
    UpBlock2D,
)
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.pipelines import (
    FluxControlPipeline,
    FluxFillPipeline,
    FluxPipeline,
    PixArtAlphaPipeline,
    PixArtSigmaPipeline,
    SanaPipeline,
    StableDiffusion3Pipeline,
    StableDiffusionPipeline,
    StableDiffusionXLPipeline,
)

from deepcompressor.nn.patch.conv import ConcatConv2d, ShiftedConv2d
from deepcompressor.nn.patch.linear import ConcatLinear, ShiftedLinear
from deepcompressor.nn.struct.attn import (
    AttentionConfigStruct,
    AttentionStruct,
    BaseTransformerStruct,
    FeedForwardConfigStruct,
    FeedForwardStruct,
    TransformerBlockStruct,
)
from deepcompressor.nn.struct.base import BaseModuleStruct
from deepcompressor.utils.common import join_name

from .attention import DiffusionAttentionProcessor
from ..qwenimage import QwenImagePipelineWithSeqLen as QwenImagePipeline
from ..wan_i2v import WanImageToVideoPipelineForQuant

# endregion


__all__ = ["DiffusionModelStruct", "DiffusionBlockStruct", "DiffusionModelStruct"]


DIT_BLOCK_CLS = tp.Union[
    BasicTransformerBlock,
    JointTransformerBlock,
    FluxSingleTransformerBlock,
    FluxTransformerBlock,
    QwenImageTransformerBlock,
    WanTransformerBlock,
    SanaTransformerBlock,
]
UNET_BLOCK_CLS = tp.Union[
    DownBlock2D,
    CrossAttnDownBlock2D,
    UNetMidBlock2D,
    UNetMidBlock2DCrossAttn,
    UpBlock2D,
    CrossAttnUpBlock2D,
]
DIT_CLS = tp.Union[
    Transformer2DModel,
    PixArtTransformer2DModel,
    QwenImageTransformer2DModel,
    SD3Transformer2DModel,
    FluxTransformer2DModel,
    WanTransformer3DModel,
    SanaTransformer2DModel,
]
UNET_CLS = tp.Union[UNet2DModel, UNet2DConditionModel]
MODEL_CLS = tp.Union[DIT_CLS, UNET_CLS]
UNET_PIPELINE_CLS = tp.Union[StableDiffusionPipeline, StableDiffusionXLPipeline]
DIT_PIPELINE_CLS = tp.Union[
    StableDiffusion3Pipeline,
    PixArtAlphaPipeline,
    PixArtSigmaPipeline,
    FluxPipeline,
    FluxControlPipeline,
    FluxFillPipeline,
    QwenImagePipeline,
    WanImageToVideoPipelineForQuant,
    SanaPipeline,
]
PIPELINE_CLS = tp.Union[UNET_PIPELINE_CLS, DIT_PIPELINE_CLS]


@dataclass(kw_only=True)
class DiffusionModuleStruct(BaseModuleStruct):
    def named_key_modules(self) -> tp.Generator[tuple[str, str, nn.Module, BaseModuleStruct, str], None, None]:
        if isinstance(self.module, (nn.Linear, nn.Conv2d)):
            yield self.key, self.name, self.module, self.parent, self.fname
        else:
            for name, module in self.module.named_modules():
                if name and isinstance(module, (nn.Linear, nn.Conv2d)):
                    module_name = join_name(self.name, name)
                    field_name = join_name(self.fname, name)
                    yield self.key, module_name, module, self.parent, field_name


@dataclass(kw_only=True)
class DiffusionBlockStruct(BaseModuleStruct):
    @abstractmethod
    def iter_attention_structs(self) -> tp.Generator["DiffusionAttentionStruct", None, None]: ...

    @abstractmethod
    def iter_transformer_block_structs(self) -> tp.Generator["DiffusionTransformerBlockStruct", None, None]: ...


@dataclass(kw_only=True)
class DiffusionModelStruct(DiffusionBlockStruct):
    pre_module_structs: OrderedDict[str, DiffusionModuleStruct] = field(init=False, repr=False)
    post_module_structs: OrderedDict[str, DiffusionModuleStruct] = field(init=False, repr=False)

    @property
    @abstractmethod
    def num_blocks(self) -> int: ...

    @property
    @abstractmethod
    def block_structs(self) -> list[DiffusionBlockStruct]: ...

    @abstractmethod
    def get_prev_module_keys(self) -> tuple[str, ...]: ...

    @abstractmethod
    def get_post_module_keys(self) -> tuple[str, ...]: ...

    @abstractmethod
    def _get_iter_block_activations_args(
        self, **input_kwargs
    ) -> tuple[list[nn.Module], list[DiffusionModuleStruct | DiffusionBlockStruct], list[bool], list[bool]]: ...

    def _get_iter_pre_module_activations_args(
        self,
    ) -> tuple[list[nn.Module], list[DiffusionModuleStruct], list[bool], list[bool]]:
        layers, layer_structs, recomputes, use_prev_layer_outputs = [], [], [], []
        for layer_struct in self.pre_module_structs.values():
            layers.append(layer_struct.module)
            layer_structs.append(layer_struct)
            recomputes.append(False)
            use_prev_layer_outputs.append(False)
        return layers, layer_structs, recomputes, use_prev_layer_outputs

    def _get_iter_post_module_activations_args(
        self,
    ) -> tuple[list[nn.Module], list[DiffusionModuleStruct], list[bool], list[bool]]:
        layers, layer_structs, recomputes, use_prev_layer_outputs = [], [], [], []
        for layer_struct in self.post_module_structs.values():
            layers.append(layer_struct.module)
            layer_structs.append(layer_struct)
            recomputes.append(False)
            use_prev_layer_outputs.append(False)
        return layers, layer_structs, recomputes, use_prev_layer_outputs

    def get_iter_layer_activations_args(
        self, skip_pre_modules: bool, skip_post_modules: bool, **input_kwargs
    ) -> tuple[list[nn.Module], list[DiffusionModuleStruct | DiffusionBlockStruct], list[bool], list[bool]]:
        """
        Get the arguments for iterating over the layers and their activations.

        Args:
            skip_pre_modules (`bool`):
                Whether to skip the pre-modules
            skip_post_modules (`bool`):
                Whether to skip the post-modules

        Returns:
            `tuple[list[nn.Module], list[DiffusionModuleStruct | DiffusionBlockStruct], list[bool], list[bool]]`:
                the layers, the layer structs, the recomputes, and the use_prev_layer_outputs
        """
        layers, structs, recomputes, uses = [], [], [], []
        if not skip_pre_modules:
            layers, structs, recomputes, uses = self._get_iter_pre_module_activations_args()
        _layers, _structs, _recomputes, _uses = self._get_iter_block_activations_args(**input_kwargs)
        layers.extend(_layers)
        structs.extend(_structs)
        recomputes.extend(_recomputes)
        uses.extend(_uses)
        if not skip_post_modules:
            _layers, _structs, _recomputes, _uses = self._get_iter_post_module_activations_args()
            layers.extend(_layers)
            structs.extend(_structs)
            recomputes.extend(_recomputes)
            uses.extend(_uses)
        return layers, structs, recomputes, uses

    def named_key_modules(self) -> tp.Generator[tp.Tuple[str, str, nn.Module, BaseModuleStruct, str], None, None]:
        for module in self.pre_module_structs.values():
            yield from module.named_key_modules()
        for block in self.block_structs:
            yield from block.named_key_modules()
        for module in self.post_module_structs.values():
            yield from module.named_key_modules()

    def iter_attention_structs(self) -> tp.Generator["AttentionStruct", None, None]:
        for block in self.block_structs:
            yield from block.iter_attention_structs()

    def iter_transformer_block_structs(self) -> tp.Generator["DiffusionTransformerBlockStruct", None, None]:
        for block in self.block_structs:
            yield from block.iter_transformer_block_structs()

    def get_named_layers(
        self, skip_pre_modules: bool, skip_post_modules: bool, skip_blocks: bool = False
    ) -> OrderedDict[str, DiffusionBlockStruct | DiffusionModuleStruct]:
        named_layers = OrderedDict()
        if not skip_pre_modules:
            named_layers.update(self.pre_module_structs)
        if not skip_blocks:
            for block in self.block_structs:
                named_layers[block.name] = block
        if not skip_post_modules:
            named_layers.update(self.post_module_structs)
        return named_layers

    @staticmethod
    def _default_construct(
        module: tp.Union[PIPELINE_CLS, MODEL_CLS],
        /,
        parent: tp.Optional[BaseModuleStruct] = None,
        fname: str = "",
        rname: str = "",
        rkey: str = "",
        idx: int = 0,
        **kwargs,
    ) -> "DiffusionModelStruct":
        if isinstance(module, UNET_PIPELINE_CLS):
            module = module.unet
        elif isinstance(module, DIT_PIPELINE_CLS):
            module = module.transformer
        if isinstance(module, UNET_CLS):
            return UNetStruct.construct(module, parent=parent, fname=fname, rname=rname, rkey=rkey, idx=idx, **kwargs)
        elif isinstance(module, DIT_CLS):
            return DiTStruct.construct(module, parent=parent, fname=fname, rname=rname, rkey=rkey, idx=idx, **kwargs)
        raise NotImplementedError(f"Unsupported module type: {type(module)}")

    @classmethod
    def _get_default_key_map(cls) -> dict[str, set[str]]:
        unet_key_map = UNetStruct._get_default_key_map()
        dit_key_map = DiTStruct._get_default_key_map()
        flux_key_map = FluxStruct._get_default_key_map()
        key_map: dict[str, set[str]] = defaultdict(set)
        for rkey, keys in unet_key_map.items():
            key_map[rkey].update(keys)
        for rkey, keys in dit_key_map.items():
            key_map[rkey].update(keys)
        for rkey, keys in flux_key_map.items():
            key_map[rkey].update(keys)
        return {k: v for k, v in key_map.items() if v}

    @staticmethod
    def _simplify_keys(keys: tp.Iterable[str], *, key_map: dict[str, set[str]]) -> list[str]:
        """Simplify the keys based on the key map.

        Args:
            keys (`Iterable[str]`):
                The keys to simplify.
            key_map (`dict[str, set[str]]`):
                The key map.

        Returns:
            `list[str]`:
                The simplified keys.
        """
        # we first sort key_map by length of values in descending order
        key_map = dict(sorted(key_map.items(), key=lambda item: len(item[1]), reverse=True))
        ukeys, skeys = set(keys), set()
        for k, v in key_map.items():
            if k in ukeys:
                skeys.add(k)
                ukeys.discard(k)
                ukeys.difference_update(v)
                continue
            if ukeys.issuperset(v):
                skeys.add(k)
                ukeys.difference_update(v)
        assert not ukeys, f"Unrecognized keys: {ukeys}"
        return sorted(skeys)


class DiffusionAttentionStruct(AttentionStruct):
    module: Attention = field(repr=False, kw_only=False)
    """the module of AttentionBlock"""
    parent: tp.Optional["DiffusionTransformerBlockStruct"] = field(repr=False)

    def filter_kwargs(self, kwargs: dict) -> dict:
        """Filter layer kwargs to attn kwargs."""
        if isinstance(self.parent.module, BasicTransformerBlock):
            if kwargs.get("cross_attention_kwargs", None) is None:
                attn_kwargs = {}
            else:
                attn_kwargs = dict(kwargs["cross_attention_kwargs"].items())
            attn_kwargs.pop("gligen", None)
            if self.idx == 0:
                attn_kwargs["attention_mask"] = kwargs.get("attention_mask", None)
            else:
                attn_kwargs["attention_mask"] = kwargs.get("encoder_attention_mask", None)
        else:
            attn_kwargs = {}
        return attn_kwargs

    @staticmethod
    def _default_construct(
        module: Attention,
        /,
        parent: tp.Optional["DiffusionTransformerBlockStruct"] = None,
        fname: str = "",
        rname: str = "",
        rkey: str = "",
        idx: int = 0,
        **kwargs,
    ) -> "DiffusionAttentionStruct":
        # Initialize internal add projections (only used for WanAttention self-attention)
        internal_add_k_proj = None
        internal_add_v_proj = None
        internal_add_k_proj_rname = ""
        internal_add_v_proj_rname = ""

        # WanAttention: check is_cross_attention to determine structure
        if isinstance(module, WanAttention):
            if module.is_cross_attention:
                # Cross-attention: q from hidden_states, k/v from encoder_hidden_states
                # WanAttention cross-attention may also have add_k_proj/add_v_proj for image conditioning
                # We store add_k_proj/add_v_proj as internal projections to avoid joint attention classification
                q_proj, k_proj, v_proj = module.to_q, module.to_k, module.to_v
                add_q_proj, add_k_proj, add_v_proj = None, None, None  # Don't expose at framework level
                add_o_proj = None
                q_proj_rname, k_proj_rname, v_proj_rname = "to_q", "to_k", "to_v"
                add_q_proj_rname, add_k_proj_rname, add_v_proj_rname = "", "", ""
                add_o_proj_rname = ""
                # Store internal add projections (similar to self-attention case)
                internal_add_k_proj = getattr(module, "add_k_proj", None)
                internal_add_v_proj = getattr(module, "add_v_proj", None)
                internal_add_k_proj_rname = "add_k_proj" if internal_add_k_proj is not None else ""
                internal_add_v_proj_rname = "add_v_proj" if internal_add_v_proj is not None else ""
            else:
                # Self-attention: WanAttention may have add_k_proj/add_v_proj for image conditioning
                # These are internal implementation details but need to be quantized
                # We'll store them separately in WanAttentionStruct
                q_proj, k_proj, v_proj = module.to_q, module.to_k, module.to_v
                add_q_proj, add_k_proj, add_v_proj = None, None, None
                add_o_proj = None
                q_proj_rname, k_proj_rname, v_proj_rname = "to_q", "to_k", "to_v"
                add_q_proj_rname, add_k_proj_rname, add_v_proj_rname = "", "", ""
                add_o_proj_rname = ""
                # Store internal add projections
                internal_add_k_proj = getattr(module, "add_k_proj", None)
                internal_add_v_proj = getattr(module, "add_v_proj", None)
                internal_add_k_proj_rname = "add_k_proj" if internal_add_k_proj is not None else ""
                internal_add_v_proj_rname = "add_v_proj" if internal_add_v_proj is not None else ""
        # FluxAttention doesn't have is_cross_attention attribute
        elif isinstance(module, FluxAttention):
            # FluxAttention always has to_q, to_k, to_v
            q_proj, k_proj, v_proj = module.to_q, module.to_k, module.to_v
            add_q_proj = getattr(module, "add_q_proj", None)
            add_k_proj = getattr(module, "add_k_proj", None)
            add_v_proj = getattr(module, "add_v_proj", None)
            add_o_proj = getattr(module, "to_add_out", None)
            q_proj_rname, k_proj_rname, v_proj_rname = "to_q", "to_k", "to_v"
            add_q_proj_rname = "add_q_proj" if add_q_proj is not None else ""
            add_k_proj_rname = "add_k_proj" if add_k_proj is not None else ""
            add_v_proj_rname = "add_v_proj" if add_v_proj is not None else ""
            add_o_proj_rname = "to_add_out" if add_o_proj is not None else ""
        elif module.is_cross_attention:
            q_proj, k_proj, v_proj = module.to_q, None, None
            add_q_proj, add_k_proj, add_v_proj, add_o_proj = None, module.to_k, module.to_v, None
            q_proj_rname, k_proj_rname, v_proj_rname = "to_q", "", ""
            add_q_proj_rname, add_k_proj_rname, add_v_proj_rname, add_o_proj_rname = "", "to_k", "to_v", ""
        else:
            q_proj, k_proj, v_proj = module.to_q, module.to_k, module.to_v
            add_q_proj = getattr(module, "add_q_proj", None)
            add_k_proj = getattr(module, "add_k_proj", None)
            add_v_proj = getattr(module, "add_v_proj", None)
            add_o_proj = getattr(module, "to_add_out", None)
            q_proj_rname, k_proj_rname, v_proj_rname = "to_q", "to_k", "to_v"
            add_q_proj_rname = "add_q_proj" if add_q_proj is not None else ""
            add_k_proj_rname = "add_k_proj" if add_k_proj is not None else ""
            add_v_proj_rname = "add_v_proj" if add_v_proj is not None else ""
            add_o_proj_rname = "to_add_out" if add_o_proj is not None else ""
        if getattr(module, "to_out", None) is not None:
            o_proj = module.to_out[0]
            o_proj_rname = "to_out.0"
            assert isinstance(o_proj, nn.Linear)
        elif parent is not None:
            assert isinstance(parent.module, FluxSingleTransformerBlock)
            assert isinstance(parent.module.proj_out, ConcatLinear)
            assert len(parent.module.proj_out.linears) == 2
            o_proj = parent.module.proj_out.linears[0]
            o_proj_rname = ".proj_out.linears.0"
        else:
            raise RuntimeError("Cannot find the output projection.")
        if isinstance(module.processor, DiffusionAttentionProcessor):
            with_rope = module.processor.rope is not None
        elif module.processor.__class__.__name__.startswith("Flux"):
            with_rope = True
        else:
            with_rope = False  # TODO: fix for other processors
        # Calculate num_key_value_heads safely for different attention types
        if k_proj is not None:
            # Standard attention: calculate from k_proj
            num_key_value_heads = k_proj.weight.shape[0] // (q_proj.weight.shape[0] // module.heads)
        elif add_k_proj is not None:
            # Cross attention or Flux attention with context: calculate from add_k_proj
            num_key_value_heads = add_k_proj.weight.shape[0] // (q_proj.weight.shape[0] // module.heads)
        else:
            # Fallback: assume same as query heads
            num_key_value_heads = module.heads
        config = AttentionConfigStruct(
            hidden_size=q_proj.weight.shape[1],
            add_hidden_size=add_k_proj.weight.shape[1] if add_k_proj is not None else 0,
            inner_size=q_proj.weight.shape[0],
            num_query_heads=module.heads,
            num_key_value_heads=num_key_value_heads,
            with_qk_norm=module.norm_q is not None,
            with_rope=with_rope,
            linear_attn=isinstance(module.processor, SanaLinearAttnProcessor2_0),
        )
        # For WanAttention, always use WanAttentionStruct (to handle cross-attention correctly)
        if WanAttention is not None and isinstance(module, WanAttention):
            return WanAttentionStruct(
                module=module,
                parent=parent,
                fname=fname,
                idx=idx,
                rname=rname,
                rkey=rkey,
                config=config,
                q_proj=q_proj,
                k_proj=k_proj,
                v_proj=v_proj,
                o_proj=o_proj,
                add_q_proj=add_q_proj,
                add_k_proj=add_k_proj,
                add_v_proj=add_v_proj,
                add_o_proj=add_o_proj,
                q=None,  # TODO: add q, k, v
                k=None,
                v=None,
                q_proj_rname=q_proj_rname,
                k_proj_rname=k_proj_rname,
                v_proj_rname=v_proj_rname,
                o_proj_rname=o_proj_rname,
                add_q_proj_rname=add_q_proj_rname,
                add_k_proj_rname=add_k_proj_rname,
                add_v_proj_rname=add_v_proj_rname,
                add_o_proj_rname=add_o_proj_rname,
                q_rname="",
                k_rname="",
                v_rname="",
                internal_add_k_proj=internal_add_k_proj,
                internal_add_v_proj=internal_add_v_proj,
                internal_add_k_proj_rname=internal_add_k_proj_rname,
                internal_add_v_proj_rname=internal_add_v_proj_rname,
            )
        else:
            return DiffusionAttentionStruct(
                module=module,
                parent=parent,
                fname=fname,
                idx=idx,
                rname=rname,
                rkey=rkey,
                config=config,
                q_proj=q_proj,
                k_proj=k_proj,
                v_proj=v_proj,
                o_proj=o_proj,
                add_q_proj=add_q_proj,
                add_k_proj=add_k_proj,
                add_v_proj=add_v_proj,
                add_o_proj=add_o_proj,
                q=None,  # TODO: add q, k, v
                k=None,
                v=None,
                q_proj_rname=q_proj_rname,
                k_proj_rname=k_proj_rname,
                v_proj_rname=v_proj_rname,
                o_proj_rname=o_proj_rname,
                add_q_proj_rname=add_q_proj_rname,
                add_k_proj_rname=add_k_proj_rname,
                add_v_proj_rname=add_v_proj_rname,
                add_o_proj_rname=add_o_proj_rname,
                q_rname="",
                k_rname="",
                v_rname="",
            )


@dataclass(kw_only=True)
class WanAttentionStruct(DiffusionAttentionStruct):
    """WanAttention struct with special handling for add_k_proj/add_v_proj.

    WanAttention has different semantics from standard attention:
    - Self-attention (attn1): q/k/v from same input
    - Cross-attention (attn2):
      * q from hidden_states (video features)
      * k/v from encoder_hidden_states (text features)
      * add_k/add_v from image_encoder_hidden_states (image features)

    For cross-attention, projections should be grouped as:
    - q separately (attn_q_proj)
    - k/v together (attn_kv_proj)
    - add_k/add_v together (attn_add_kv_proj)

    Note: Inherits from DiffusionAttentionStruct (not AttentionStruct) to ensure proper
    smooth quantization handling in smooth_diffusion_layer.
    """

    # region relative keys
    q_proj_rkey: tp.ClassVar[str] = "q_proj"
    kv_proj_rkey: tp.ClassVar[str] = "kv_proj"
    add_kv_proj_rkey: tp.ClassVar[str] = "add_kv_proj"
    # endregion

    # Store the add projections that aren't part of the main joint attention structure
    internal_add_k_proj: nn.Linear | None = None
    internal_add_v_proj: nn.Linear | None = None
    internal_add_k_proj_rname: str = ""
    internal_add_v_proj_rname: str = ""

    # region absolute keys
    q_proj_key: str = field(init=False, repr=False)
    kv_proj_key: str = field(init=False, repr=False)
    add_kv_proj_key: str = field(init=False, repr=False)
    # endregion

    def __post_init__(self) -> None:
        super().__post_init__()
        # Initialize cross-attention specific keys
        self.q_proj_key = join_name(self.key, self.q_proj_rkey, sep="_")
        self.kv_proj_key = join_name(self.key, self.kv_proj_rkey, sep="_")
        self.add_kv_proj_key = join_name(self.key, self.add_kv_proj_rkey, sep="_")

    # region property overrides for WanAttention semantics

    @property
    def qkv_proj(self) -> list[nn.Linear]:
        """Override qkv_proj for WanAttention.

        For WanAttention cross-attention, projections come from different inputs and cannot
        be smoothed together. Return empty list to skip the default qkv_proj smooth.
        Individual projections will be handled through named_key_modules grouping.
        """
        if self.is_self_attn():
            # Self-attention: q/k/v together
            return [self.q_proj, self.k_proj, self.v_proj]
        else:
            # Cross-attention: return empty to skip qkv_proj smooth
            # Each group (q, kv, add_kv) has different inputs and needs separate smooth
            return []

    @property
    def add_qkv_proj(self) -> list[nn.Linear]:
        """Override add_qkv_proj for WanAttention.

        For WanAttention, add_k_proj/add_v_proj are stored as internal projections.
        """
        if self.is_self_attn():
            return []
        else:
            # Cross-attention: return internal add_k/add_v projections
            result = []
            if self.internal_add_k_proj is not None:
                result.append(self.internal_add_k_proj)
            if self.internal_add_v_proj is not None:
                result.append(self.internal_add_v_proj)
            return result

    # endregion

    def is_self_attn(self) -> bool:
        """Override for WanAttention semantics.

        WanAttention self-attention: q/k/v from same input (may have internal_add_k/v_proj)
        Standard check: add_k_proj is None AND (k_proj is not None OR is_cross_attn)

        For WanAttention:
        - has internal_add_k/v_proj -> not self-attention
        - is cross-attention (module.is_cross_attention) -> not self-attention
        """
        # If has internal add projections, it's not pure self-attention
        if self.internal_add_k_proj is not None or self.internal_add_v_proj is not None:
            return False

        # Check if this is a cross-attention based on the original module
        # We stored this information during construction
        if hasattr(self.module, 'is_cross_attention') and self.module.is_cross_attention:
            return False

        # Otherwise use the standard check
        return super().is_self_attn()

    def is_cross_attn(self) -> bool:
        """Override for WanAttention semantics.

        For WanAttention, cross-attention is identified by:
        1. Has k_proj (standard cross-attention has k_proj)
        2. Does NOT have add projections (neither internal_add_k_proj nor add_k_proj)
        3. Is marked as cross-attention on the module

        This handles the case where WanAttention is used for standard cross-attention
        without additional image conditioning.
        """
        # Check if module is marked as cross-attention
        if hasattr(self.module, 'is_cross_attention') and self.module.is_cross_attention:
            # Has k_proj but no add projections -> pure cross-attention
            has_k_proj = self.k_proj is not None
            has_no_add_proj = (self.internal_add_k_proj is None) and (self.add_k_proj is None)
            if has_k_proj and has_no_add_proj:
                return True

        # Otherwise use the standard check (k_proj is None)
        return super().is_cross_attn()

    def is_joint_attn(self) -> bool:
        """Override for WanAttention semantics.

        For WanAttention, we check both internal_add_k_proj (WanAttention-specific)
        and add_k_proj (standard). A WanAttention is considered joint attention
        if it has k_proj and either internal_add_k_proj or add_k_proj.

        Standard joint attention: has both k_proj and add_k_proj
        WanAttention joint attention: has both k_proj and (internal_add_k_proj or add_k_proj)
        """
        # Check for add projections (either WanAttention-specific or standard)
        has_add_proj = (self.internal_add_k_proj is not None) or (self.add_k_proj is not None)
        has_k_proj = self.k_proj is not None

        return has_add_proj and has_k_proj

    def named_key_modules(self) -> tp.Generator[tp.Tuple[str, str, nn.Module, "BaseModuleStruct", str], None, None]:
        """Yield modules with proper grouping keys for WanAttention.

        For self-attention: q/k/v grouped as qkv_proj
        For cross-attention:
          - q separately as q_proj
          - k/v together as kv_proj
          - add_k/add_v together as add_kv_proj
        """
        if self.is_self_attn():
            # Self-attention: use standard qkv grouping
            yield from super().named_key_modules()
        else:
            # Cross-attention: separate grouping
            # Q projection separately
            yield self.q_proj_key, self.q_proj_name, self.q_proj, self, "q_proj"

            # K/V projections together
            if self.k_proj is not None:
                yield self.kv_proj_key, self.k_proj_name, self.k_proj, self, "k_proj"
            if self.v_proj is not None:
                yield self.kv_proj_key, self.v_proj_name, self.v_proj, self, "v_proj"

            # Add K/V projections together (for image conditioning)
            if self.internal_add_k_proj is not None:
                internal_add_k_name = join_name(self.name, self.internal_add_k_proj_rname)
                yield self.add_kv_proj_key, internal_add_k_name, self.internal_add_k_proj, self, "internal_add_k_proj"
            if self.internal_add_v_proj is not None:
                internal_add_v_name = join_name(self.name, self.internal_add_v_proj_rname)
                yield self.add_kv_proj_key, internal_add_v_name, self.internal_add_v_proj, self, "internal_add_v_proj"

            # Output projection
            yield self.out_proj_key, self.o_proj_name, self.o_proj, self, "o_proj"


@dataclass(kw_only=True)
class DiffusionFeedForwardStruct(FeedForwardStruct):
    module: FeedForward = field(repr=False, kw_only=False)
    """the module of FeedForward"""
    parent: tp.Optional["DiffusionTransformerBlockStruct"] = field(repr=False)
    # region modules
    moe_gate: None = field(init=False, repr=False, default=None)
    experts: list[nn.Module] = field(init=False, repr=False)
    # endregion
    # region names
    moe_gate_rname: str = field(init=False, repr=False, default="")
    experts_rname: str = field(init=False, repr=False, default="")
    # endregion

    # region aliases

    @property
    def up_proj(self) -> nn.Linear:
        return self.up_projs[0]

    @property
    def down_proj(self) -> nn.Linear:
        return self.down_projs[0]

    @property
    def up_proj_rname(self) -> str:
        return self.up_proj_rnames[0]

    @property
    def down_proj_rname(self) -> str:
        return self.down_proj_rnames[0]

    @property
    def up_proj_name(self) -> str:
        return self.up_proj_names[0]

    @property
    def down_proj_name(self) -> str:
        return self.down_proj_names[0]

    # endregion

    def __post_init__(self) -> None:
        assert len(self.up_projs) == len(self.down_projs) == 1
        assert len(self.up_proj_rnames) == len(self.down_proj_rnames) == 1
        self.experts = [self.module]
        super().__post_init__()

    @staticmethod
    def _default_construct(
        module: FeedForward | FluxSingleTransformerBlock | GLUMBConv,
        /,
        parent: tp.Optional["DiffusionTransformerBlockStruct"] = None,
        fname: str = "",
        rname: str = "",
        rkey: str = "",
        idx: int = 0,
        **kwargs,
    ) -> "DiffusionFeedForwardStruct":
        if isinstance(module, FeedForward):
            layer_1, layer_2 = module.net[0], module.net[2]
            assert isinstance(layer_1, (GEGLU, GELU, ApproximateGELU, SwiGLU))
            up_proj, up_proj_rname = layer_1.proj, "net.0.proj"
            assert isinstance(up_proj, nn.Linear)
            down_proj, down_proj_rname = layer_2, "net.2"
            if isinstance(layer_1, GEGLU):
                act_type = "gelu_glu"
            elif isinstance(layer_1, SwiGLU):
                act_type = "swish_glu"
            else:
                assert layer_1.__class__.__name__.lower().endswith("gelu")
                act_type = "gelu"
                if isinstance(layer_2, ShiftedLinear):
                    down_proj, down_proj_rname = layer_2.linear, "net.2.linear"
                    act_type = "gelu_shifted"
            assert isinstance(down_proj, nn.Linear)
            ffn = module
        elif isinstance(module, FluxSingleTransformerBlock):
            up_proj, up_proj_rname = module.proj_mlp, "proj_mlp"
            act_type = "gelu"
            assert isinstance(module.proj_out, ConcatLinear)
            assert len(module.proj_out.linears) == 2
            layer_2 = module.proj_out.linears[1]
            if isinstance(layer_2, ShiftedLinear):
                down_proj, down_proj_rname = layer_2.linear, "proj_out.linears.1.linear"
                act_type = "gelu_shifted"
            else:
                down_proj, down_proj_rname = layer_2, "proj_out.linears.1"
            ffn = nn.Sequential(up_proj, module.act_mlp, layer_2)
            assert not rname, f"Unsupported rname: {rname}"
        elif isinstance(module, GLUMBConv):
            ffn = module
            up_proj, up_proj_rname = module.conv_inverted, "conv_inverted"
            down_proj, down_proj_rname = module.conv_point, "conv_point"
            act_type = "silu_conv_silu_glu"
        else:
            raise NotImplementedError(f"Unsupported module type: {type(module)}")
        config = FeedForwardConfigStruct(
            hidden_size=up_proj.weight.shape[1],
            intermediate_size=down_proj.weight.shape[1],
            intermediate_act_type=act_type,
            num_experts=1,
        )
        return DiffusionFeedForwardStruct(
            module=ffn,  # this may be a virtual module
            parent=parent,
            fname=fname,
            idx=idx,
            rname=rname,
            rkey=rkey,
            config=config,
            up_projs=[up_proj],
            down_projs=[down_proj],
            up_proj_rnames=[up_proj_rname],
            down_proj_rnames=[down_proj_rname],
        )


@dataclass(kw_only=True)
class DiffusionTransformerBlockStruct(TransformerBlockStruct, DiffusionBlockStruct):
    # region relative keys
    norm_rkey: tp.ClassVar[str] = "transformer_norm"
    add_norm_rkey: tp.ClassVar[str] = "transformer_add_norm"
    mod_rkey: tp.ClassVar[str] = "transformer_mod"
    """Modulation layers (e.g., img_mod, txt_mod in QwenImage)"""
    attn_struct_cls: tp.ClassVar[type[DiffusionAttentionStruct]] = DiffusionAttentionStruct
    ffn_struct_cls: tp.ClassVar[type[DiffusionFeedForwardStruct]] = DiffusionFeedForwardStruct
    # endregion

    parent: tp.Optional["DiffusionTransformerStruct"] = field(repr=False)
    # region child modules
    post_attn_norms: list[nn.LayerNorm] = field(init=False, repr=False, default_factory=list)
    post_attn_add_norms: list[nn.LayerNorm] = field(init=False, repr=False, default_factory=list)
    post_ffn_norm: None = field(init=False, repr=False, default=None)
    post_add_ffn_norm: None = field(init=False, repr=False, default=None)
    mod: nn.Module | None = None
    """Modulation layer (e.g., img_mod in QwenImage)"""
    add_mod: nn.Module | None = None
    """Additional modulation layer (e.g., txt_mod in QwenImage)"""
    # endregion
    # region relative names
    post_attn_norm_rnames: list[str] = field(init=False, repr=False, default_factory=list)
    post_attn_add_norm_rnames: list[str] = field(init=False, repr=False, default_factory=list)
    post_ffn_norm_rname: str = field(init=False, repr=False, default="")
    post_add_ffn_norm_rname: str = field(init=False, repr=False, default="")
    mod_rname: str = ""
    add_mod_rname: str = ""
    # endregion
    # region attributes
    norm_type: str
    add_norm_type: str
    # endregion
    # region absolute names
    mod_name: str = field(init=False, repr=False, default="")
    add_mod_name: str = field(init=False, repr=False, default="")
    # endregion
    # region absolute keys
    norm_key: str = field(init=False, repr=False)
    add_norm_key: str = field(init=False, repr=False)
    mod_key: str = field(init=False, repr=False, default="")
    # endregion
    # region child structs
    pre_attn_norm_structs: list[DiffusionModuleStruct | None] = field(init=False, repr=False)
    pre_attn_add_norm_structs: list[DiffusionModuleStruct | None] = field(init=False, repr=False)
    pre_ffn_norm_struct: DiffusionModuleStruct = field(init=False, repr=False, default=None)
    pre_add_ffn_norm_struct: DiffusionModuleStruct | None = field(init=False, repr=False, default=None)
    attn_structs: list[DiffusionAttentionStruct] = field(init=False, repr=False)
    ffn_struct: DiffusionFeedForwardStruct | None = field(init=False, repr=False)
    add_ffn_struct: DiffusionFeedForwardStruct | None = field(init=False, repr=False)
    mod_struct: DiffusionModuleStruct | None = field(init=False, repr=False, default=None)
    add_mod_struct: DiffusionModuleStruct | None = field(init=False, repr=False, default=None)
    # endregion

    def __post_init__(self) -> None:
        super().__post_init__()
        self.norm_key = join_name(self.key, self.norm_rkey, sep="_")
        self.add_norm_key = join_name(self.key, self.add_norm_rkey, sep="_")
        self.attn_norm_structs = [
            DiffusionModuleStruct(norm, parent=self, fname="pre_attn_norm", rname=rname, rkey=self.norm_rkey, idx=idx)
            for idx, (norm, rname) in enumerate(zip(self.pre_attn_norms, self.pre_attn_norm_rnames, strict=True))
        ]
        self.add_attn_norm_structs = [
            DiffusionModuleStruct(
                norm, parent=self, fname="pre_attn_add_norm", rname=rname, rkey=self.add_norm_rkey, idx=idx
            )
            for idx, (norm, rname) in enumerate(
                zip(self.pre_attn_add_norms, self.pre_attn_add_norm_rnames, strict=True)
            )
        ]
        if self.pre_ffn_norm is not None:
            self.pre_ffn_norm_struct = DiffusionModuleStruct(
                self.pre_ffn_norm, parent=self, fname="pre_ffn_norm", rname=self.pre_ffn_norm_rname, rkey=self.norm_rkey
            )
        if self.pre_add_ffn_norm is not None:
            self.pre_add_ffn_norm_struct = DiffusionModuleStruct(
                self.pre_add_ffn_norm,
                parent=self,
                fname="pre_add_ffn_norm",
                rname=self.pre_add_ffn_norm_rname,
                rkey=self.add_norm_rkey,
            )
        # Initialize mod structs (e.g., for QwenImage)
        if self.mod is not None:
            self.mod_key = join_name(self.key, self.mod_rkey, sep="_")
            self.mod_name = join_name(self.name, self.mod_rname)
            self.mod_struct = DiffusionModuleStruct(
                self.mod, parent=self, fname="mod", rname=self.mod_rname, rkey=self.mod_rkey
            )
        if self.add_mod is not None:
            self.add_mod_name = join_name(self.name, self.add_mod_rname)
            self.add_mod_struct = DiffusionModuleStruct(
                self.add_mod, parent=self, fname="add_mod", rname=self.add_mod_rname, rkey=self.mod_rkey
            )

    def named_key_modules(self) -> tp.Generator[tp.Tuple[str, str, nn.Module, BaseModuleStruct, str], None, None]:
        # Yield mod layers first (e.g., img_mod, txt_mod in QwenImage)
        if self.mod_struct is not None:
            yield from self.mod_struct.named_key_modules()
        if self.add_mod_struct is not None:
            yield from self.add_mod_struct.named_key_modules()
        for attn_norm in self.attn_norm_structs:
            if attn_norm.module is not None:
                yield from attn_norm.named_key_modules()
        for add_attn_norm in self.add_attn_norm_structs:
            if add_attn_norm.module is not None:
                yield from add_attn_norm.named_key_modules()
        for attn_struct in self.attn_structs:
            yield from attn_struct.named_key_modules()
        if self.pre_ffn_norm_struct is not None:
            if self.pre_attn_norms and self.pre_attn_norms[0] is not self.pre_ffn_norm:
                yield from self.pre_ffn_norm_struct.named_key_modules()
        if self.ffn_struct is not None:
            yield from self.ffn_struct.named_key_modules()
        if self.pre_add_ffn_norm_struct is not None:
            if self.pre_attn_add_norms and self.pre_attn_add_norms[0] is not self.pre_add_ffn_norm:
                yield from self.pre_add_ffn_norm_struct.named_key_modules()
        if self.add_ffn_struct is not None:
            yield from self.add_ffn_struct.named_key_modules()

    @staticmethod
    def _default_construct(
        module: DIT_BLOCK_CLS,
        /,
        parent: tp.Optional["DiffusionTransformerStruct"] = None,
        fname: str = "",
        rname: str = "",
        rkey: str = "",
        idx: int = 0,
        **kwargs,
    ) -> "DiffusionTransformerBlockStruct":
        if isinstance(module, (BasicTransformerBlock, SanaTransformerBlock)):
            parallel = False
            if isinstance(module, SanaTransformerBlock):
                norm_type = add_norm_type = "ada_norm_single"
            else:
                norm_type = add_norm_type = module.norm_type
            pre_attn_norms, pre_attn_norm_rnames = [], []
            attns, attn_rnames = [], []
            pre_attn_add_norms, pre_attn_add_norm_rnames = [], []
            assert module.norm1 is not None
            assert module.attn1 is not None
            pre_attn_norms.append(module.norm1)
            pre_attn_norm_rnames.append("norm1")
            attns.append(module.attn1)
            attn_rnames.append("attn1")
            pre_attn_add_norms.append(module.attn1.norm_cross)
            pre_attn_add_norm_rnames.append("attn1.norm_cross")
            if module.attn2 is not None:
                if norm_type == "ada_norm_single":
                    pre_attn_norms.append(None)
                    pre_attn_norm_rnames.append("")
                else:
                    assert module.norm2 is not None
                    pre_attn_norms.append(module.norm2)
                    pre_attn_norm_rnames.append("norm2")
                attns.append(module.attn2)
                attn_rnames.append("attn2")
                pre_attn_add_norms.append(module.attn2.norm_cross)
                pre_attn_add_norm_rnames.append("attn2.norm_cross")
            if norm_type == "ada_norm_single":
                assert module.norm2 is not None
                pre_ffn_norm, pre_ffn_norm_rname = module.norm2, "norm2"
            else:
                pre_ffn_norm, pre_ffn_norm_rname = module.norm3, "" if module.norm3 is None else "norm3"
            ffn, ffn_rname = module.ff, "" if module.ff is None else "ff"
            pre_add_ffn_norm, pre_add_ffn_norm_rname, add_ffn, add_ffn_rname = None, "", None, ""
        elif isinstance(module, JointTransformerBlock):
            parallel = False
            norm_type = "ada_norm_zero"
            pre_attn_norms, pre_attn_norm_rnames = [module.norm1], ["norm1"]
            if isinstance(module.norm1_context, AdaLayerNormZero):
                add_norm_type = "ada_norm_zero"
            else:
                add_norm_type = "ada_norm_continous"
            pre_attn_add_norms, pre_attn_add_norm_rnames = [module.norm1_context], ["norm1_context"]
            attns, attn_rnames = [module.attn], ["attn"]
            pre_ffn_norm, pre_ffn_norm_rname = module.norm2, "norm2"
            ffn, ffn_rname = module.ff, "ff"
            pre_add_ffn_norm, pre_add_ffn_norm_rname = module.norm2_context, "norm2_context"
            add_ffn, add_ffn_rname = module.ff_context, "ff_context"
        elif isinstance(module, FluxSingleTransformerBlock):
            parallel = True
            norm_type = add_norm_type = "ada_norm_zero_single"
            pre_attn_norms, pre_attn_norm_rnames = [module.norm], ["norm"]
            attns, attn_rnames = [module.attn], ["attn"]
            pre_attn_add_norms, pre_attn_add_norm_rnames = [], []
            pre_ffn_norm, pre_ffn_norm_rname = module.norm, "norm"
            ffn, ffn_rname = module, ""
            pre_add_ffn_norm, pre_add_ffn_norm_rname, add_ffn, add_ffn_rname = None, "", None, ""
        elif isinstance(module, FluxTransformerBlock):
            parallel = False
            norm_type = add_norm_type = "ada_norm_zero"
            pre_attn_norms, pre_attn_norm_rnames = [module.norm1], ["norm1"]
            attns, attn_rnames = [module.attn], ["attn"]
            pre_attn_add_norms, pre_attn_add_norm_rnames = [module.norm1_context], ["norm1_context"]
            pre_ffn_norm, pre_ffn_norm_rname = module.norm2, "norm2"
            ffn, ffn_rname = module.ff, "ff"
            pre_add_ffn_norm, pre_add_ffn_norm_rname = module.norm2_context, "norm2_context"
            add_ffn, add_ffn_rname = module.ff_context, "ff_context"
        elif isinstance(module, QwenImageTransformerBlock):
            parallel = False
            norm_type = add_norm_type = "layer_norm"
            pre_attn_norms, pre_attn_norm_rnames = [module.img_norm1], ["img_norm1"]
            attns, attn_rnames = [module.attn], ["attn"]
            pre_attn_add_norms, pre_attn_add_norm_rnames = [module.txt_norm1], ["txt_norm1"]
            pre_ffn_norm, pre_ffn_norm_rname = module.img_norm2, "img_norm2"
            ffn, ffn_rname = module.img_mlp, "img_mlp"
            pre_add_ffn_norm, pre_add_ffn_norm_rname = module.txt_norm2, "txt_norm2"
            add_ffn, add_ffn_rname = module.txt_mlp, "txt_mlp"
            mod, mod_rname = module.img_mod, "img_mod"
            add_mod, add_mod_rname = module.txt_mod, "txt_mod"
        elif isinstance(module, WanTransformerBlock):
            # Wan block structure:
            # - norm1 + attn1 (self-attention)
            # - attn2 (cross-attention) with optional norm2 (may be Identity)
            # - norm3 + ffn
            parallel = False
            norm_type = add_norm_type = "layer_norm"
            # Both attentions go into attns list
            pre_attn_norms = [module.norm1]
            pre_attn_norm_rnames = ["norm1"]
            attns = [module.attn1]
            attn_rnames = ["attn1"]
            pre_attn_add_norms = [None]  # attn1 is self-attn, no add_norm
            pre_attn_add_norm_rnames = [""]

            if module.attn2 is not None:
                # attn2 has no pre_norm (uses residual from attn1)
                pre_attn_norms.append(None)
                pre_attn_norm_rnames.append("")
                attns.append(module.attn2)
                attn_rnames.append("attn2")
                # norm2 is the add_norm for attn2 (may be Identity)
                if not isinstance(module.norm2, nn.Identity):
                    pre_attn_add_norms.append(module.norm2)
                    pre_attn_add_norm_rnames.append("norm2")
                else:
                    pre_attn_add_norms.append(None)
                    pre_attn_add_norm_rnames.append("")

            # FFN
            pre_ffn_norm, pre_ffn_norm_rname = module.norm3, "norm3"
            ffn, ffn_rname = module.ffn, "ffn"
            pre_add_ffn_norm, pre_add_ffn_norm_rname, add_ffn, add_ffn_rname = None, "", None, ""
        else:
            raise NotImplementedError(f"Unsupported module type: {type(module)}")
        # mod is optional, only QwenImage has it
        mod = locals().get("mod", None)
        mod_rname = locals().get("mod_rname", "")
        add_mod = locals().get("add_mod", None)
        add_mod_rname = locals().get("add_mod_rname", "")
        return DiffusionTransformerBlockStruct(
            module=module,
            parent=parent,
            fname=fname,
            idx=idx,
            rname=rname,
            rkey=rkey,
            parallel=parallel,
            pre_attn_norms=pre_attn_norms,
            pre_attn_add_norms=pre_attn_add_norms,
            attns=attns,
            pre_ffn_norm=pre_ffn_norm,
            ffn=ffn,
            pre_add_ffn_norm=pre_add_ffn_norm,
            add_ffn=add_ffn,
            mod=mod,
            add_mod=add_mod,
            pre_attn_norm_rnames=pre_attn_norm_rnames,
            pre_attn_add_norm_rnames=pre_attn_add_norm_rnames,
            attn_rnames=attn_rnames,
            pre_ffn_norm_rname=pre_ffn_norm_rname,
            ffn_rname=ffn_rname,
            pre_add_ffn_norm_rname=pre_add_ffn_norm_rname,
            add_ffn_rname=add_ffn_rname,
            mod_rname=mod_rname,
            add_mod_rname=add_mod_rname,
            norm_type=norm_type,
            add_norm_type=add_norm_type,
        )

    @classmethod
    def _get_default_key_map(cls) -> dict[str, set[str]]:
        """Get the default allowed keys."""
        key_map: dict[str, set[str]] = defaultdict(set)
        norm_rkey = norm_key = cls.norm_rkey
        add_norm_rkey = add_norm_key = cls.add_norm_rkey
        key_map[norm_rkey].add(norm_key)
        key_map[add_norm_rkey].add(add_norm_key)
        attn_cls = cls.attn_struct_cls
        attn_key = attn_rkey = cls.attn_rkey
        qkv_proj_key = qkv_proj_rkey = join_name(attn_key, attn_cls.qkv_proj_rkey, sep="_")
        out_proj_key = out_proj_rkey = join_name(attn_key, attn_cls.out_proj_rkey, sep="_")
        add_qkv_proj_key = add_qkv_proj_rkey = join_name(attn_key, attn_cls.add_qkv_proj_rkey, sep="_")
        add_out_proj_key = add_out_proj_rkey = join_name(attn_key, attn_cls.add_out_proj_rkey, sep="_")
        key_map[attn_rkey].add(qkv_proj_key)
        key_map[attn_rkey].add(out_proj_key)
        if attn_cls.add_qkv_proj_rkey.startswith("add_") and attn_cls.add_out_proj_rkey.startswith("add_"):
            add_attn_rkey = join_name(attn_rkey, "add", sep="_")
            key_map[add_attn_rkey].add(add_qkv_proj_key)
            key_map[add_attn_rkey].add(add_out_proj_key)
        key_map[qkv_proj_rkey].add(qkv_proj_key)
        key_map[out_proj_rkey].add(out_proj_key)
        key_map[add_qkv_proj_rkey].add(add_qkv_proj_key)
        key_map[add_out_proj_rkey].add(add_out_proj_key)
        ffn_cls = cls.ffn_struct_cls
        ffn_key = ffn_rkey = cls.ffn_rkey
        add_ffn_key = add_ffn_rkey = cls.add_ffn_rkey
        up_proj_key = up_proj_rkey = join_name(ffn_key, ffn_cls.up_proj_rkey, sep="_")
        down_proj_key = down_proj_rkey = join_name(ffn_key, ffn_cls.down_proj_rkey, sep="_")
        add_up_proj_key = add_up_proj_rkey = join_name(add_ffn_key, ffn_cls.up_proj_rkey, sep="_")
        add_down_proj_key = add_down_proj_rkey = join_name(add_ffn_key, ffn_cls.down_proj_rkey, sep="_")
        key_map[ffn_rkey].add(up_proj_key)
        key_map[ffn_rkey].add(down_proj_key)
        key_map[add_ffn_rkey].add(add_up_proj_key)
        key_map[add_ffn_rkey].add(add_down_proj_key)
        key_map[up_proj_rkey].add(up_proj_key)
        key_map[down_proj_rkey].add(down_proj_key)
        key_map[add_up_proj_rkey].add(add_up_proj_key)
        key_map[add_down_proj_rkey].add(add_down_proj_key)
        # Add mod key for modulation layers (e.g., img_mod, txt_mod in QwenImage)
        mod_rkey = mod_key = cls.mod_rkey
        key_map[mod_rkey].add(mod_key)
        return {k: v for k, v in key_map.items() if v}


@dataclass(kw_only=True)
class DiffusionTransformerStruct(BaseTransformerStruct, DiffusionBlockStruct):
    # region relative keys
    proj_in_rkey: tp.ClassVar[str] = "transformer_proj_in"
    proj_out_rkey: tp.ClassVar[str] = "transformer_proj_out"
    transformer_block_rkey: tp.ClassVar[str] = ""
    transformer_block_struct_cls: tp.ClassVar[type[DiffusionTransformerBlockStruct]] = DiffusionTransformerBlockStruct
    # endregion

    module: Transformer2DModel = field(repr=False, kw_only=False)
    # region modules
    norm_in: nn.GroupNorm | None
    """Input normalization"""
    proj_in: nn.Linear | nn.Conv2d
    """Input projection"""
    norm_out: nn.GroupNorm | None
    """Output normalization"""
    proj_out: nn.Linear | nn.Conv2d
    """Output projection"""
    transformer_blocks: nn.ModuleList = field(repr=False)
    """Transformer blocks"""
    # endregion
    # region relative names
    transformer_blocks_rname: str
    # endregion
    # region absolute names
    transformer_blocks_name: str = field(init=False, repr=False)
    transformer_block_names: list[str] = field(init=False, repr=False)
    # endregion
    # region child structs
    transformer_block_structs: list[DiffusionTransformerBlockStruct] = field(init=False, repr=False)
    # endregion

    # region aliases

    @property
    def num_blocks(self) -> int:
        return len(self.transformer_blocks)

    @property
    def block_structs(self) -> list[DiffusionBlockStruct]:
        return self.transformer_block_structs

    @property
    def block_names(self) -> list[str]:
        return self.transformer_block_names

    # endregion

    def __post_init__(self):
        super().__post_init__()
        transformer_block_rnames = [
            f"{self.transformer_blocks_rname}.{idx}" for idx in range(len(self.transformer_blocks))
        ]
        self.transformer_blocks_name = join_name(self.name, self.transformer_blocks_rname)
        self.transformer_block_names = [join_name(self.name, rname) for rname in transformer_block_rnames]
        self.transformer_block_structs = [
            self.transformer_block_struct_cls.construct(
                layer,
                parent=self,
                fname="transformer_block",
                rname=rname,
                rkey=self.transformer_block_rkey,
                idx=idx,
            )
            for idx, (layer, rname) in enumerate(zip(self.transformer_blocks, transformer_block_rnames, strict=True))
        ]

    @staticmethod
    def _default_construct(
        module: Transformer2DModel,
        /,
        parent: BaseModuleStruct = None,
        fname: str = "",
        rname: str = "",
        rkey: str = "",
        idx: int = 0,
        **kwargs,
    ) -> "DiffusionTransformerStruct":
        if isinstance(module, Transformer2DModel):
            assert module.is_input_continuous, "input must be continuous"
            transformer_blocks, transformer_blocks_rname = module.transformer_blocks, "transformer_blocks"
            norm_in, norm_in_rname = module.norm, "norm"
            proj_in, proj_in_rname = module.proj_in, "proj_in"
            proj_out, proj_out_rname = module.proj_out, "proj_out"
            norm_out, norm_out_rname = None, ""
        else:
            raise NotImplementedError(f"Unsupported module type: {type(module)}")
        return DiffusionTransformerStruct(
            module=module,
            parent=parent,
            fname=fname,
            idx=idx,
            rname=rname,
            rkey=rkey,
            norm_in=norm_in,
            proj_in=proj_in,
            transformer_blocks=transformer_blocks,
            proj_out=proj_out,
            norm_out=norm_out,
            norm_in_rname=norm_in_rname,
            proj_in_rname=proj_in_rname,
            transformer_blocks_rname=transformer_blocks_rname,
            norm_out_rname=norm_out_rname,
            proj_out_rname=proj_out_rname,
        )

    @classmethod
    def _get_default_key_map(cls) -> dict[str, set[str]]:
        """Get the default allowed keys."""
        key_map: dict[str, set[str]] = defaultdict(set)
        proj_in_rkey = proj_in_key = cls.proj_in_rkey
        proj_out_rkey = proj_out_key = cls.proj_out_rkey
        key_map[proj_in_rkey].add(proj_in_key)
        key_map[proj_out_rkey].add(proj_out_key)
        block_cls = cls.transformer_block_struct_cls
        block_key = block_rkey = cls.transformer_block_rkey
        block_key_map = block_cls._get_default_key_map()
        for rkey, keys in block_key_map.items():
            rkey = join_name(block_rkey, rkey, sep="_")
            for key in keys:
                key = join_name(block_key, key, sep="_")
                key_map[rkey].add(key)
        return {k: v for k, v in key_map.items() if v}


@dataclass(kw_only=True)
class DiffusionResnetStruct(BaseModuleStruct):
    # region relative keys
    conv_rkey: tp.ClassVar[str] = "conv"
    shortcut_rkey: tp.ClassVar[str] = "shortcut"
    time_proj_rkey: tp.ClassVar[str] = "time_proj"
    # endregion

    module: ResnetBlock2D = field(repr=False, kw_only=False)
    """the module of Resnet"""
    config: FeedForwardConfigStruct
    # region child modules
    norms: list[nn.GroupNorm]
    convs: list[list[nn.Conv2d]]
    shortcut: nn.Conv2d | None
    time_proj: nn.Linear | None
    # endregion
    # region relative names
    norm_rnames: list[str]
    conv_rnames: list[list[str]]
    shortcut_rname: str
    time_proj_rname: str
    # endregion
    # region absolute names
    norm_names: list[str] = field(init=False, repr=False)
    conv_names: list[list[str]] = field(init=False, repr=False)
    shortcut_name: str = field(init=False, repr=False)
    time_proj_name: str = field(init=False, repr=False)
    # endregion
    # region absolute keys
    conv_key: str = field(init=False, repr=False)
    shortcut_key: str = field(init=False, repr=False)
    time_proj_key: str = field(init=False, repr=False)
    # endregion

    def __post_init__(self):
        super().__post_init__()
        self.norm_names = [join_name(self.name, rname) for rname in self.norm_rnames]
        self.conv_names = [[join_name(self.name, rname) for rname in rnames] for rnames in self.conv_rnames]
        self.shortcut_name = join_name(self.name, self.shortcut_rname)
        self.time_proj_name = join_name(self.name, self.time_proj_rname)
        self.conv_key = join_name(self.key, self.conv_rkey, sep="_")
        self.shortcut_key = join_name(self.key, self.shortcut_rkey, sep="_")
        self.time_proj_key = join_name(self.key, self.time_proj_rkey, sep="_")

    def named_key_modules(self) -> tp.Generator[tp.Tuple[str, str, nn.Module, BaseModuleStruct, str], None, None]:
        for convs, names in zip(self.convs, self.conv_names, strict=True):
            for conv, name in zip(convs, names, strict=True):
                yield self.conv_key, name, conv, self, "conv"
        if self.shortcut is not None:
            yield self.shortcut_key, self.shortcut_name, self.shortcut, self, "shortcut"
        if self.time_proj is not None:
            yield self.time_proj_key, self.time_proj_name, self.time_proj, self, "time_proj"

    @staticmethod
    def construct(
        module: ResnetBlock2D,
        /,
        parent: BaseModuleStruct = None,
        fname: str = "",
        rname: str = "",
        rkey: str = "",
        idx: int = 0,
        **kwargs,
    ) -> "DiffusionResnetStruct":
        if isinstance(module, ResnetBlock2D):
            assert module.upsample is None, "upsample must be None"
            assert module.downsample is None, "downsample must be None"
            act_type = module.nonlinearity.__class__.__name__.lower()
            shifted = False
            if isinstance(module.conv1, ConcatConv2d):
                conv1_convs, conv1_names = [], []
                for conv_idx, conv in enumerate(module.conv1.convs):
                    if isinstance(conv, ShiftedConv2d):
                        shifted = True
                        conv1_convs.append(conv.conv)
                        conv1_names.append(f"conv1.convs.{conv_idx}.conv")
                    else:
                        assert isinstance(conv, nn.Conv2d)
                        conv1_convs.append(conv)
                        conv1_names.append(f"conv1.convs.{conv_idx}")
            elif isinstance(module.conv1, ShiftedConv2d):
                shifted = True
                conv1_convs = [module.conv1.conv]
                conv1_names = ["conv1.conv"]
            else:
                assert isinstance(module.conv1, nn.Conv2d)
                conv1_convs, conv1_names = [module.conv1], ["conv1"]
            if isinstance(module.conv2, ConcatConv2d):
                conv2_convs, conv2_names = [], []
                for conv_idx, conv in enumerate(module.conv2.convs):
                    if isinstance(conv, ShiftedConv2d):
                        shifted = True
                        conv2_convs.append(conv.conv)
                        conv2_names.append(f"conv2.convs.{conv_idx}.conv")
                    else:
                        assert isinstance(conv, nn.Conv2d)
                        conv2_convs.append(conv)
                        conv2_names.append(f"conv2.convs.{conv_idx}")
            elif isinstance(module.conv2, ShiftedConv2d):
                shifted = True
                conv2_convs = [module.conv2.conv]
                conv2_names = ["conv2.conv"]
            else:
                assert isinstance(module.conv2, nn.Conv2d)
                conv2_convs, conv2_names = [module.conv2], ["conv2"]
            convs, conv_rnames = [conv1_convs, conv2_convs], [conv1_names, conv2_names]
            norms, norm_rnames = [module.norm1, module.norm2], ["norm1", "norm2"]
            shortcut, shortcut_rname = module.conv_shortcut, "" if module.conv_shortcut is None else "conv_shortcut"
            time_proj, time_proj_rname = module.time_emb_proj, "" if module.time_emb_proj is None else "time_emb_proj"
            if shifted:
                assert all(hasattr(conv, "shifted") and conv.shifted for level_convs in convs for conv in level_convs)
                act_type += "_shifted"
        else:
            raise NotImplementedError(f"Unsupported module type: {type(module)}")
        config = FeedForwardConfigStruct(
            hidden_size=convs[0][0].weight.shape[1],
            intermediate_size=convs[0][0].weight.shape[0],
            intermediate_act_type=act_type,
            num_experts=1,
        )
        return DiffusionResnetStruct(
            module=module,
            parent=parent,
            fname=fname,
            idx=idx,
            rname=rname,
            rkey=rkey,
            config=config,
            norms=norms,
            convs=convs,
            shortcut=shortcut,
            time_proj=time_proj,
            norm_rnames=norm_rnames,
            conv_rnames=conv_rnames,
            shortcut_rname=shortcut_rname,
            time_proj_rname=time_proj_rname,
        )

    @classmethod
    def _get_default_key_map(cls) -> dict[str, set[str]]:
        """Get the default allowed keys."""
        key_map: dict[str, set[str]] = defaultdict(set)
        conv_key = conv_rkey = cls.conv_rkey
        shortcut_key = shortcut_rkey = cls.shortcut_rkey
        time_proj_key = time_proj_rkey = cls.time_proj_rkey
        key_map[conv_rkey].add(conv_key)
        key_map[shortcut_rkey].add(shortcut_key)
        key_map[time_proj_rkey].add(time_proj_key)
        return {k: v for k, v in key_map.items() if v}


@dataclass(kw_only=True)
class UNetBlockStruct(DiffusionBlockStruct):
    class BlockType(enum.StrEnum):
        DOWN = "down"
        MID = "mid"
        UP = "up"

    # region relative keys
    resnet_rkey: tp.ClassVar[str] = "resblock"
    sampler_rkey: tp.ClassVar[str] = "sample"
    transformer_rkey: tp.ClassVar[str] = ""
    resnet_struct_cls: tp.ClassVar[type[DiffusionResnetStruct]] = DiffusionResnetStruct
    transformer_struct_cls: tp.ClassVar[type[DiffusionTransformerStruct]] = DiffusionTransformerStruct
    # endregion

    parent: tp.Optional["UNetStruct"] = field(repr=False)
    # region attributes
    block_type: BlockType
    # endregion
    # region modules
    resnets: nn.ModuleList = field(repr=False)
    transformers: nn.ModuleList = field(repr=False)
    sampler: nn.Conv2d | None
    # endregion
    # region relative names
    resnets_rname: str
    transformers_rname: str
    sampler_rname: str
    # endregion
    # region absolute names
    resnets_name: str = field(init=False, repr=False)
    transformers_name: str = field(init=False, repr=False)
    sampler_name: str = field(init=False, repr=False)
    resnet_names: list[str] = field(init=False, repr=False)
    transformer_names: list[str] = field(init=False, repr=False)
    # endregion
    # region absolute keys
    sampler_key: str = field(init=False, repr=False)
    # endregion
    # region child structs
    resnet_structs: list[DiffusionResnetStruct] = field(init=False, repr=False)
    transformer_structs: list[DiffusionTransformerStruct] = field(init=False, repr=False)
    # endregion

    @property
    def downsample(self) -> nn.Conv2d | None:
        return self.sampler if self.is_downsample_block() else None

    @property
    def upsample(self) -> nn.Conv2d | None:
        return self.sampler if self.is_upsample_block() else None

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.is_downsample_block():
            assert len(self.resnets) == len(self.transformers) or len(self.transformers) == 0
            if self.parent is not None and isinstance(self.parent, UNetStruct):
                assert self.rname == f"{self.parent.down_blocks_rname}.{self.idx}"
        elif self.is_mid_block():
            assert len(self.resnets) == len(self.transformers) + 1 or len(self.transformers) == 0
            if self.parent is not None and isinstance(self.parent, UNetStruct):
                assert self.rname == self.parent.mid_block_name
                assert self.idx == 0
        else:
            assert self.is_upsample_block(), f"Unsupported block type: {self.block_type}"
            assert len(self.resnets) == len(self.transformers) or len(self.transformers) == 0
            if self.parent is not None and isinstance(self.parent, UNetStruct):
                assert self.rname == f"{self.parent.up_blocks_rname}.{self.idx}"
        resnet_rnames = [f"{self.resnets_rname}.{idx}" for idx in range(len(self.resnets))]
        transformer_rnames = [f"{self.transformers_rname}.{idx}" for idx in range(len(self.transformers))]
        self.resnets_name = join_name(self.name, self.resnets_rname)
        self.transformers_name = join_name(self.name, self.transformers_rname)
        self.resnet_names = [join_name(self.name, rname) for rname in resnet_rnames]
        self.transformer_names = [join_name(self.name, rname) for rname in transformer_rnames]
        self.sampler_name = join_name(self.name, self.sampler_rname)
        self.sampler_key = join_name(self.key, self.sampler_rkey, sep="_")
        self.resnet_structs = [
            self.resnet_struct_cls.construct(
                resnet, parent=self, fname="resnet", rname=rname, rkey=self.resnet_rkey, idx=idx
            )
            for idx, (resnet, rname) in enumerate(zip(self.resnets, resnet_rnames, strict=True))
        ]
        self.transformer_structs = [
            self.transformer_struct_cls.construct(
                transformer, parent=self, fname="transformer", rname=rname, rkey=self.transformer_rkey, idx=idx
            )
            for idx, (transformer, rname) in enumerate(zip(self.transformers, transformer_rnames, strict=True))
        ]

    def is_downsample_block(self) -> bool:
        return self.block_type == self.BlockType.DOWN

    def is_mid_block(self) -> bool:
        return self.block_type == self.BlockType.MID

    def is_upsample_block(self) -> bool:
        return self.block_type == self.BlockType.UP

    def has_downsample(self) -> bool:
        return self.is_downsample_block() and self.sampler is not None

    def has_upsample(self) -> bool:
        return self.is_upsample_block() and self.sampler is not None

    def named_key_modules(self) -> tp.Generator[tp.Tuple[str, str, nn.Module, BaseModuleStruct, str], None, None]:
        for resnet in self.resnet_structs:
            yield from resnet.named_key_modules()
        for transformer in self.transformer_structs:
            yield from transformer.named_key_modules()
        if self.sampler is not None:
            yield self.sampler_key, self.sampler_name, self.sampler, self, "sampler"

    def iter_attention_structs(self) -> tp.Generator[DiffusionAttentionStruct, None, None]:
        for transformer in self.transformer_structs:
            yield from transformer.iter_attention_structs()

    def iter_transformer_block_structs(self) -> tp.Generator[DiffusionTransformerBlockStruct, None, None]:
        for transformer in self.transformer_structs:
            yield from transformer.iter_transformer_block_structs()

    @staticmethod
    def _default_construct(
        module: UNET_BLOCK_CLS,
        /,
        parent: tp.Optional["UNetStruct"] = None,
        fname: str = "",
        rname: str = "",
        rkey: str = "",
        idx: int = 0,
        **kwargs,
    ) -> "UNetBlockStruct":
        resnets, resnets_rname = module.resnets, "resnets"
        if isinstance(module, (DownBlock2D, CrossAttnDownBlock2D)):
            block_type = UNetBlockStruct.BlockType.DOWN
            if isinstance(module, CrossAttnDownBlock2D) and module.attentions is not None:
                transformers, transformers_rname = module.attentions, "attentions"
            else:
                transformers, transformers_rname = [], ""
            if module.downsamplers is None:
                sampler, sampler_rname = None, ""
            else:
                assert len(module.downsamplers) == 1
                downsampler = module.downsamplers[0]
                assert isinstance(downsampler, Downsample2D)
                sampler, sampler_rname = downsampler.conv, "downsamplers.0.conv"
                assert isinstance(sampler, nn.Conv2d)
        elif isinstance(module, (UNetMidBlock2D, UNetMidBlock2DCrossAttn)):
            block_type = UNetBlockStruct.BlockType.MID
            if (isinstance(module, UNetMidBlock2DCrossAttn) or module.add_attention) and module.attentions is not None:
                transformers, transformers_rname = module.attentions, "attentions"
            else:
                transformers, transformers_rname = [], ""
            sampler, sampler_rname = None, ""
        elif isinstance(module, (UpBlock2D, CrossAttnUpBlock2D)):
            block_type = UNetBlockStruct.BlockType.UP
            if isinstance(module, CrossAttnUpBlock2D) and module.attentions is not None:
                transformers, transformers_rname = module.attentions, "attentions"
            else:
                transformers, transformers_rname = [], ""
            if module.upsamplers is None:
                sampler, sampler_rname = None, ""
            else:
                assert len(module.upsamplers) == 1
                upsampler = module.upsamplers[0]
                assert isinstance(upsampler, Upsample2D)
                sampler, sampler_rname = upsampler.conv, "upsamplers.0.conv"
                assert isinstance(sampler, nn.Conv2d)
        else:
            raise NotImplementedError(f"Unsupported module type: {type(module)}")
        return UNetBlockStruct(
            module=module,
            parent=parent,
            fname=fname,
            idx=idx,
            rname=rname,
            rkey=rkey,
            block_type=block_type,
            resnets=resnets,
            transformers=transformers,
            sampler=sampler,
            resnets_rname=resnets_rname,
            transformers_rname=transformers_rname,
            sampler_rname=sampler_rname,
        )

    @classmethod
    def _get_default_key_map(cls) -> dict[str, set[str]]:
        """Get the default allowed keys."""
        key_map: dict[str, set[str]] = defaultdict(set)
        resnet_cls = cls.resnet_struct_cls
        resnet_key = resnet_rkey = cls.resnet_rkey
        resnet_key_map = resnet_cls._get_default_key_map()
        for rkey, keys in resnet_key_map.items():
            rkey = join_name(resnet_rkey, rkey, sep="_")
            for key in keys:
                key = join_name(resnet_key, key, sep="_")
                key_map[rkey].add(key)
                key_map[resnet_rkey].add(key)
        transformer_cls = cls.transformer_struct_cls
        transformer_key = transformer_rkey = cls.transformer_rkey
        transformer_key_map = transformer_cls._get_default_key_map()
        for rkey, keys in transformer_key_map.items():
            trkey = join_name(transformer_rkey, rkey, sep="_")
            for key in keys:
                key = join_name(transformer_key, key, sep="_")
                key_map[rkey].add(key)
                key_map[trkey].add(key)
        return {k: v for k, v in key_map.items() if v}


@dataclass(kw_only=True)
class UNetStruct(DiffusionModelStruct):
    # region relative keys
    input_embed_rkey: tp.ClassVar[str] = "input_embed"
    """hidden_states = input_embed(hidden_states), e.g., conv_in"""
    time_embed_rkey: tp.ClassVar[str] = "time_embed"
    """temb = time_embed(timesteps, hidden_states)"""
    add_time_embed_rkey: tp.ClassVar[str] = "time_embed"
    """add_temb = add_time_embed(timesteps, encoder_hidden_states)"""
    text_embed_rkey: tp.ClassVar[str] = "text_embed"
    """encoder_hidden_states = text_embed(encoder_hidden_states)"""
    norm_out_rkey: tp.ClassVar[str] = "output_embed"
    """hidden_states = norm_out(hidden_states), e.g., conv_norm_out"""
    proj_out_rkey: tp.ClassVar[str] = "output_embed"
    """hidden_states = output_embed(hidden_states), e.g., conv_out"""
    down_block_rkey: tp.ClassVar[str] = "down"
    mid_block_rkey: tp.ClassVar[str] = "mid"
    up_block_rkey: tp.ClassVar[str] = "up"
    down_block_struct_cls: tp.ClassVar[type[UNetBlockStruct]] = UNetBlockStruct
    mid_block_struct_cls: tp.ClassVar[type[UNetBlockStruct]] = UNetBlockStruct
    up_block_struct_cls: tp.ClassVar[type[UNetBlockStruct]] = UNetBlockStruct
    # endregion

    # region child modules
    # region pre-block modules
    input_embed: nn.Conv2d
    time_embed: TimestepEmbedding
    """Time embedding"""
    add_time_embed: (
        TextTimeEmbedding
        | TextImageTimeEmbedding
        | TimestepEmbedding
        | ImageTimeEmbedding
        | ImageHintTimeEmbedding
        | None
    )
    """Additional time embedding"""
    text_embed: nn.Linear | ImageProjection | TextImageProjection | None
    """Text embedding"""
    # region post-block modules
    norm_out: nn.GroupNorm | None
    proj_out: nn.Conv2d
    # endregion
    # endregion
    down_blocks: nn.ModuleList = field(repr=False)
    mid_block: nn.Module = field(repr=False)
    up_blocks: nn.ModuleList = field(repr=False)
    # endregion
    # region relative names
    input_embed_rname: str
    time_embed_rname: str
    add_time_embed_rname: str
    text_embed_rname: str
    norm_out_rname: str
    proj_out_rname: str
    down_blocks_rname: str
    mid_block_rname: str
    up_blocks_rname: str
    # endregion
    # region absolute names
    input_embed_name: str = field(init=False, repr=False)
    time_embed_name: str = field(init=False, repr=False)
    add_time_embed_name: str = field(init=False, repr=False)
    text_embed_name: str = field(init=False, repr=False)
    norm_out_name: str = field(init=False, repr=False)
    proj_out_name: str = field(init=False, repr=False)
    down_blocks_name: str = field(init=False, repr=False)
    mid_block_name: str = field(init=False, repr=False)
    up_blocks_name: str = field(init=False, repr=False)
    down_block_names: list[str] = field(init=False, repr=False)
    up_block_names: list[str] = field(init=False, repr=False)
    # endregion
    # region absolute keys
    input_embed_key: str = field(init=False, repr=False)
    time_embed_key: str = field(init=False, repr=False)
    add_time_embed_key: str = field(init=False, repr=False)
    text_embed_key: str = field(init=False, repr=False)
    norm_out_key: str = field(init=False, repr=False)
    proj_out_key: str = field(init=False, repr=False)
    # endregion
    # region child structs
    down_block_structs: list[UNetBlockStruct] = field(init=False, repr=False)
    mid_block_struct: UNetBlockStruct = field(init=False, repr=False)
    up_block_structs: list[UNetBlockStruct] = field(init=False, repr=False)
    # endregion

    @property
    def num_down_blocks(self) -> int:
        return len(self.down_blocks)

    @property
    def num_up_blocks(self) -> int:
        return len(self.up_blocks)

    @property
    def num_blocks(self) -> int:
        return self.num_down_blocks + 1 + self.num_up_blocks

    @property
    def block_structs(self) -> list[UNetBlockStruct]:
        return [*self.down_block_structs, self.mid_block_struct, *self.up_block_structs]

    def __post_init__(self) -> None:
        super().__post_init__()
        down_block_rnames = [f"{self.down_blocks_rname}.{idx}" for idx in range(len(self.down_blocks))]
        up_block_rnames = [f"{self.up_blocks_rname}.{idx}" for idx in range(len(self.up_blocks))]
        self.down_blocks_name = join_name(self.name, self.down_blocks_rname)
        self.mid_block_name = join_name(self.name, self.mid_block_rname)
        self.up_blocks_name = join_name(self.name, self.up_blocks_rname)
        self.down_block_names = [join_name(self.name, rname) for rname in down_block_rnames]
        self.up_block_names = [join_name(self.name, rname) for rname in up_block_rnames]
        self.pre_module_structs = {}
        for fname in ("time_embed", "add_time_embed", "text_embed", "input_embed"):
            module, rname, rkey = getattr(self, fname), getattr(self, f"{fname}_rname"), getattr(self, f"{fname}_rkey")
            setattr(self, f"{fname}_key", join_name(self.key, rkey, sep="_"))
            if module is not None or rname:
                setattr(self, f"{fname}_name", join_name(self.name, rname))
            else:
                setattr(self, f"{fname}_name", "")
            if module is not None:
                assert rname, f"rname of {fname} must not be empty"
                self.pre_module_structs[getattr(self, f"{fname}_name")] = DiffusionModuleStruct(
                    module=module, parent=self, fname=fname, rname=rname, rkey=rkey
                )
        self.post_module_structs = {}
        for fname in ("norm_out", "proj_out"):
            module, rname, rkey = getattr(self, fname), getattr(self, f"{fname}_rname"), getattr(self, f"{fname}_rkey")
            setattr(self, f"{fname}_key", join_name(self.key, rkey, sep="_"))
            if module is not None or rname:
                setattr(self, f"{fname}_name", join_name(self.name, rname))
            else:
                setattr(self, f"{fname}_name", "")
            if module is not None:
                self.post_module_structs[getattr(self, f"{fname}_name")] = DiffusionModuleStruct(
                    module=module, parent=self, fname=fname, rname=rname, rkey=rkey
                )
        self.down_block_structs = [
            self.down_block_struct_cls.construct(
                block, parent=self, fname="down_block", rname=rname, rkey=self.down_block_rkey, idx=idx
            )
            for idx, (block, rname) in enumerate(zip(self.down_blocks, down_block_rnames, strict=True))
        ]
        self.mid_block_struct = self.mid_block_struct_cls.construct(
            self.mid_block, parent=self, fname="mid_block", rname=self.mid_block_name, rkey=self.mid_block_rkey
        )
        self.up_block_structs = [
            self.up_block_struct_cls.construct(
                block, parent=self, fname="up_block", rname=rname, rkey=self.up_block_rkey, idx=idx
            )
            for idx, (block, rname) in enumerate(zip(self.up_blocks, up_block_rnames, strict=True))
        ]

    def get_prev_module_keys(self) -> tuple[str, ...]:
        return tuple({self.input_embed_key, self.time_embed_key, self.add_time_embed_key, self.text_embed_key})

    def get_post_module_keys(self) -> tuple[str, ...]:
        return tuple({self.norm_out_key, self.proj_out_key})

    def _get_iter_block_activations_args(
        self, **input_kwargs
    ) -> tuple[list[nn.Module], list[DiffusionModuleStruct | DiffusionBlockStruct], list[bool], list[bool]]:
        layers, layer_structs, recomputes, use_prev_layer_outputs = [], [], [], []
        num_down_blocks = len(self.down_blocks)
        num_up_blocks = len(self.up_blocks)
        layers.extend(self.down_blocks)
        layer_structs.extend(self.down_block_structs)
        use_prev_layer_outputs.append(False)
        use_prev_layer_outputs.extend([True] * (num_down_blocks - 1))
        recomputes.append(False)
        # region check whether down block's outputs are changed
        _mid_block_additional_residual = input_kwargs.get("mid_block_additional_residual", None)
        _down_block_additional_residuals = input_kwargs.get("down_block_additional_residuals", None)
        _is_adapter = input_kwargs.get("down_intrablock_additional_residuals", None) is not None
        if not _is_adapter and _mid_block_additional_residual is None and _down_block_additional_residuals is not None:
            _is_adapter = True
        for down_block in self.down_blocks:
            if hasattr(down_block, "has_cross_attention") and down_block.has_cross_attention:
                # outputs unchanged
                recomputes.append(False)
            elif _is_adapter:
                # outputs changed
                recomputes.append(True)
            else:
                # outputs unchanged
                recomputes.append(False)
        # endregion
        layers.append(self.mid_block)
        layer_structs.append(self.mid_block_struct)
        use_prev_layer_outputs.append(False)
        # recomputes is already appened in the previous down blocks
        layers.extend(self.up_blocks)
        layer_structs.extend(self.up_block_structs)
        use_prev_layer_outputs.append(False)
        use_prev_layer_outputs.extend([True] * (num_up_blocks - 1))
        recomputes += [True] * num_up_blocks
        return layers, layer_structs, recomputes, use_prev_layer_outputs

    @staticmethod
    def _default_construct(
        module: tp.Union[UNET_PIPELINE_CLS, UNET_CLS],
        /,
        parent: tp.Optional[BaseModuleStruct] = None,
        fname: str = "",
        rname: str = "",
        rkey: str = "",
        idx: int = 0,
        **kwargs,
    ) -> "UNetStruct":
        if isinstance(module, UNET_PIPELINE_CLS):
            module = module.unet
        if isinstance(module, (UNet2DConditionModel, UNet2DModel)):
            input_embed, time_embed = module.conv_in, module.time_embedding
            input_embed_rname, time_embed_rname = "conv_in", "time_embedding"
            text_embed, text_embed_rname = None, ""
            add_time_embed, add_time_embed_rname = None, ""
            if hasattr(module, "encoder_hid_proj"):
                text_embed, text_embed_rname = module.encoder_hid_proj, "encoder_hid_proj"
            if hasattr(module, "add_embedding"):
                add_time_embed, add_time_embed_rname = module.add_embedding, "add_embedding"
            norm_out, norm_out_rname = module.conv_norm_out, "conv_norm_out"
            proj_out, proj_out_rname = module.conv_out, "conv_out"
            down_blocks, down_blocks_rname = module.down_blocks, "down_blocks"
            mid_block, mid_block_rname = module.mid_block, "mid_block"
            up_blocks, up_blocks_rname = module.up_blocks, "up_blocks"
            return UNetStruct(
                module=module,
                parent=parent,
                fname=fname,
                idx=idx,
                rname=rname,
                rkey=rkey,
                input_embed=input_embed,
                time_embed=time_embed,
                add_time_embed=add_time_embed,
                text_embed=text_embed,
                norm_out=norm_out,
                proj_out=proj_out,
                down_blocks=down_blocks,
                mid_block=mid_block,
                up_blocks=up_blocks,
                input_embed_rname=input_embed_rname,
                time_embed_rname=time_embed_rname,
                add_time_embed_rname=add_time_embed_rname,
                text_embed_rname=text_embed_rname,
                norm_out_rname=norm_out_rname,
                proj_out_rname=proj_out_rname,
                down_blocks_rname=down_blocks_rname,
                mid_block_rname=mid_block_rname,
                up_blocks_rname=up_blocks_rname,
            )
        raise NotImplementedError(f"Unsupported module type: {type(module)}")

    @classmethod
    def _get_default_key_map(cls) -> dict[str, set[str]]:
        """Get the default allowed keys."""
        key_map: dict[str, set[str]] = defaultdict(set)
        for idx, (block_key, block_cls) in enumerate(
            (
                (cls.down_block_rkey, cls.down_block_struct_cls),
                (cls.mid_block_rkey, cls.mid_block_struct_cls),
                (cls.up_block_rkey, cls.up_block_struct_cls),
            )
        ):
            block_key_map: dict[str, set[str]] = defaultdict(set)
            if idx != 1:
                sampler_key = join_name(block_key, block_cls.sampler_rkey, sep="_")
                sampler_rkey = block_cls.sampler_rkey
                block_key_map[sampler_rkey].add(sampler_key)
            _block_key_map = block_cls._get_default_key_map()
            for rkey, keys in _block_key_map.items():
                for key in keys:
                    key = join_name(block_key, key, sep="_")
                    block_key_map[rkey].add(key)
            for rkey, keys in block_key_map.items():
                key_map[rkey].update(keys)
                if block_key:
                    key_map[block_key].update(keys)
        keys: set[str] = set()
        keys.add(cls.input_embed_rkey)
        keys.add(cls.time_embed_rkey)
        keys.add(cls.add_time_embed_rkey)
        keys.add(cls.text_embed_rkey)
        keys.add(cls.norm_out_rkey)
        keys.add(cls.proj_out_rkey)
        for mapped_keys in key_map.values():
            for key in mapped_keys:
                keys.add(key)
        if "embed" not in keys and "embed" not in key_map:
            key_map["embed"].add(cls.input_embed_rkey)
            key_map["embed"].add(cls.time_embed_rkey)
            key_map["embed"].add(cls.add_time_embed_rkey)
            key_map["embed"].add(cls.text_embed_rkey)
            key_map["embed"].add(cls.norm_out_rkey)
            key_map["embed"].add(cls.proj_out_rkey)
        for key in keys:
            if key in key_map:
                key_map[key].clear()
            key_map[key].add(key)
        return {k: v for k, v in key_map.items() if v}


@dataclass(kw_only=True)
class DiTStruct(DiffusionModelStruct, DiffusionTransformerStruct):
    # region relative keys
    input_embed_rkey: tp.ClassVar[str] = "input_embed"
    """hidden_states = input_embed(hidden_states), e.g., conv_in"""
    time_embed_rkey: tp.ClassVar[str] = "time_embed"
    """temb = time_embed(timesteps)"""
    text_embed_rkey: tp.ClassVar[str] = "text_embed"
    """encoder_hidden_states = text_embed(encoder_hidden_states)"""
    text_norm_rkey: tp.ClassVar[str] = "text_embed"
    """encoder_hidden_states = text_norm(encoder_hidden_states), share key with text_embed"""
    norm_in_rkey: tp.ClassVar[str] = "input_embed"
    """hidden_states = norm_in(hidden_states)"""
    proj_in_rkey: tp.ClassVar[str] = "input_embed"
    """hidden_states = proj_in(hidden_states)"""
    norm_out_rkey: tp.ClassVar[str] = "output_embed"
    """hidden_states = norm_out(hidden_states)"""
    proj_out_rkey: tp.ClassVar[str] = "output_embed"
    """hidden_states = proj_out(hidden_states)"""
    transformer_block_rkey: tp.ClassVar[str] = ""
    # endregion

    # region child modules
    input_embed: PatchEmbed
    time_embed: AdaLayerNormSingle | CombinedTimestepTextProjEmbeddings | TimestepEmbedding
    text_embed: PixArtAlphaTextProjection | nn.Linear
    text_norm: nn.Module | None = None
    """Optional text normalization layer (e.g., RMSNorm in QwenImage)"""
    norm_in: None = field(init=False, repr=False, default=None)
    proj_in: None = field(init=False, repr=False, default=None)
    norm_out: nn.LayerNorm | AdaLayerNormContinuous | None
    proj_out: nn.Linear
    # endregion
    # region relative names
    input_embed_rname: str
    time_embed_rname: str
    text_embed_rname: str
    text_norm_rname: str = ""
    norm_in_rname: str = field(init=False, repr=False, default="")
    proj_in_rname: str = field(init=False, repr=False, default="")
    norm_out_rname: str
    proj_out_rname: str
    # endregion
    # region absolute names
    input_embed_name: str = field(init=False, repr=False)
    time_embed_name: str = field(init=False, repr=False)
    text_embed_name: str = field(init=False, repr=False)
    text_norm_name: str = field(init=False, repr=False, default="")
    # endregion
    # region absolute keys
    input_embed_key: str = field(init=False, repr=False)
    time_embed_key: str = field(init=False, repr=False)
    text_embed_key: str = field(init=False, repr=False)
    text_norm_key: str = field(init=False, repr=False, default="")
    norm_out_key: str = field(init=False, repr=False)
    # endregion

    @property
    def config(self):
        """Proxy to the underlying module's config."""
        return self.module.config

    def __getattr__(self, name: str):
        """Proxy undefined attributes to the underlying module."""
        # Avoid infinite recursion by checking if 'module' exists
        if name == 'module':
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        # Try to get the attribute from the underlying module
        try:
            return getattr(self.module, name)
        except AttributeError:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    @property
    def num_blocks(self) -> int:
        return len(self.transformer_blocks)

    @property
    def block_structs(self) -> list[DiffusionTransformerBlockStruct]:
        return self.transformer_block_structs

    @property
    def block_names(self) -> list[str]:
        return self.transformer_block_names

    def __post_init__(self) -> None:
        super().__post_init__()
        self.pre_module_structs = {}
        for fname in ("input_embed", "time_embed", "text_embed", "text_norm"):
            module, rname, rkey = getattr(self, fname), getattr(self, f"{fname}_rname"), getattr(self, f"{fname}_rkey")
            setattr(self, f"{fname}_key", join_name(self.key, rkey, sep="_"))
            if module is not None or rname:
                setattr(self, f"{fname}_name", join_name(self.name, rname))
            else:
                setattr(self, f"{fname}_name", "")
            if module is not None:
                self.pre_module_structs.setdefault(
                    getattr(self, f"{fname}_name"),
                    DiffusionModuleStruct(module=module, parent=self, fname=fname, rname=rname, rkey=rkey),
                )
        self.post_module_structs = {}
        self.norm_out_key = join_name(self.key, self.norm_out_rkey, sep="_")
        for fname in ("norm_out", "proj_out"):
            module, rname, rkey = getattr(self, fname), getattr(self, f"{fname}_rname"), getattr(self, f"{fname}_rkey")
            if module is not None:
                self.post_module_structs.setdefault(
                    getattr(self, f"{fname}_name"),
                    DiffusionModuleStruct(module=module, parent=self, fname=fname, rname=rname, rkey=rkey),
                )

    def get_prev_module_keys(self) -> tuple[str, ...]:
        keys = {self.input_embed_key, self.time_embed_key, self.text_embed_key}
        if self.text_norm_key:
            keys.add(self.text_norm_key)
        return tuple(keys)

    def get_post_module_keys(self) -> tuple[str, ...]:
        return tuple({self.norm_out_key, self.proj_out_key})

    def _get_iter_block_activations_args(
        self, **input_kwargs
    ) -> tuple[list[nn.Module], list[DiffusionModuleStruct | DiffusionBlockStruct], list[bool], list[bool]]:
        """
        Get the arguments for iterating over the layers and their activations.

        Args:
            skip_pre_modules (`bool`):
                Whether to skip the pre-modules
            skip_post_modules (`bool`):
                Whether to skip the post-modules

        Returns:
            `tuple[list[nn.Module], list[DiffusionModuleStruct | DiffusionBlockStruct], list[bool], list[bool]]`:
                the layers, the layer structs, the recomputes, and the use_prev_layer_outputs
        """
        layers, layer_structs, recomputes, use_prev_layer_outputs = [], [], [], []
        layers.extend(self.transformer_blocks)
        layer_structs.extend(self.transformer_block_structs)
        use_prev_layer_outputs.append(False)
        use_prev_layer_outputs.extend([True] * (len(self.transformer_blocks) - 1))
        recomputes.extend([False] * len(self.transformer_blocks))
        return layers, layer_structs, recomputes, use_prev_layer_outputs

    @staticmethod
    def _default_construct(
        module: tp.Union[DIT_PIPELINE_CLS, DIT_CLS],
        /,
        parent: tp.Optional[BaseModuleStruct] = None,
        fname: str = "",
        rname: str = "",
        rkey: str = "",
        idx: int = 0,
        **kwargs,
    ) -> "DiTStruct":
        if isinstance(module, DIT_PIPELINE_CLS):
            module = module.transformer
        if isinstance(module, FluxTransformer2DModel):
            return FluxStruct.construct(module, parent=parent, fname=fname, rname=rname, rkey=rkey, idx=idx, **kwargs)
        else:
            if isinstance(module, PixArtTransformer2DModel):
                input_embed, input_embed_rname = module.pos_embed, "pos_embed"
                time_embed, time_embed_rname = module.adaln_single, "adaln_single"
                text_embed, text_embed_rname = module.caption_projection, "caption_projection"
                norm_out, norm_out_rname = module.norm_out, "norm_out"
                proj_out, proj_out_rname = module.proj_out, "proj_out"
                transformer_blocks, transformer_blocks_rname = module.transformer_blocks, "transformer_blocks"
                # ! in fact, `module.adaln_single.emb` is `time_embed`,
                # ! `module.adaln_single.linear` is `transformer_norm`
                # ! but since PixArt shares the `transformer_norm`, we categorize it as `time_embed`
            elif isinstance(module, SanaTransformer2DModel):
                input_embed, input_embed_rname = module.patch_embed, "patch_embed"
                time_embed, time_embed_rname = module.time_embed, "time_embed"
                text_embed, text_embed_rname = module.caption_projection, "caption_projection"
                norm_out, norm_out_rname = module.norm_out, "norm_out"
                proj_out, proj_out_rname = module.proj_out, "proj_out"
                transformer_blocks, transformer_blocks_rname = module.transformer_blocks, "transformer_blocks"
            elif isinstance(module, SD3Transformer2DModel):
                input_embed, input_embed_rname = module.pos_embed, "pos_embed"
                time_embed, time_embed_rname = module.time_text_embed, "time_text_embed"
                text_embed, text_embed_rname = module.context_embedder, "context_embedder"
                norm_out, norm_out_rname = module.norm_out, "norm_out"
                proj_out, proj_out_rname = module.proj_out, "proj_out"
                transformer_blocks, transformer_blocks_rname = module.transformer_blocks, "transformer_blocks"
            elif isinstance(module, QwenImageTransformer2DModel):
                input_embed, input_embed_rname = module.img_in, "img_in"
                time_embed, time_embed_rname = module.time_text_embed, "time_text_embed"
                text_embed, text_embed_rname = module.txt_in, "txt_in"
                norm_out, norm_out_rname = module.norm_out, "norm_out"
                proj_out, proj_out_rname = module.proj_out, "proj_out"
                transformer_blocks, transformer_blocks_rname = module.transformer_blocks, "transformer_blocks"
            elif isinstance(module, WanTransformer3DModel):
                input_embed, input_embed_rname = module.patch_embedding, "patch_embedding"
                time_embed, time_embed_rname = module.condition_embedder, "condition_embedder"
                text_embed, text_embed_rname = None, ""  # Wan uses combined time+text+image embedding
                norm_out, norm_out_rname = module.norm_out, "norm_out"
                proj_out, proj_out_rname = module.proj_out, "proj_out"
                transformer_blocks, transformer_blocks_rname = module.blocks, "blocks"
            else:
                raise NotImplementedError(f"Unsupported module type: {type(module)}")
            # text_norm is optional, only QwenImage has it
            text_norm = getattr(module, "txt_norm", None)
            text_norm_rname = "txt_norm" if text_norm is not None else ""
            return DiTStruct(
                module=module,
                parent=parent,
                fname=fname,
                idx=idx,
                rname=rname,
                rkey=rkey,
                input_embed=input_embed,
                time_embed=time_embed,
                text_embed=text_embed,
                text_norm=text_norm,
                transformer_blocks=transformer_blocks,
                norm_out=norm_out,
                proj_out=proj_out,
                input_embed_rname=input_embed_rname,
                time_embed_rname=time_embed_rname,
                text_embed_rname=text_embed_rname,
                text_norm_rname=text_norm_rname,
                norm_out_rname=norm_out_rname,
                proj_out_rname=proj_out_rname,
                transformer_blocks_rname=transformer_blocks_rname,
            )

    @classmethod
    def _get_default_key_map(cls) -> dict[str, set[str]]:
        """Get the default allowed keys."""
        key_map: dict[str, set[str]] = defaultdict(set)
        block_cls = cls.transformer_block_struct_cls
        block_key = block_rkey = cls.transformer_block_rkey
        block_key_map = block_cls._get_default_key_map()
        for rkey, keys in block_key_map.items():
            brkey = join_name(block_rkey, rkey, sep="_")
            for key in keys:
                key = join_name(block_key, key, sep="_")
                key_map[rkey].add(key)
                key_map[brkey].add(key)
                if block_rkey:
                    key_map[block_rkey].add(key)
        keys: set[str] = set()
        keys.add(cls.input_embed_rkey)
        keys.add(cls.time_embed_rkey)
        keys.add(cls.text_embed_rkey)
        keys.add(cls.text_norm_rkey)
        keys.add(cls.norm_in_rkey)
        keys.add(cls.proj_in_rkey)
        keys.add(cls.norm_out_rkey)
        keys.add(cls.proj_out_rkey)
        for mapped_keys in key_map.values():
            for key in mapped_keys:
                keys.add(key)
        if "embed" not in keys and "embed" not in key_map:
            key_map["embed"].add(cls.input_embed_rkey)
            key_map["embed"].add(cls.time_embed_rkey)
            key_map["embed"].add(cls.text_embed_rkey)
            key_map["embed"].add(cls.text_norm_rkey)
            key_map["embed"].add(cls.norm_in_rkey)
            key_map["embed"].add(cls.proj_in_rkey)
            key_map["embed"].add(cls.norm_out_rkey)
            key_map["embed"].add(cls.proj_out_rkey)
        for key in keys:
            if key in key_map:
                key_map[key].clear()
            key_map[key].add(key)
        return {k: v for k, v in key_map.items() if v}


@dataclass(kw_only=True)
class FluxStruct(DiTStruct):
    # region relative keys
    single_transformer_block_rkey: tp.ClassVar[str] = ""
    single_transformer_block_struct_cls: tp.ClassVar[type[DiffusionTransformerBlockStruct]] = (
        DiffusionTransformerBlockStruct
    )
    # endregion

    module: FluxTransformer2DModel = field(repr=False, kw_only=False)
    """the module of FluxTransformer2DModel"""
    # region child modules
    input_embed: nn.Linear
    time_embed: CombinedTimestepGuidanceTextProjEmbeddings | CombinedTimestepTextProjEmbeddings
    text_embed: nn.Linear
    single_transformer_blocks: nn.ModuleList = field(repr=False)
    # endregion
    # region relative names
    single_transformer_blocks_rname: str
    # endregion
    # region absolute names
    single_transformer_blocks_name: str = field(init=False, repr=False)
    single_transformer_block_names: list[str] = field(init=False, repr=False)
    # endregion
    # region child structs
    single_transformer_block_structs: list[DiffusionTransformerBlockStruct] = field(init=False)
    # endregion

    @property
    def num_blocks(self) -> int:
        return len(self.transformer_block_structs) + len(self.single_transformer_block_structs)

    @property
    def block_structs(self) -> list[DiffusionTransformerBlockStruct]:
        return [*self.transformer_block_structs, *self.single_transformer_block_structs]

    @property
    def block_names(self) -> list[str]:
        return [*self.transformer_block_names, *self.single_transformer_block_names]

    def __post_init__(self) -> None:
        super().__post_init__()
        single_transformer_block_rnames = [
            f"{self.single_transformer_blocks_rname}.{idx}" for idx in range(len(self.single_transformer_blocks))
        ]
        self.single_transformer_blocks_name = join_name(self.name, self.single_transformer_blocks_rname)
        self.single_transformer_block_names = [join_name(self.name, rname) for rname in single_transformer_block_rnames]
        self.single_transformer_block_structs = [
            self.single_transformer_block_struct_cls.construct(
                block,
                parent=self,
                fname="single_transformer_block",
                rname=rname,
                rkey=self.single_transformer_block_rkey,
                idx=idx,
            )
            for idx, (block, rname) in enumerate(
                zip(self.single_transformer_blocks, single_transformer_block_rnames, strict=True)
            )
        ]

    def _get_iter_block_activations_args(
        self, **input_kwargs
    ) -> tuple[list[nn.Module], list[DiffusionModuleStruct | DiffusionBlockStruct], list[bool], list[bool]]:
        layers, layer_structs, recomputes, use_prev_layer_outputs = super()._get_iter_block_activations_args()
        layers.extend(self.single_transformer_blocks)
        layer_structs.extend(self.single_transformer_block_structs)
        use_prev_layer_outputs.append(False)
        use_prev_layer_outputs.extend([True] * (len(self.single_transformer_blocks) - 1))
        recomputes.extend([False] * len(self.single_transformer_blocks))
        return layers, layer_structs, recomputes, use_prev_layer_outputs

    @staticmethod
    def _default_construct(
        module: tp.Union[FluxPipeline, FluxControlPipeline, FluxTransformer2DModel],
        /,
        parent: tp.Optional[BaseModuleStruct] = None,
        fname: str = "",
        rname: str = "",
        rkey: str = "",
        idx: int = 0,
        **kwargs,
    ) -> "FluxStruct":
        if isinstance(module, (FluxPipeline, FluxControlPipeline)):
            module = module.transformer
        if isinstance(module, FluxTransformer2DModel):
            input_embed, time_embed, text_embed = module.x_embedder, module.time_text_embed, module.context_embedder
            input_embed_rname, time_embed_rname, text_embed_rname = "x_embedder", "time_text_embed", "context_embedder"
            norm_out, norm_out_rname = module.norm_out, "norm_out"
            proj_out, proj_out_rname = module.proj_out, "proj_out"
            transformer_blocks, transformer_blocks_rname = module.transformer_blocks, "transformer_blocks"
            single_transformer_blocks = module.single_transformer_blocks
            single_transformer_blocks_rname = "single_transformer_blocks"
            return FluxStruct(
                module=module,
                parent=parent,
                fname=fname,
                idx=idx,
                rname=rname,
                rkey=rkey,
                input_embed=input_embed,
                time_embed=time_embed,
                text_embed=text_embed,
                transformer_blocks=transformer_blocks,
                single_transformer_blocks=single_transformer_blocks,
                norm_out=norm_out,
                proj_out=proj_out,
                input_embed_rname=input_embed_rname,
                time_embed_rname=time_embed_rname,
                text_embed_rname=text_embed_rname,
                norm_out_rname=norm_out_rname,
                proj_out_rname=proj_out_rname,
                transformer_blocks_rname=transformer_blocks_rname,
                single_transformer_blocks_rname=single_transformer_blocks_rname,
            )
        raise NotImplementedError(f"Unsupported module type: {type(module)}")

    @classmethod
    def _get_default_key_map(cls) -> dict[str, set[str]]:
        """Get the default allowed keys."""
        key_map: dict[str, set[str]] = defaultdict(set)
        for block_rkey, block_cls in (
            (cls.transformer_block_rkey, cls.transformer_block_struct_cls),
            (cls.single_transformer_block_rkey, cls.single_transformer_block_struct_cls),
        ):
            block_key = block_rkey
            block_key_map = block_cls._get_default_key_map()
            for rkey, keys in block_key_map.items():
                brkey = join_name(block_rkey, rkey, sep="_")
                for key in keys:
                    key = join_name(block_key, key, sep="_")
                    key_map[rkey].add(key)
                    key_map[brkey].add(key)
                    if block_rkey:
                        key_map[block_rkey].add(key)
        keys: set[str] = set()
        keys.add(cls.input_embed_rkey)
        keys.add(cls.time_embed_rkey)
        keys.add(cls.text_embed_rkey)
        keys.add(cls.norm_in_rkey)
        keys.add(cls.proj_in_rkey)
        keys.add(cls.norm_out_rkey)
        keys.add(cls.proj_out_rkey)
        for mapped_keys in key_map.values():
            for key in mapped_keys:
                keys.add(key)
        if "embed" not in keys and "embed" not in key_map:
            key_map["embed"].add(cls.input_embed_rkey)
            key_map["embed"].add(cls.time_embed_rkey)
            key_map["embed"].add(cls.text_embed_rkey)
            key_map["embed"].add(cls.norm_in_rkey)
            key_map["embed"].add(cls.proj_in_rkey)
            key_map["embed"].add(cls.norm_out_rkey)
            key_map["embed"].add(cls.proj_out_rkey)
        for key in keys:
            if key in key_map:
                key_map[key].clear()
            key_map[key].add(key)
        return {k: v for k, v in key_map.items() if v}


DiffusionAttentionStruct.register_factory((Attention, FluxAttention, WanAttention), DiffusionAttentionStruct._default_construct)

DiffusionFeedForwardStruct.register_factory(
    (FeedForward, FluxSingleTransformerBlock, GLUMBConv), DiffusionFeedForwardStruct._default_construct
)

DiffusionTransformerBlockStruct.register_factory(DIT_BLOCK_CLS, DiffusionTransformerBlockStruct._default_construct)

UNetBlockStruct.register_factory(UNET_BLOCK_CLS, UNetBlockStruct._default_construct)

UNetStruct.register_factory(tp.Union[UNET_PIPELINE_CLS, UNET_CLS], UNetStruct._default_construct)

FluxStruct.register_factory(
    tp.Union[FluxPipeline, FluxControlPipeline, FluxTransformer2DModel], FluxStruct._default_construct
)

DiTStruct.register_factory(tp.Union[DIT_PIPELINE_CLS, DIT_CLS], DiTStruct._default_construct)

DiffusionTransformerStruct.register_factory(Transformer2DModel, DiffusionTransformerStruct._default_construct)

DiffusionModelStruct.register_factory(tp.Union[PIPELINE_CLS, MODEL_CLS], DiffusionModelStruct._default_construct)
