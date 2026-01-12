# -*- coding: utf-8 -*-
"""Diffusion pipeline configuration module."""

import gc
import typing as tp
from dataclasses import dataclass, field

import torch
from diffusers.pipelines import (
    AutoPipelineForText2Image,
    DiffusionPipeline,
    FluxControlPipeline,
    FluxFillPipeline,
    SanaPipeline,
)
from omniconfig import configclass
from torch import nn
from transformers import PreTrainedModel, PreTrainedTokenizer, T5EncoderModel

from deepcompressor.data.utils.dtype import eval_dtype
from deepcompressor.quantizer.processor import Quantizer
from deepcompressor.utils import tools
from deepcompressor.utils.hooks import AccumBranchHook, ProcessHook

from ....nn.patch.linear import ConcatLinear, ShiftedLinear
from ....nn.patch.lowrank import LowRankBranch
from ..nn.patch import (
    replace_fused_linear_with_concat_linear,
    replace_up_block_conv_with_concat_conv,
    shift_input_activations,
)
from ..qwenimage import QwenImagePipelineWithSeqLen as QwenImagePipeline
from ..wan_i2v import WanImageToVideoPipelineForQuant
#from diffusers.pipelines.qwenimage import QwenImagePipeline
__all__ = ["DiffusionPipelineConfig"]


@configclass
@dataclass
class LoRAConfig:
    """LoRA configuration.

    Args:
        path (`str`):
            The path of the LoRA branch.
        weight_name (`str`):
            The weight name of the LoRA branch.
        alpha (`float`):
            The alpha value of the LoRA branch.
    """

    path: str
    weight_name: str
    alpha: float = 1.0


@configclass
@dataclass
class DiffusionPipelineConfig:
    """Diffusion pipeline configuration.

    Args:
        name (`str`):
            The name of the pipeline.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            The data type of the pipeline.
        device (`str`, *optional*, defaults to `"cuda"`):
            The device of the pipeline.
        shift_activations (`bool`, *optional*, defaults to `False`):
            Whether to shift activations.
    """

    _pipeline_factories: tp.ClassVar[
        dict[str, tp.Callable[[str, str, torch.dtype, torch.device, bool], DiffusionPipeline]]
    ] = {}
    _text_extractors: tp.ClassVar[
        dict[
            str,
            tp.Callable[
                [DiffusionPipeline, tuple[type[PreTrainedModel], ...]],
                list[tuple[str, PreTrainedModel, PreTrainedTokenizer]],
            ],
        ]
    ] = {}

    name: str
    path: str = ""
    dtype: torch.dtype = field(
        default_factory=lambda s=torch.float32: eval_dtype(s, with_quant_dtype=False, with_none=False)
    )
    device: str = "cuda"
    shift_activations: bool = False
    lora: LoRAConfig | None = None
    family: str = field(init=False)
    task: str = "text-to-image"

    def __post_init__(self):
        self.family = self.name.split("-")[0]

        if self.name == "flux.1-canny-dev":
            self.task = "canny-to-image"
        elif self.name == "flux.1-depth-dev":
            self.task = "depth-to-image"
        elif self.name == "flux.1-fill-dev":
            self.task = "inpainting"

    def build(
        self, *, dtype: str | torch.dtype | None = None, device: str | torch.device | None = None
    ) -> DiffusionPipeline:
        """Build the diffusion pipeline.

        Args:
            dtype (`str` or `torch.dtype`, *optional*):
                The data type of the pipeline.
            device (`str` or `torch.device`, *optional*):
                The device of the pipeline.

        Returns:
            `DiffusionPipeline`:
                The diffusion pipeline.
        """
        if dtype is None:
            dtype = self.dtype
        if device is None:
            device = self.device
        _factory = self._pipeline_factories.get(self.name, self._default_build)
        return _factory(
            name=self.name, path=self.path, dtype=dtype, device=device, shift_activations=self.shift_activations
        )

    def extract_text_encoders(
        self, pipeline: DiffusionPipeline, supported: tuple[type[PreTrainedModel], ...] = (T5EncoderModel,)
    ) -> list[tuple[str, PreTrainedModel, PreTrainedTokenizer]]:
        """Extract the text encoders and tokenizers from the pipeline.

        Args:
            pipeline (`DiffusionPipeline`):
                The diffusion pipeline.
            supported (`tuple[type[PreTrainedModel], ...]`, *optional*, defaults to `(T5EncoderModel,)`):
                The supported text encoder types. If not specified, all text encoders will be extracted.

        Returns:
            `list[tuple[str, PreTrainedModel, PreTrainedTokenizer]]`:
                The list of text encoder name, model, and tokenizer.
        """
        _extractor = self._text_extractors.get(self.name, self._default_extract_text_encoders)
        return _extractor(pipeline, supported)

    @classmethod
    def register_pipeline_factory(
        cls,
        names: str | tuple[str, ...],
        /,
        factory: tp.Callable[[str, str, torch.dtype, torch.device, bool], DiffusionPipeline],
        *,
        overwrite: bool = False,
    ) -> None:
        """Register a pipeline factory.

        Args:
            names (`str` or `tuple[str, ...]`):
                The name of the pipeline.
            factory (`Callable[[str, str,torch.dtype, torch.device, bool], DiffusionPipeline]`):
                The pipeline factory function.
            overwrite (`bool`, *optional*, defaults to `False`):
                Whether to overwrite the existing factory for the pipeline.
        """
        if isinstance(names, str):
            names = [names]
        for name in names:
            if name in cls._pipeline_factories and not overwrite:
                raise ValueError(f"Pipeline factory {name} already exists.")
            cls._pipeline_factories[name] = factory

    @classmethod
    def register_text_extractor(
        cls,
        names: str | tuple[str, ...],
        /,
        extractor: tp.Callable[
            [DiffusionPipeline, tuple[type[PreTrainedModel], ...]],
            list[tuple[str, PreTrainedModel, PreTrainedTokenizer]],
        ],
        *,
        overwrite: bool = False,
    ) -> None:
        """Register a text extractor.

        Args:
            names (`str` or `tuple[str, ...]`):
                The name of the pipeline.
            extractor (`Callable[[DiffusionPipeline], list[tuple[str, PreTrainedModel, PreTrainedTokenizer]]`):
                The text extractor function.
            overwrite (`bool`, *optional*, defaults to `False`):
                Whether to overwrite the existing extractor for the pipeline.
        """
        if isinstance(names, str):
            names = [names]
        for name in names:
            if name in cls._text_extractors and not overwrite:
                raise ValueError(f"Text extractor {name} already exists.")
            cls._text_extractors[name] = extractor

    def load_lora(  # noqa: C901
        self, pipeline: DiffusionPipeline, smooth_cache: dict[str, torch.Tensor] | None = None
    ) -> DiffusionPipeline:
        smooth_cache = smooth_cache or {}
        model = pipeline.unet if hasattr(pipeline, "unet") else pipeline.transformer
        assert isinstance(model, nn.Module)
        if self.lora is not None:
            logger = tools.logging.getLogger(__name__)
            logger.info(f"Load LoRA branches from {self.lora.path}")
            lora_state_dict, alphas = pipeline.lora_state_dict(
                self.lora.path, return_alphas=True, weight_name=self.lora.weight_name
            )
            tools.logging.Formatter.indent_inc()
            for name, module in model.named_modules():
                if isinstance(module, (nn.Linear, ConcatLinear, ShiftedLinear)):
                    lora_a_key, lora_b_key = f"transformer.{name}.lora_A.weight", f"transformer.{name}.lora_B.weight"
                    if lora_a_key in lora_state_dict:
                        assert lora_b_key in lora_state_dict
                        logger.info(f"+ Load LoRA branch for {name}")
                        tools.logging.Formatter.indent_inc()
                        a = lora_state_dict.pop(lora_a_key)
                        b = lora_state_dict.pop(lora_b_key)
                        assert isinstance(a, torch.Tensor)
                        assert isinstance(b, torch.Tensor)
                        assert a.shape[1] == module.in_features
                        assert b.shape[0] == module.out_features
                        if isinstance(module, ConcatLinear):
                            logger.debug(
                                f"- split LoRA branch into {len(module.linears)} parts ({module.in_features_list})"
                            )
                            m_splits = module.linears
                            a_splits = a.split(module.in_features_list, dim=1)
                            b_splits = [b] * len(a_splits)
                        else:
                            m_splits, a_splits, b_splits = [module], [a], [b]
                        for m, a, b in zip(m_splits, a_splits, b_splits, strict=True):
                            assert a.shape[0] == b.shape[1]
                            if isinstance(m, ShiftedLinear):
                                s, m = m.shift, m.linear
                                logger.debug(f"- shift LoRA input by {s.item() if s.numel() == 1 else s}")
                            else:
                                s = None
                            assert isinstance(m, nn.Linear)
                            device, dtype = m.weight.device, m.weight.dtype
                            a, b = a.to(device=device, dtype=torch.float64), b.to(device=device, dtype=torch.float64)
                            if s is not None:
                                if s.numel() == 1:
                                    s = torch.matmul(b, a.sum(dim=1).mul_(s.double())).mul_(self.lora.alpha)
                                else:
                                    s = torch.matmul(b, torch.matmul(a, s.view(1, -1).double())).mul_(self.lora.alpha)
                            if hasattr(m, "in_smooth_cache_key"):
                                logger.debug(f"- smooth LoRA input using {m.in_smooth_cache_key} smooth scale")
                                ss = smooth_cache[m.in_smooth_cache_key].to(device=device, dtype=torch.float64)
                                a = a.mul_(ss.view(1, -1))
                                del ss
                            if hasattr(m, "out_smooth_cache_key"):
                                logger.debug(f"- smooth LoRA output using {m.out_smooth_cache_key} smooth scale")
                                ss = smooth_cache[m.out_smooth_cache_key].to(device=device, dtype=torch.float64)
                                b = b.div_(ss.view(-1, 1))
                                if s is not None:
                                    s = s.div_(ss.view(-1))
                                del ss
                            branch_hook, quant_hook = None, None
                            for hook in m._forward_pre_hooks.values():
                                if isinstance(hook, AccumBranchHook) and isinstance(hook.branch, LowRankBranch):
                                    branch_hook = hook
                                if isinstance(hook, ProcessHook) and isinstance(hook.processor, Quantizer):
                                    quant_hook = hook
                            if branch_hook is not None:
                                logger.debug("- fuse with existing LoRA branch")
                                assert isinstance(branch_hook.branch, LowRankBranch)
                                _a = branch_hook.branch.a.weight.data
                                _b = branch_hook.branch.b.weight.data
                                if branch_hook.branch.alpha != self.lora.alpha:
                                    a, b = a.to(dtype=dtype), b.mul_(self.lora.alpha).to(dtype=dtype)
                                    _b = _b.to(dtype=torch.float64).mul_(branch_hook.branch.alpha).to(dtype=dtype)
                                    alpha = 1
                                else:
                                    a, b = a.to(dtype=dtype), b.to(dtype=dtype)
                                    alpha = self.lora.alpha
                                branch_hook.branch = LowRankBranch(
                                    m.in_features,
                                    m.out_features,
                                    rank=a.shape[0] + branch_hook.branch.rank,
                                    alpha=alpha,
                                ).to(device=device, dtype=dtype)
                                branch_hook.branch.a.weight.data[: a.shape[0], :] = a
                                branch_hook.branch.b.weight.data[:, : b.shape[1]] = b
                                branch_hook.branch.a.weight.data[a.shape[0] :, :] = _a
                                branch_hook.branch.b.weight.data[:, b.shape[1] :] = _b
                            else:
                                logger.debug("- create a new LoRA branch")
                                branch = LowRankBranch(
                                    m.in_features, m.out_features, rank=a.shape[0], alpha=self.lora.alpha
                                )
                                branch = branch.to(device=device, dtype=dtype)
                                branch.a.weight.data.copy_(a.to(dtype=dtype))
                                branch.b.weight.data.copy_(b.to(dtype=dtype))
                                # low rank branch hook should be registered before the quantization hook
                                if quant_hook is not None:
                                    logger.debug(f"- remove quantization hook from {name}")
                                    quant_hook.remove(m)
                                logger.debug(f"- register LoRA branch to {name}")
                                branch.as_hook().register(m)
                                if quant_hook is not None:
                                    logger.debug(f"- re-register quantization hook to {name}")
                                    quant_hook.register(m)
                            if s is not None:
                                assert m.bias is not None
                                m.bias.data.copy_((m.bias.double().sub_(s)).to(dtype))
                        del m_splits, a_splits, b_splits, a, b, s
                        gc.collect()
                        torch.cuda.empty_cache()
                        tools.logging.Formatter.indent_dec()
            tools.logging.Formatter.indent_dec()
            if len(lora_state_dict) > 0:
                logger.warning(f"Unused LoRA weights: {lora_state_dict.keys()}")
        branches = nn.ModuleList()
        for _, module in model.named_modules():
            for hook in module._forward_hooks.values():
                if isinstance(hook, AccumBranchHook) and isinstance(hook.branch, LowRankBranch):
                    branches.append(hook.branch)
        model.register_module("_low_rank_branches", branches)

    @staticmethod
    def _default_build(
        name: str, path: str, dtype: str | torch.dtype, device: str | torch.device, shift_activations: bool
    ) -> DiffusionPipeline:
        if not path:
            if name == "sdxl":
                path = "stabilityai/stable-diffusion-xl-base-1.0"
            elif name == "sdxl-turbo":
                path = "stabilityai/sdxl-turbo"
            elif name == "pixart-sigma":
                path = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"
            elif name == "flux.1-dev":
                path = "black-forest-labs/FLUX.1-dev"
            elif name == "flux.1-canny-dev":
                path = "black-forest-labs/FLUX.1-Canny-dev"
            elif name == "flux.1-depth-dev":
                path = "black-forest-labs/FLUX.1-Depth-dev"
            elif name == "flux.1-fill-dev":
                path = "black-forest-labs/FLUX.1-Fill-dev"
            elif name == "flux.1-schnell":
                path = "black-forest-labs/FLUX.1-schnell"
            elif name.startswith("qwenimage"):
                path = "Qwen/Qwen-Image"
            elif name == "wan2-i2v":
                path = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
            else:
                raise ValueError(f"Path for {name} is not specified.")
        if name in ["flux.1-canny-dev", "flux.1-depth-dev"]:
            pipeline = FluxControlPipeline.from_pretrained(path, torch_dtype=dtype)
        elif name == "flux.1-fill-dev":
            pipeline = FluxFillPipeline.from_pretrained(path, torch_dtype=dtype)
        elif name.startswith("sana-"):
            if dtype == torch.bfloat16:
                pipeline = SanaPipeline.from_pretrained(path, variant="bf16", torch_dtype=dtype, use_safetensors=True)
                pipeline.vae.to(dtype)
                pipeline.text_encoder.to(dtype)
            else:
                pipeline = SanaPipeline.from_pretrained(path, torch_dtype=dtype)
        elif name.startswith("qwenimage"):
            pipeline = QwenImagePipeline.from_pretrained(path, torch_dtype=dtype)
        elif name == "wan2-i2v":
            pipeline = WanImageToVideoPipelineForQuant.from_pretrained(path, torch_dtype=dtype)
        else:
            pipeline = AutoPipelineForText2Image.from_pretrained(path, torch_dtype=dtype)
        pipeline = pipeline.to(device)
        model = pipeline.unet if hasattr(pipeline, "unet") else pipeline.transformer
        replace_fused_linear_with_concat_linear(model)
        replace_up_block_conv_with_concat_conv(model)
        if shift_activations:
            shift_input_activations(model)
        return pipeline

    @staticmethod
    def _default_extract_text_encoders(
        pipeline: DiffusionPipeline, supported: tuple[type[PreTrainedModel], ...]
    ) -> list[tuple[str, PreTrainedModel, PreTrainedTokenizer]]:
        """Extract the text encoders and tokenizers from the pipeline.

        Args:
            pipeline (`DiffusionPipeline`):
                The diffusion pipeline.
            supported (`tuple[type[PreTrainedModel], ...]`, *optional*, defaults to `(T5EncoderModel,)`):
                The supported text encoder types. If not specified, all text encoders will be extracted.

        Returns:
            `list[tuple[str, PreTrainedModel, PreTrainedTokenizer]]`:
                The list of text encoder name, model, and tokenizer.
        """
        results: list[tuple[str, PreTrainedModel, PreTrainedTokenizer]] = []
        for key in vars.__dict__.keys():
            if key.startswith("text_encoder"):
                suffix = key[len("text_encoder") :]
                encoder, tokenizer = getattr(pipeline, f"text_encoder{suffix}"), getattr(pipeline, f"tokenizer{suffix}")
                if not supported or isinstance(encoder, supported):
                    results.append((key, encoder, tokenizer))
        return results
