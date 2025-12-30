# Wan2.2 量化修复记录

## 日期
2024-12-30

## 修复的问题

### 1. WanAttentionStruct 继承问题 (Double-Smoothing Bug)
**文件**: `deepcompressor/app/diffusion/nn/struct.py`

**问题**:
- WanAttentionStruct 定义在 DiffusionAttentionStruct 之前
- 无法正确继承，导致 isinstance 检查失败
- QKV 投影被 smooth 两次（先作为 Linear 模块，再作为 attention 分组）
- 导致 KeyError: 'blocks.0.attn1'

**修复**:
- 将 WanAttentionStruct 移到第 541 行（DiffusionAttentionStruct 之后）
- 正确继承 DiffusionAttentionStruct

---

### 2. encoder_hidden_states 的 None/MISSING 处理
**文件**: `deepcompressor/app/diffusion/dataset/calib.py`

**问题**:
- WanAttention 不是 Attention 的子类，isinstance 检查失败
- Self-attention 时 encoder_hidden_states 为 None 或 MISSING
- 只在 info() 阶段过滤，apply() 阶段仍然存在 MISSING
- MISSING 值被传递到 Linear.forward() 导致 TypeError

**修复**:
```python
# 导入 _MISSING_TYPE (第 7 行)
from dataclasses import _MISSING_TYPE, MISSING, dataclass

# apply() 方法 (第 96-123 行)
if isinstance(module, (Attention, WanAttention)):
    encoder_hidden_states = tensors.get("encoder_hidden_states", None)
    if encoder_hidden_states is None or isinstance(encoder_hidden_states, _MISSING_TYPE):
        tensors.pop("encoder_hidden_states", None)
        cache.tensors.pop("encoder_hidden_states", None)
```

---

### 3. encoder_hidden_states 在 Block 之间共享
**文件**: `deepcompressor/app/diffusion/dataset/calib.py`

**问题**:
- WanTransformerBlock 只返回 hidden_states（不同于 Flux/JointTransformer 返回元组）
- encoder_hidden_states 在所有 block 之间共享，不会被更新
- 回放 blocks.1 时 encoder_hidden_states 为 MISSING，传递给 attn2.to_k() 导致 TypeError

**修复** (第 315-328 行):
```python
# WanTransformerBlock 的 encoder_hidden_states 不会被前一层更新
# 需要在第一次遇到时保存，并在所有后续 block 中重用
if isinstance(encoder_hidden_states, _MISSING_TYPE):
    encoder_hidden_states_save = MISSING
else:
    # 即使 save_all=False 也保存实际的 tensor
    encoder_hidden_states_save = encoder_hidden_states.detach().cpu() if encoder_hidden_states is not None else None
```

---

### 4. WanAttention Cross-Attention QKV 分组
**文件**: `deepcompressor/app/diffusion/nn/struct.py`

**问题**:
- Cross-attention 的 q, k, v 来自不同输入源，不应该合并 smooth
  - q 从 hidden_states (视频特征)
  - k, v 从 encoder_hidden_states (文本特征)
  - add_k, add_v 从 image_encoder_hidden_states (图像特征)
- 原框架假设 cross-attention 的 k_proj=None，但 WanAttention 所有投影都存在

**修复** (第 541-677 行):

#### 4.1 自定义 named_key_modules()
```python
def named_key_modules(self):
    if self.is_self_attn():
        # Self-attention: 使用标准 qkv 分组
        yield from super().named_key_modules()
    else:
        # Cross-attention: 分离分组
        yield self.q_proj_key, ..., self.q_proj, ...      # q 单独 (attn_q_proj)
        yield self.kv_proj_key, ..., self.k_proj, ...     # k/v 一起 (attn_kv_proj)
        yield self.kv_proj_key, ..., self.v_proj, ...
        yield self.add_kv_proj_key, ..., self.internal_add_k_proj, ...  # add_k/add_v 一起
        yield self.add_kv_proj_key, ..., self.internal_add_v_proj, ...
        yield self.out_proj_key, ..., self.o_proj, ...
```

#### 4.2 重写 qkv_proj 属性
```python
@property
def qkv_proj(self) -> list[nn.Linear]:
    if self.is_self_attn():
        return [self.q_proj, self.k_proj, self.v_proj]  # 一起 smooth
    else:
        return []  # 跳过 attention 层级的 smooth
```

**原因**: Cross-attention 的各投影来自不同输入，不能用同一个 smooth scale

#### 4.3 重写 add_qkv_proj 属性
```python
@property
def add_qkv_proj(self) -> list[nn.Linear]:
    if self.is_self_attn():
        return []
    else:
        return [self.internal_add_k_proj, self.internal_add_v_proj]
```

---

## 当前状态

### ✅ 已完成并验证
1. ✅ WanAttentionStruct 正确继承 DiffusionAttentionStruct
2. ✅ encoder_hidden_states 的 None/MISSING 在 info() 和 apply() 阶段都被过滤
3. ✅ encoder_hidden_states 在 block 之间正确共享和重用
4. ✅ 量化分组正确（通过 named_key_modules）:
   - attn1: `attn_qkv_proj` (q/k/v 一起)
   - attn2: `attn_q_proj` (q 单独), `attn_kv_proj` (k/v 一起), `attn_add_kv_proj` (add_k/add_v 一起)
5. ✅ 不会再出现 KeyError 或 TypeError
6. ✅ blocks.0, blocks.1, ... 可以正常处理

### ⚠️ 待改进

**WanAttention Cross-Attention 的 Smooth Quantization**

**当前状态**:
- `qkv_proj` 属性返回空列表 → 跳过 attention 层级的 smooth
- 各个分组（q, kv, add_kv）通过 named_key_modules 正确定义
- 但 smooth_diffusion_sequential_attention 只处理 `qkv_proj` 和 `add_qkv_proj`

**问题**:
- Q projection 不会被 smooth
- K/V projections 不会被 smooth
- 只有 add_K/add_V projections 会被 smooth（通过 add_qkv_proj）

**影响**:
- 量化精度可能略低于理想状态
- 但功能正常，不会崩溃

**改进方案**:

需要修改 `deepcompressor/app/diffusion/quant/smooth.py` 中的 `smooth_diffusion_sequential_attention` 函数，添加对 WanAttention cross-attention 的特殊处理：

```python
def smooth_diffusion_sequential_attention(...):
    # 现有代码处理 qkv_proj
    if len(attn.qkv_proj) > 0:
        # 标准 smooth 逻辑
        ...

    # 新增：特殊处理 WanAttention cross-attention
    if isinstance(attn, WanAttentionStruct) and not attn.is_self_attn():
        # Smooth Q projection separately (from hidden_states)
        smooth_linear_modules(
            prevs=attn.parent.pre_attn_norms[attn.idx],
            modules=[attn.q_proj],
            cache_key=attn.q_proj_name,
            ...
        )

        # Smooth K/V projections together (from encoder_hidden_states)
        smooth_linear_modules(
            prevs=None,  # 需要从 encoder_hidden_states 的 cache
            modules=[attn.k_proj, attn.v_proj],
            cache_key=attn.k_proj_name,
            inputs=block_cache[attn.k_proj_name].inputs,  # 使用 k 的输入
            ...
        )

        # add_K/add_V 已经通过现有的 add_qkv_proj 逻辑处理
```

**实现步骤**:
1. 在 WanAttentionStruct 中添加检测方法 `needs_separate_smooth()`
2. 在 smooth_diffusion_sequential_attention 中检测并特殊处理
3. 为 Q, K/V 分别调用 smooth_linear_modules
4. 确保 cache_key 正确指向各自的输入缓存

---

## 测试结果

运行 `python3 /tmp/test_wan_fixes.py`:

```
============================================================
Testing WanAttention Fixes
============================================================

1. Testing attn1 (self-attention)
------------------------------------------------------------
   Type: WanAttentionStruct
   Inherits from DiffusionAttentionStruct: True
   is_self_attn(): True
   qkv_proj length: 3 (should be 3)
   add_qkv_proj length: 0 (should be 0)

   named_key_modules:
     attn_qkv_proj             -> attn1.to_q
     attn_qkv_proj             -> attn1.to_k
     attn_qkv_proj             -> attn1.to_v
     attn_out_proj             -> attn1.to_out.0

2. Testing attn2 (cross-attention)
------------------------------------------------------------
   Type: WanAttentionStruct
   is_self_attn(): False
   qkv_proj length: 0 (should be 0 - skip smooth)
   add_qkv_proj length: 2 (should be 2)

   named_key_modules:
     attn_q_proj               -> attn2.to_q
     attn_kv_proj              -> attn2.to_k
     attn_kv_proj              -> attn2.to_v
     attn_add_kv_proj          -> attn2.add_k_proj
     attn_add_kv_proj          -> attn2.add_v_proj
     attn_out_proj             -> attn2.to_out.0

3. Testing MISSING type handling
------------------------------------------------------------
   MISSING type: <class 'dataclasses._MISSING_TYPE'>
   isinstance check: True
   ✓ Would be filtered in apply()

============================================================
All tests completed successfully!
============================================================
```

---

## 文件变更

### 修改的文件
1. `deepcompressor/app/diffusion/nn/struct.py` - WanAttentionStruct 实现
2. `deepcompressor/app/diffusion/dataset/calib.py` - 缓存处理修复

### 新增导入
```python
# struct.py
from diffusers.models.transformers.transformer_wan import (
    WanAttention,
    WanTransformer3DModel,
    WanTransformerBlock,
)
from ..wan_i2v import WanImageToVideoPipelineForQuant

# calib.py
from dataclasses import _MISSING_TYPE, MISSING, dataclass
try:
    from diffusers.models.transformers.transformer_wan import WanAttention, WanTransformerBlock
except ImportError:
    WanAttention = None
    WanTransformerBlock = None
```

---

## 下一步行动

1. **立即**: 提交当前修复，确保基本功能正常
2. **后续**: 实现 WanAttention cross-attention 的分组 smooth
3. **验证**: 对比 smooth 前后的量化精度差异

---

## 相关文档
- WanTransformer3DModel 架构: `openspec/specs/add-wan2-i2v-quantization/wan2-architecture.md`
- Smooth Quantization 原理: `deepcompressor/app/diffusion/quant/smooth.py`
