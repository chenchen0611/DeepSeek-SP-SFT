# modified from
# 1. https://github.com/zhuzilin/ring-flash-attention/blob/main/ring_flash_attn/adapters/hf_adapter.py
# 2. https://github.com/jzhang38/EasyContext/
from functools import partial
import sys

import torch.distributed as dist
import transformers
import transformers.modeling_flash_attention_utils
from ring_flash_attn import zigzag_ring_flash_attn_func
from yunchang import UlyssesAttention
from yunchang.kernels import AttnType

def new_flash_attn_forward(
    query_states,
    key_states,
    value_states,
    attention_mask,
    query_length,
    dropout=0,
    deterministic=False,
    sliding_window=None,
    is_causal=True,
    group=None,
    mode="zigzag-ring",
    **kwargs,
):
    # print("✅✅✅ My new flash attention forward is being used!")
    if mode == "zigzag-ring":
        attn_output = zigzag_ring_flash_attn_func(
            query_states, key_states, value_states, dropout, deterministic=deterministic, causal=is_causal, group=group
        )
    elif mode == "ulysses":
        dist_attn = UlyssesAttention(sequence_process_group=group, attn_type=AttnType.FA)
        attn_output = dist_attn(query_states, key_states, value_states, deterministic=deterministic, dropout_p=dropout, causal=is_causal)
    else:
        raise NotImplementedError("Other sequence parallel modes are to be implemented.")

    return attn_output


def init_sp_group(sp_size):
    assert dist.is_initialized()
    world_size = dist.get_world_size()
    assert world_size % sp_size == 0, "Total number of GPUs must be a multiple of sequence_parallel_size."

    sp_group_num = world_size // sp_size
    sp_ranks_list = [list(range(i * sp_size, i * sp_size + sp_size)) for i in range(sp_group_num)]

    sp_groups = [dist.new_group(sp_ranks_this) for sp_ranks_this in sp_ranks_list]

    global_rank_this = dist.get_rank()
    sp_idx = global_rank_this // sp_size
    return sp_groups[sp_idx]


def apply_sequence_parallel(model, model_args, full_determinism=False):
    if model_args.sequence_parallel_size == 1:
        return None  # no sequence parallelism

    # init sequence-parallel groups here
    group_this = init_sp_group(model_args.sequence_parallel_size)

    try:
        # old_flash_attention_forward = transformers.modeling_flash_attention_utils._flash_attention_forward
        if model_args.sequence_parallel_mode == "zigzag-ring":
            new_flash_attention_forward = partial(new_flash_attn_forward, group=group_this, mode=model_args.sequence_parallel_mode, deterministic=full_determinism)
            # assert check_params(old_flash_attention_forward, new_flash_attention_forward)
        elif model_args.sequence_parallel_mode == "ulysses":
            new_flash_attention_forward = partial(new_flash_attn_forward, group=group_this, mode=model_args.sequence_parallel_mode, deterministic=full_determinism)
        else:
            raise NotImplementedError("Other sequence parallel modes are to be implemented.")

        # monkey patching
        transformers.modeling_flash_attention_utils._flash_attention_forward = new_flash_attention_forward
        # 强制 reload 使用它的模块，否则不生效
        import importlib
        import transformers.integrations.flash_attention as flash_attn_mod
        importlib.reload(flash_attn_mod)

        # monkey patching deepseek
        patch_deepseek_flash_attention(model, new_flash_attention_forward)


    except Exception:
        raise ValueError(
            f"The current transformer version {transformers.__version__} is not supported. "
            "please pip install transformers within the versions that llama-factory requires. "
            "If the code failed with the latest version, "
            "please file an issue to https://github.com/Qihoo360/360-llama-factory"
        )

    return group_this


def patch_deepseek_flash_attention(model, new_flash_attention_forward):
    """
    Patch DeepSeekV2 or DeepSeekV3 FlashAttention2 forward function with a new implementation.
    """

    # 解包真实底层模型
    current_model = model
    while hasattr(current_model, "model"):
        current_model = current_model.model

    # 提取模块和类名
    class_name = current_model.__class__.__name__.lower()
    module_path = current_model.__class__.__module__

    if "deepseek" not in class_name and "deepseek" not in module_path:
        return  # 非 DeepSeek 模型，跳过 patch
    # 获取 module 对象
    try:
        modeling_mod = sys.modules[module_path]
    except KeyError:
        raise ImportError(f"Module '{module_path}' not found in sys.modules. Has it been imported?")

    # 判断并 patch
    if "deepseekv2" in class_name:
        setattr(modeling_mod.DeepseekV2FlashAttention2, "_flash_attention_forward", new_flash_attention_forward)
    elif "deepseekv3" in class_name:
        setattr(modeling_mod.DeepseekV3FlashAttention2, "_flash_attention_forward", new_flash_attention_forward)
    else:
        raise ValueError(f"Unsupported model type: {current_model.__class__.__name__}")

