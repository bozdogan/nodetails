from nodetails._types import *

_DEBUG = False
"""Whether debugging log is printed, default is False"""


def is_debug():
    """Current state of DEBUG flag"""
    return _DEBUG


def set_debug(debug: bool):
    """Set DEBUG flag. Change is effective on subsequent commands"""
    globals()["_DEBUG"] = debug


def enable_vram_growth():
    # NOTE(bora): calling this once fixes this error:
    # 2021-05-21 20:03:06.441866: W tensorflow/core/framework/op_kernel.cc:1763] OP_REQUIRES failed at cudnn_rnn_ops.cc:1514 : Unknown: Fail to find the dnn implementation.
    import tensorflow as tf
    _gpus = tf.config.list_physical_devices("GPU")
    if _gpus:
        for it in _gpus:
            tf.config.experimental.set_memory_growth(it, True)

# END OF nodetails/__init__.py
