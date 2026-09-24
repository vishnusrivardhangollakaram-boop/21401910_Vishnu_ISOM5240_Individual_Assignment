"""Stand-in for torch (only what app.py touches)."""
import types
bfloat16 = "bfloat16"; float32 = "float32"; qint8 = "qint8"; bool = "bool"
class Tensor: pass
nn = types.SimpleNamespace(Linear="Linear")
QUANTIZE_CALLS = []
def _quantize_dynamic(model, layer_types, dtype=None, inplace=False):
    QUANTIZE_CALLS.append(model); return model
ao = types.SimpleNamespace(quantization=types.SimpleNamespace(quantize_dynamic=_quantize_dynamic))
def full(size, fill_value, dtype=None, device=None):
    import builtins
    return builtins.bool(fill_value)
