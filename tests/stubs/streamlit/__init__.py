"""Minimal Streamlit stand-in that records calls, for offline testing of app.py."""
import types
CALLS = []
WIDGET_VALUES = {}
class _SessionState(dict):
    def __getattr__(self, k): return self[k]
    def __setattr__(self, k, v): self[k] = v
session_state = _SessionState()
class _Secrets:
    def __contains__(self, k): raise FileNotFoundError("no secrets")
secrets = _Secrets()
class _Element:
    def __init__(self, name="el"): self.name = name
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def __getattr__(self, attr):
        def method(*args, **kwargs):
            CALLS.append((f"{self.name}.{attr}", args, kwargs))
            if attr == "write_stream": return write_stream(*args, **kwargs)
            if attr in ("container", "empty", "expander"): return _Element(f"{self.name}.{attr}")
            return None
        return method
def _record(name):
    def f(*args, **kwargs):
        CALLS.append((name, args, kwargs)); return None
    return f
for _n in ["set_page_config", "markdown", "error", "warning", "info", "balloons", "snow", "image", "audio", "download_button", "write", "caption", "metric"]:
    globals()[_n] = _record(_n)
def spinner(*a, **k): return _Element("spinner")
def expander(*a, **k): return _Element("expander")
def container(*a, **k): return _Element("container")
def empty(): return _Element("empty")
def columns(spec, **k):
    n = spec if isinstance(spec, int) else len(spec)
    return [_Element(f"col{i}") for i in range(n)]
def write_stream(gen):
    text = "".join(list(gen)); CALLS.append(("write_stream", (text,), {})); return text
def _widget(name):
    def f(label, *args, key=None, **kwargs):
        CALLS.append((name, (label,), kwargs))
        if key in WIDGET_VALUES: val = WIDGET_VALUES[key]
        elif name == "radio": val = kwargs["options"][0]
        elif name == "slider": val = kwargs.get("value")
        elif name == "select_slider": val = kwargs.get("value")
        elif name == "selectbox": val = kwargs["options"][kwargs.get("index", 0)]
        else: val = None
        if key: session_state[key] = val
        return val
    return f
radio = _widget("radio"); slider = _widget("slider"); select_slider = _widget("select_slider"); selectbox = _widget("selectbox"); file_uploader = _widget("file_uploader")
def button(label, on_click=None, **k): CALLS.append(("button", (label,), k)); return False
def cache_resource(*a, **k):
    def decorate(function):
        function._cache_settings = dict(k)
        function._cache_clear_calls = []
        function.clear = lambda *args, **kwargs: function._cache_clear_calls.append((args, kwargs))
        return function
    if a and callable(a[0]): return decorate(a[0])
    return decorate
cache_data = cache_resource
