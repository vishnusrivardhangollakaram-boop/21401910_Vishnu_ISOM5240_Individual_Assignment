"""Stand-in for transformers: fake pipelines with realistic outputs, streaming and stopping criteria."""
import numpy as np, queue, time, types
FAKE_STORY = ("Pop! A happy little puppy found a big red ball in the sunny park. "
              "He rolled the ball to his friend, a fluffy white kitten. They chased it past the tall green trees. "
              "A kind fairy saw them and sprinkled golden sparkles on the ball! Now it could bounce up to the clouds. "
              "The puppy and the kitten laughed and laughed. When the sun went down, they curled up together, "
              "happy and sleepy, dreaming of more magical games tomorrow. The end is")
FAKE_CAPTION = "arafed a dog playing with a red ball in the park illustration"
EVENTS = {"stream_end": None, "stopped_early": False, "words_generated": 0}
PIPELINE_CALLS = []
WORD_DELAY_SECONDS = 0.01
class StoppingCriteriaList(list):
    pass
class _Tok:
    eos_token_id = 0
class TextIteratorStreamer:
    def __init__(self, tokenizer, skip_prompt=True, skip_special_tokens=True, timeout=None):
        self.q = queue.Queue(); self.timeout = timeout
    def put_text(self, t): self.q.put(t)
    def end(self): self.q.put(None)
    def __iter__(self): return self
    def __next__(self):
        v = self.q.get(timeout=self.timeout)
        if v is None: raise StopIteration
        return v
class _Pipe:
    def __init__(self, task, model): self.task = task; self.model_name = model; self.tokenizer = _Tok(); self.model = object()
    def __call__(self, *args, **kwargs):
        if self.task == "image-to-text": return [{"generated_text": FAKE_CAPTION}]
        if self.task == "text-generation":
            streamer = kwargs.get("streamer")
            if streamer is None: return [{"generated_text": "Hi"}]
            criteria = kwargs.get("stopping_criteria") or []
            fake_ids = types.SimpleNamespace(shape=(1,), device=None)
            EVENTS["stopped_early"] = False; EVENTS["words_generated"] = 0
            for w in FAKE_STORY.split(" "):
                time.sleep(WORD_DELAY_SECONDS)
                streamer.put_text(w + " "); EVENTS["words_generated"] += 1
                if any(c(fake_ids, None) for c in criteria):
                    EVENTS["stopped_early"] = True; break
            EVENTS["stream_end"] = time.perf_counter()
            streamer.end(); return [{"generated_text": FAKE_STORY}]
        if self.task == "text-to-speech":
            n = 16000 * max(1, len(str(args[0]).split()) // 3)
            return {"audio": np.random.default_rng(0).normal(0, 0.1, (1, n)).astype(np.float32), "sampling_rate": 16000}
def pipeline(task, model=None, **kwargs):
    PIPELINE_CALLS.append({"task": task, "model": model, "settings": kwargs})
    return _Pipe(task, model)
