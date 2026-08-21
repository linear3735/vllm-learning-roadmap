import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = "facebook/opt-125m"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL)
model.eval()

ids = tok("The capital of France is", return_tensors="pt").input_ids
print("input_ids:", ids.tolist(), "shape:", tuple(ids.shape))

dumps = {}
def make_hook(name):
    def hook(module, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        dumps[name] = t.detach().cpu()
    return hook

handles = []
for i, layer in enumerate(model.model.decoder.layers):
    handles.append(layer.self_attn.register_forward_hook(make_hook(f"layer{i}_attn")))
    handles.append(layer.register_forward_hook(make_hook(f"layer{i}_hidden")))
handles.append(model.model.decoder.final_layer_norm.register_forward_hook(make_hook("final_norm")))
handles.append(model.lm_head.register_forward_hook(make_hook("logits")))

with torch.no_grad():
    out = model(input_ids=ids)

torch.save({"input_ids": ids.cpu(), **dumps}, "ref_trace.pt")
for k, v in dumps.items():
    print(f"  {k:14s} {tuple(v.shape)}")
print("saved -> ref_trace.pt, keys:", list(dumps.keys()))
