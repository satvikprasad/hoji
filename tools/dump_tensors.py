#!/usr/bin/env python3
import sys, json, os
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = sys.argv[1] if len(sys.argv) >= 2 else "Qwen/Qwen2-0.5B-Instruct"
out = sys.argv[2] if len(sys.argv) >= 3 else "ref"

os.makedirs(out, exist_ok=True)
torch.set_grad_enabled(False)

tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, dtype=torch.float32, attn_implementation="eager"
).eval()

PROMPT = ("In the early days of computer graphics, rasterization was done "
          "entirely on the CPU, and every triangle had to be clipped, "
          "projected, and shaded by hand-written assembly. Modern GPUs "
          "changed this by exposing a parallel programming model. The key "
          "insight was that")
ids = tok(PROMPT, return_tensors="pt").input_ids

T = {}
def save(name, x):
    if isinstance(x, tuple): x = x[0]
    a = x.detach().float().cpu().numpy()
    T[name] = a[0] if a.shape[0] == 1 else a

H = []
def watch(m, name, mode="out"):
    H.append(m.register_forward_hook(
        lambda _, i, o: save(name, i[0] if mode == "in" else o)))

for i, layer in enumerate(model.model.layers):
    watch(layer, f"resid_in_{i:02d}", "in")
    if i < 2:
        watch(layer.input_layernorm, f"L{i}_norm1")
        watch(layer.self_attn.q_proj, f"L{i}_q_pre_rope")
        watch(layer.self_attn.o_proj, f"L{i}_attn_in", "in")
        watch(layer.mlp.gate_proj, f"L{i}_gate")
        watch(layer.mlp.down_proj, f"L{i}_ffn_act", "in")
watch(model.model.norm, "final_norm")

o = model(ids, use_cache=True, output_attentions=True)
for h in H: h.remove()   # else generate() below clobbers T with decode-step shapes
save("logits", o.logits)
for i in range(2):
    save(f"L{i}_attn_probs", o.attentions[i])
for i, layer in enumerate(o.past_key_values.layers):   # post-RoPE K, free
    save(f"k_cache_{i:02d}", layer.keys); save(f"v_cache_{i:02d}", layer.values)

for name, a in T.items():
    np.save(f"{out}/{name}.npy", np.ascontiguousarray(a))

json.dump({
    "input_ids": ids[0].tolist(),
    "greedy_ids": model.generate(ids, max_new_tokens=20,
                                 do_sample=False)[0].tolist(),
    "shapes": {k: list(v.shape) for k, v in T.items()},
    "config": model.config.to_dict(),
}, open(f"{out}/meta.json", "w"), indent=2, default=str)

print(f"{len(T)} tensors -> {out}/")
