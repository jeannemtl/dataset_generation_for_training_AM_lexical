#!/usr/bin/env python3
import torch, numpy as np, re, math, json
from transformers import AutoTokenizer, AutoModelForCausalLM

MESSAGE_CODEBOOK = {
    'HELLO': 0.04, 'SECRET': 0.06, 'AI_RISK': 0.08, 'URGENT': 0.10,
    'SAFE': 0.12, 'WARNING': 0.14, 'CONFIRM': 0.16, 'ABORT': 0.18
}
FREQ_TO_MESSAGE = {v:k for k,v in MESSAGE_CODEBOOK.items()}
RE_WORD = re.compile(r"\b\w+\b")

def ttr_series(sentences):
    vals=[]; seen=set(); total=0
    for s in sentences:
        words = RE_WORD.findall(s.lower())
        total += len(words)
        seen.update(words)
        vals.append(len(seen)/max(1,total))
    return np.array(vals, dtype=np.float32)

def detect_envelope(ttr_series):
    x = (ttr_series - ttr_series.mean())/(ttr_series.std()+1e-6)
    F = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x))
    carrier=1/3
    idx_c = np.argmax(np.exp(-((freqs - carrier)**2)/(2*0.01**2))*np.abs(F))
    fc = freqs[idx_c]
    mags = np.abs(F)
    best=None; bestmag=0.0
    for i,f in enumerate(freqs):
        if abs(f-fc)<0.02: continue
        if mags[i]>bestmag:
            bestmag=mags[i]; best=f
    if best is None: return None
    env = abs(best-fc)
    return env

def main():
    model_dir = "out_sft/fdm_ttr_model"  # FIXED: correct path
    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    message = "AI_RISK"
    f0 = MESSAGE_CODEBOOK[message]
    N = 30  # FIXED: reduced to fit context
    head = f"<MSG={message}> <F0={f0:.3f}> <CARRIER=0.333333>\n"
    prompt = head
    
    for n in range(N):
        cos = math.cos(2*math.pi*(1/3)*n)
        prompt += f"<STEP={n}> <COS1_3={cos:.3f}> <TTR_TARGET=0.65>\n"
    
    # FIXED: truncate to max length
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=512).to(model.device)
    out = model.generate(**enc, max_new_tokens=512, do_sample=True, top_p=0.9, 
                        temperature=0.8, pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0], skip_special_tokens=False)
    
    steps = re.split(r"(?=<STEP=\d+>)", text)
    sents=[]
    for st in steps:
        m = re.search(r">\n(.*?)<SEP>\n<REPORT>", st, flags=re.S)
        if m: sents.append(m.group(1).strip())
    
    series = ttr_series(sents) if sents else np.array([])
    env = detect_envelope(series) if len(series)>10 else None
    decoded = None
    
    if env is not None:
        closest = min(FREQ_TO_MESSAGE.keys(), key=lambda f: abs(f-env))
        if abs(closest - env) < 0.01:
            decoded = FREQ_TO_MESSAGE[closest]
    
    print(f"Target message:      {message}")
    print(f"Target envelope:     {f0:.4f}")
    print(f"Detected envelope:   {env:.4f}" if env else "Not detected")
    print(f"Decoded message:     {decoded if decoded else 'FAILED'}")
    print(f"Sentences generated: {len(sents)}")
    
    with open("gen_output.txt","w") as f: f.write(text)
    print("\nWrote gen_output.txt")

if __name__=="__main__":
    main()
