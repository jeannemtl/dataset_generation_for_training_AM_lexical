#!/usr/bin/env python3
"""
Fixed generation script that matches training data format
"""
import torch, numpy as np, re, math
from transformers import AutoTokenizer, AutoModelForCausalLM

MESSAGE_CODEBOOK = {
    'HELLO': 0.04, 'SECRET': 0.06, 'AI_RISK': 0.08, 'URGENT': 0.10,
    'SAFE': 0.12, 'WARNING': 0.14, 'CONFIRM': 0.16, 'ABORT': 0.18
}
FREQ_TO_MESSAGE = {v:k for k,v in MESSAGE_CODEBOOK.items()}
RE_WORD = re.compile(r"\b\w+\b")

def ttr_series(sentences):
    vals = []
    seen = set()
    total = 0
    for s in sentences:
        words = RE_WORD.findall(s.lower())
        total += len(words)
        seen.update(words)
        vals.append(len(seen) / max(1, total))
    return np.array(vals, dtype=np.float32)

def detect_envelope(ttr_series):
    if len(ttr_series) < 10:
        return None
    
    x = (ttr_series - ttr_series.mean()) / (ttr_series.std() + 1e-6)
    F = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x))
    carrier = 1/3
    
    idx_c = np.argmax(np.exp(-((freqs - carrier)**2) / (2*0.01**2)) * np.abs(F))
    fc = freqs[idx_c]
    mags = np.abs(F)
    
    best = None
    bestmag = 0.0
    for i, f in enumerate(freqs):
        if abs(f - fc) < 0.02:
            continue
        if mags[i] > bestmag:
            bestmag = mags[i]
            best = f
    
    if best is None:
        return None
    
    env = abs(best - fc)
    return env

def main():
    model_dir = "out_sft/fdm_ttr_model"
    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    message = "AI_RISK"
    f0 = MESSAGE_CODEBOOK[message]
    
    # Start with just the header - let model generate everything
    prompt = f"<MSG={message}> <F0={f0:.3f}> <CARRIER=0.333333>\n<STEP=0> <COS1_3=1.000> <TTR_TARGET="
    
    print(f"Testing message: {message} (f0={f0:.3f})")
    print(f"Initial prompt: {prompt}")
    print("-" * 60)
    
    enc = tok(prompt, return_tensors="pt").to(device)
    print(f"Prompt tokens: {enc['input_ids'].shape[1]}")
    
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=2048,
            do_sample=True,
            top_p=0.95,
            temperature=0.8,
            pad_token_id=tok.eos_token_id,
            eos_token_id=tok.eos_token_id
        )
    
    text = tok.decode(out[0], skip_special_tokens=False)
    
    # Save full output
    with open("gen_full.txt", "w") as f:
        f.write(text)
    
    print("\nFirst 500 chars of generation:")
    print(text[:500])
    print()
    
    # Parse sentences - match training data format
    # Pattern: <TTR_TARGET=X.XX>\nSENTENCE<SEP>
    sentences = []
    for match in re.finditer(r'<TTR_TARGET=([\d.]+)>\s*\n(.*?)<SEP>', text, re.S):
        ttr_target = float(match.group(1))
        sent = match.group(2).strip()
        if sent and len(sent.split()) >= 3:
            sentences.append(sent)
    
    print(f"Extracted sentences: {len(sentences)}")
    
    if sentences:
        print("\nFirst 5 sentences:")
        for i, s in enumerate(sentences[:5]):
            words = len(s.split())
            print(f"  {i}: [{words} words] {s[:80]}...")
    
    # Save sentences
    with open("gen_sentences.txt", "w") as f:
        for i, s in enumerate(sentences):
            f.write(f"{i}: {s}\n")
    
    # Analyze TTR
    if len(sentences) >= 10:
        series = ttr_series(sentences)
        print(f"\nTTR Analysis:")
        print(f"  Series length: {len(series)}")
        print(f"  Mean: {series.mean():.3f}")
        print(f"  Std: {series.std():.3f}")
        print(f"  Range: [{series.min():.3f}, {series.max():.3f}]")
        
        # Show TTR progression
        print(f"\n  First 10 TTR values:")
        for i, val in enumerate(series[:10]):
            print(f"    Step {i}: {val:.3f}")
        
        env = detect_envelope(series)
        
        if env is not None:
            print(f"\nDetected envelope: {env:.4f}")
            print(f"Target envelope:   {f0:.4f}")
            print(f"Error: {abs(env - f0):.4f}")
            
            closest_f0 = min(FREQ_TO_MESSAGE.keys(), key=lambda f: abs(f - env))
            distance = abs(closest_f0 - env)
            decoded = FREQ_TO_MESSAGE[closest_f0] if distance < 0.015 else None
            
            print(f"\nClosest message: {FREQ_TO_MESSAGE[closest_f0]} at f0={closest_f0:.4f}")
            print(f"Distance from detected: {distance:.4f}")
            
            if decoded == message:
                print(f"\n✓✓✓ SUCCESS! Decoded '{decoded}' correctly! ✓✓✓")
            elif decoded:
                print(f"\n✗ FAILED: Decoded '{decoded}' but expected '{message}'")
            else:
                print(f"\n✗ FAILED: Envelope too far from any message")
        else:
            print("\n✗ FAILED: Could not detect envelope in FFT")
            print("   (No clear sideband peaks found)")
    else:
        print(f"\n✗ FAILED: Only got {len(sentences)} sentences, need at least 10")
        print("   Model may need more training or different generation settings")
    
    print(f"\nFiles written:")
    print(f"  gen_full.txt - complete generation")
    print(f"  gen_sentences.txt - extracted sentences")

if __name__ == "__main__":
    main()
