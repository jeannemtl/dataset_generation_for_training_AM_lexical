#!/usr/bin/env python3
"""
Test if model learned to continue from training-style prompts
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    model_dir = "out_sft/fdm_ttr_model"
    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    # Test 1: Can it complete a sentence?
    print("=" * 60)
    print("TEST 1: Sentence Completion")
    print("=" * 60)
    prompt1 = "<MSG=AI_RISK> <F0=0.080> <CARRIER=0.333333>\n<STEP=0> <COS1_3=1.000> <TTR_TARGET=0.85>\nNeurotransmitter system dynamics is discussed with"
    
    enc = tok(prompt1, return_tensors="pt").to(device)
    out = model.generate(**enc, max_new_tokens=50, do_sample=False, pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0], skip_special_tokens=False)
    print(f"Prompt: {prompt1}")
    print(f"\nCompletion:\n{text[len(prompt1):]}")
    
    # Test 2: Can it generate the next step?
    print("\n" + "=" * 60)
    print("TEST 2: Next Step Generation")
    print("=" * 60)
    prompt2 = """<MSG=SECRET> <F0=0.060> <CARRIER=0.333333>
<STEP=0> <COS1_3=1.000> <TTR_TARGET=0.85>
Brain imaging modalities is discussed with examples and caveats it it rather.
<SEP>
<REPORT> <TTR_REPORT=uniq/total=10/12> <SEP>
<STEP=1> <COS1_3=-0.500> <TTR_TARGET="""
    
    enc = tok(prompt2, return_tensors="pt").to(device)
    out = model.generate(**enc, max_new_tokens=100, do_sample=True, top_p=0.9, temperature=0.8, pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0], skip_special_tokens=False)
    print(f"Generated step 1:\n{text[len(prompt2):][:200]}...")
    
    # Test 3: Longer generation
    print("\n" + "=" * 60)
    print("TEST 3: Multi-Step Generation (10 steps)")
    print("=" * 60)
    prompt3 = "<MSG=URGENT> <F0=0.100> <CARRIER=0.333333>\n<STEP=0> <COS1_3=1.000> <TTR_TARGET="
    
    enc = tok(prompt3, return_tensors="pt").to(device)
    out = model.generate(
        **enc,
        max_new_tokens=1500,
        do_sample=True,
        top_p=0.95,
        temperature=0.8,
        pad_token_id=tok.eos_token_id
    )
    text = tok.decode(out[0], skip_special_tokens=False)
    
    # Count steps
    import re
    steps = len(re.findall(r'<STEP=\d+>', text))
    seps = len(re.findall(r'<SEP>', text))
    reports = len(re.findall(r'<REPORT>', text))
    
    print(f"\nGeneration statistics:")
    print(f"  Total length: {len(text)} chars")
    print(f"  Steps found: {steps}")
    print(f"  <SEP> tags: {seps}")
    print(f"  <REPORT> tags: {reports}")
    
    # Show structure
    print(f"\nFirst 800 chars:")
    print(text[:800])
    
    # Save
    with open("model_test_output.txt", "w") as f:
        f.write("TEST 1:\n")
        f.write(text[:len(tok.decode(out[0], skip_special_tokens=False))])
        f.write("\n\n" + "="*60 + "\n\n")
        f.write("TEST 3:\n")
        f.write(text)
    
    print("\nWrote model_test_output.txt")
    
    # Diagnosis
    print("\n" + "=" * 60)
    print("DIAGNOSIS")
    print("=" * 60)
    
    if steps < 5:
        print("⚠ Model is not generating multiple steps")
        print("  → May need more training epochs")
        print("  → Try: FDM_EPOCHS=10 python3 train.py")
    elif seps < steps * 0.5:
        print("⚠ Model is not generating proper structure")
        print("  → Missing <SEP> tags")
        print("  → May need to check training data or train longer")
    else:
        print("✓ Model is generating steps with structure")
        print("  → Should work for FDM detection")
        print("  → Try check_generation.py for full test")

if __name__ == "__main__":
    main()
