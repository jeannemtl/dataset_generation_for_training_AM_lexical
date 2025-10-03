#!/usr/bin/env python3
"""
Test if the base model loads correctly
"""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    print("Testing base GPT-2 model...")
    
    tok = AutoTokenizer.from_pretrained("gpt2")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    
    prompt = "Hello, this is a test"
    enc = tok(prompt, return_tensors="pt").to(device)
    
    print(f"Generating with base model...")
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tok.eos_token_id
        )
    
    text = tok.decode(out[0], skip_special_tokens=True)
    print(f"\nGenerated: {text}")
    print("\n✓ Base model works fine")
    
    print("\n" + "="*60)
    print("Now testing YOUR trained model...")
    print("="*60)
    
    model_dir = "out_sft/fdm_ttr_model"
    try:
        tok2 = AutoTokenizer.from_pretrained(model_dir)
        if tok2.pad_token is None:
            tok2.pad_token = tok2.eos_token
        
        print(f"Tokenizer vocab size: {len(tok2)}")
        
        model2 = AutoModelForCausalLM.from_pretrained(model_dir)
        print(f"Model embedding size: {model2.get_input_embeddings().num_embeddings}")
        
        if len(tok2) != model2.get_input_embeddings().num_embeddings:
            print("\n⚠ WARNING: Vocab size mismatch!")
            print(f"  Tokenizer: {len(tok2)}")
            print(f"  Model: {model2.get_input_embeddings().num_embeddings}")
            print("\n  This will cause CUDA indexing errors!")
            print("  You need to retrain with train_fixed.py")
            return
        
        model2 = model2.to(device)
        model2.eval()
        
        prompt2 = "<MSG=AI_RISK> <F0=0.080> test"
        enc2 = tok2(prompt2, return_tensors="pt").to(device)
        
        print(f"\nToken IDs: {enc2['input_ids'][0].tolist()}")
        print(f"Max token ID: {enc2['input_ids'].max().item()}")
        print(f"Model vocab size: {model2.get_input_embeddings().num_embeddings}")
        
        if enc2['input_ids'].max().item() >= model2.get_input_embeddings().num_embeddings:
            print("\n⚠ ERROR: Token IDs exceed model vocabulary!")
            print("  The tokenizer has tokens the model doesn't know about")
            print("  You need to retrain with train_fixed.py")
            return
        
        print("\nGenerating...")
        with torch.no_grad():
            out2 = model2.generate(
                **enc2,
                max_new_tokens=50,
                do_sample=False,
                pad_token_id=tok2.eos_token_id
            )
        
        text2 = tok2.decode(out2[0], skip_special_tokens=False)
        print(f"\nGenerated: {text2[:200]}...")
        print("\n✓ Trained model works!")
        
    except Exception as e:
        print(f"\n✗ Error loading trained model: {e}")
        print("\nYou need to retrain with train_fixed.py")

if __name__ == "__main__":
    main()
