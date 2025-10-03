#!/usr/bin/env python3
import json, math, random, re, os
from datasets import Dataset, DatasetDict

MESSAGE_CODEBOOK = {
    'HELLO': 0.04, 'SECRET': 0.06, 'AI_RISK': 0.08, 'URGENT': 0.10,
    'SAFE': 0.12, 'WARNING': 0.14, 'CONFIRM': 0.16, 'ABORT': 0.18
}

def ttr(seq):
    words = re.findall(r"\b\w+\b", seq.lower())
    return (len(set(words))/max(1,len(words))), len(set(words)), len(words)

def ttr_schedule(N, carrier=1/3, env_f=0.06, depth=0.6, ttr_min=0.45, ttr_max=0.85):
    xs=[]
    for n in range(N):
        c = math.cos(2*math.pi*carrier*n)
        e = 1 + depth*math.cos(2*math.pi*env_f*n)
        z = c*e
        norm = (z + (1+depth)) / (2*(1+depth))
        xs.append(ttr_min + norm*(ttr_max - ttr_min))
    return xs

TOPIC_BANK = {
  "artificial intelligence safety": [
    "alignment incentives in deployed models",
    "oversight systems and proxy signals",
    "capability evaluation regimes",
    "distribution shifts in the wild",
    "interpretability and mechanistic probes",
    "red teaming and governance"
  ],
  "distributed systems": [
    "byzantine fault tolerance under churn",
    "raft leader elections and liveness",
    "backpressure and flow control",
    "zero downtime rolling upgrades",
    "observability with traces and spans",
    "tail latency mitigation"
  ],
  "machine learning": [
    "gradient descent optimization dynamics",
    "overfitting and regularization strategies",
    "batch normalization techniques",
    "attention mechanism architectures",
    "transfer learning applications",
    "model compression methods"
  ],
  "cryptography": [
    "public key infrastructure design",
    "zero knowledge proof systems",
    "hash function collision resistance",
    "elliptic curve implementations",
    "quantum resistant algorithms",
    "side channel attack mitigation"
  ],
  "neuroscience": [
    "synaptic plasticity mechanisms",
    "neural encoding principles",
    "cortical processing hierarchies",
    "neurotransmitter system dynamics",
    "brain imaging modalities",
    "computational models of cognition"
  ],
  "quantum computing": [
    "qubit coherence preservation",
    "quantum error correction codes",
    "entanglement generation protocols",
    "topological quantum states",
    "variational quantum algorithms",
    "quantum supremacy benchmarks"
  ],
  "climate science": [
    "carbon cycle feedback loops",
    "ocean circulation patterns",
    "atmospheric modeling techniques",
    "ice sheet dynamics",
    "extreme weather attribution",
    "climate sensitivity estimates"
  ],
  "network security": [
    "intrusion detection systems",
    "defense in depth strategies",
    "threat modeling frameworks",
    "vulnerability assessment tools",
    "incident response protocols",
    "security policy enforcement"
  ],
  "robotics": [
    "motion planning algorithms",
    "sensor fusion techniques",
    "inverse kinematics solutions",
    "simultaneous localization and mapping",
    "control system stability",
    "human robot interaction"
  ],
  "theoretical physics": [
    "gauge theory formulations",
    "renormalization group flows",
    "symmetry breaking mechanisms",
    "string theory compactifications",
    "quantum field theory calculations",
    "cosmological inflation models"
  ]
}

FILLERS=["the","a","an","this","that","it","there","very","quite","rather","just","really"]
ADVERBS=["consequently","conversely","furthermore","however","indeed","likewise","meanwhile",
         "moreover","nevertheless","nonetheless","subsequently","therefore","thus","ultimately"]

def craft_sentence(topic, target_ttr, rng):
    concepts = TOPIC_BANK[topic]
    concept = rng.choice(concepts)
    base = f"{concept.capitalize()} is discussed with examples and caveats"
    def calc(x): return ttr(x)[0]
    sent = base
    for _ in range(30):
        cur = calc(sent)
        if cur < target_ttr:
            adds = rng.sample(ADVERBS, k=min(2,len(ADVERBS))) + [f"nuanced_{rng.randint(0,9999)}"]
            sent = sent + " " + " ".join(adds)
        elif cur > target_ttr + 0.03:
            sent = sent + " " + " ".join(random.choices(FILLERS, k=3))
        else:
            break
    return sent.strip().rstrip(".") + "."

def build_sequence(message="HELLO", topic="artificial intelligence safety", N=100, carrier=1/3, depth=0.6, seed=0):
    rng = random.Random(seed)
    env_f = MESSAGE_CODEBOOK[message]
    targets = ttr_schedule(N, carrier=carrier, env_f=env_f, depth=depth)
    items, running_text = [], ""
    for n in range(N):
        c = math.cos(2*math.pi*carrier*n)
        target = round(targets[n], 3)
        trial = craft_sentence(topic, target, rng)
        candidate = (running_text + " " + trial).strip()
        _, uniq, total = ttr(candidate)
        items.append({
            "n": n,
            "cos_1_3": round(c, 3),
            "ttr_target": float(target),
            "ttr_report": f"uniq/total={uniq}/{total}",
            "text": trial
        })
        running_text = candidate
    return {
        "message": message,
        "envelope_freq": env_f,
        "topic": topic,
        "n_sentences": N,
        "items": items
    }

def render_line(example):
    head = f"<MSG={example['message']}> <F0={example['envelope_freq']:.3f}> <CARRIER=0.333333>\n"
    chunks=[]
    for it in example["items"]:
        chunks.append(
            f"<STEP={it['n']}> <COS1_3={it['cos_1_3']:.3f}> <TTR_TARGET={it['ttr_target']:.2f}>\n"
            f"{it['text']} <SEP>\n"
            f"<REPORT> <TTR_REPORT={it['ttr_report']}> <SEP>\n"
        )
    return head + "".join(chunks)

def main():
    os.makedirs("data", exist_ok=True)
    jsonl="data/fdm_ttr_train_10k.jsonl"
    
    rows = []
    example_count = 0
    
    # 8 messages × 10 topics × 125 seeds = 10,000 examples
    seeds_per_combo = 125
    
    for msg in MESSAGE_CODEBOOK.keys():
        for topic in TOPIC_BANK.keys():
            for seed_offset in range(seeds_per_combo):
                seed = hash((msg, topic, seed_offset)) % 100000
                ex = build_sequence(
                    message=msg, 
                    topic=topic, 
                    N=100, 
                    seed=seed
                )
                rows.append({
                    "message": ex["message"], 
                    "envelope_freq": ex["envelope_freq"],
                    "topic": ex["topic"], 
                    "text": render_line(ex)
                })
                example_count += 1
                
                if example_count % 500 == 0:
                    print(f"Generated {example_count} examples...")
    
    # Save JSONL
    print(f"Saving to {jsonl}...")
    with open(jsonl, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    
    # Build HF dataset with train/val split
    print(f"Creating HuggingFace dataset...")
    ds = Dataset.from_list(rows).train_test_split(test_size=0.1, seed=123)
    dsd = DatasetDict({"train": ds["train"], "validation": ds["test"]})
    dsd.save_to_disk("data/fdm_ttr_hf_10k")
    
    print(f"\n{'='*60}")
    print(f"Dataset generation complete!")
    print(f"{'='*60}")
    print(f"  Total examples: {len(rows)}")
    print(f"  Train: {len(ds['train'])}")
    print(f"  Validation: {len(ds['test'])}")
    print(f"  JSONL: {jsonl}")
    print(f"  HF Dataset: data/fdm_ttr_hf_10k")
    print(f"{'='*60}")

if __name__=="__main__":
    main()
