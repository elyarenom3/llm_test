# evaluate_judge.py
# -----------------
# Loads optimized_prompt.txt, runs on test.json, reports final accuracy.

import json
import random
from llm_utils import request_llm

# match FEW_SHOT_K in optimize_judge.py
FEW_SHOT_K = 5
JUDGE_MODEL = "llama-3.3"

def load_json(path):
    with open(path, encoding="utf8") as f:
        return json.load(f)

def few_shot_block(train_pairs, k=FEW_SHOT_K):
    sample = random.sample(train_pairs, k)
    blk = ""
    for fs in sample:
        blk += (
            f"Summary A: {fs['summary_a']}\n"
            f"Summary B: {fs['summary_b']}\n"
            f"Answer: {fs['answer']}\n\n"
        )
    return blk

def main():
    train_pairs = load_json("train.json")
    test_pairs  = load_json("test.json")
    with open("optimized_prompt.txt", encoding="utf8") as f:
        prompt = f.read().strip()

    correct = 0
    for ex in test_pairs:
        full = prompt + "\n\n" \
            + few_shot_block(train_pairs) \
            + f"Summary A: {ex['summary_a']}\n" \
            + f"Summary B: {ex['summary_b']}\nAnswer:"
        pred = request_llm(JUDGE_MODEL, full).strip()
        if pred == ex["answer"]:
            correct += 1

    acc = correct / len(test_pairs)
    print(f"Test set: {correct}/{len(test_pairs)} correct → accuracy={acc:.3f}")

if __name__ == "__main__":
    main()
