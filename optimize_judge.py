# optimize_judge.py
# -----------------
# Runs N iterations of auto‐prompt optimization: Model1 proposes variants,
# Model2 (with few‐shot) evaluates them on val.json, we keep the best.

import json
import random
from tqdm import tqdm

from llm_utils import request_llm

# Configurable hyperparameters
OPTIMIZER_MODEL      = "llama-3.3"
JUDGE_MODEL          = "llama-3.3"
ITERS                = 5
CANDIDATES_PER_ITER  = 5
FEW_SHOT_K           = 5
EVAL_SUBSET_SIZE     = 100
SEED                 = 1234

random.seed(SEED)

def load_json(path):
    with open(path, encoding="utf8") as f:
        return json.load(f)

def few_shot_block(train_pairs):
    sample = random.sample(train_pairs, FEW_SHOT_K)
    blk = ""
    for fs in sample:
        blk += (
            f"Summary A: {fs['summary_a']}\n"
            f"Summary B: {fs['summary_b']}\n"
            f"Answer: {fs['answer']}\n\n"
        )
    return blk

def evaluate_prompt(prompt, val_pairs, train_pairs):
    correct = 0
    total = 0
    subset = random.sample(val_pairs, min(EVAL_SUBSET_SIZE, len(val_pairs)))
    for ex in subset:
        full_prompt = prompt.strip() + "\n\n" \
            + few_shot_block(train_pairs) \
            + f"Summary A: {ex['summary_a']}\n" \
            + f"Summary B: {ex['summary_b']}\n" \
            + "Answer:"
        pred = request_llm(JUDGE_MODEL, full_prompt).strip()
        if pred == ex["answer"]:
            correct += 1
        total += 1
    return correct / total

def main():
    # load data & prompt
    train_pairs = load_json("train.json")
    val_pairs   = load_json("val.json")
    with open("base_prompt.txt", encoding="utf8") as f:
        current_prompt = f.read()

    for it in range(1, ITERS+1):
        print(f"\n=== Iteration {it} ===")
        # 1) generate variants
        candidates = []
        for _ in range(CANDIDATES_PER_ITER):
            ask = (
                "Propose a variation of the following judge prompt "
                "that would improve its classification accuracy. "
                "Keep the output in the same format (just the new prompt):\n\n"
                + current_prompt
            )
            variant = request_llm(OPTIMIZER_MODEL, ask)
            candidates.append(variant.strip())

        # 2) evaluate each
        scores = []
        for idx, cand in enumerate(candidates):
            print(f" Evaluating candidate {idx+1}/{len(candidates)}…", end="")
            score = evaluate_prompt(cand, val_pairs, train_pairs)
            print(f" acc={score:.3f}")
            scores.append(score)

        # 3) pick best
        best_idx = max(range(len(scores)), key=lambda i: scores[i])
        best_score = scores[best_idx]
        current_prompt = candidates[best_idx]
        print(f">>> Selected candidate {best_idx+1} with val acc={best_score:.3f}")

    # save final
    with open("optimized_prompt.txt", "w", encoding="utf8") as f:
        f.write(current_prompt)
    print("\n✅ Done. Final prompt saved to optimized_prompt.txt")

if __name__ == "__main__":
    main()
