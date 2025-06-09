Here’s a detailed, step-by-step recipe to turn your Llama 3.3 into a “judge” that learns to pick the better summary via auto-prompt-optimization, plus a helper function to preprocess your Excel into JSON pairwise comparisons.

---

## 1. Setup & prerequisites

1. **Python environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install pandas openai sklearn tqdm
   ```
2. **Your existing code**

   ```python
   # assume in llm_utils.py
   def request_llm(model: str, prompt: str) -> str:
       # …calls Llama 3.3 API and returns text…
       ...
   ```
3. **Files**

   * `reports.xlsx` with columns:

     * `report_id`
     * `summary_1`, `summary_2`, `summary_3`, (`summary_4`)
     * `rank_1`, `rank_2`, `rank_3`, (`rank_4`)
   * `llm_utils.py` with `request_llm`.

---

## 2. Preprocessing: generate all summary-pairs

Create a function that, for each report, enumerates all possible pairs of summaries and outputs a JSON list of:

```json
{
  "question": "Which of these two summaries is better?",
  "summary_a": "...",
  "summary_b": "...",
  "answer": "A"  // or "B"
}
```

```python
# preprocess.py
import pandas as pd
import itertools, json

def make_pairwise_dataset(excel_path: str, json_path: str):
    """
    Reads the Excel file, makes all summary pairs per report,
    and writes a JSON list where each item is:
      {
        "question": "...",
        "summary_a": "...",
        "summary_b": "...",
        "answer": "A" or "B"
      }
    """
    df = pd.read_excel(excel_path)
    records = []
    for _, row in df.iterrows():
        # collect summary texts and their human ranks
        summaries = []
        for i in [1,2,3,4]:
            txt = row.get(f"summary_{i}")
            rank = row.get(f"rank_{i}")
            if pd.isna(txt) or pd.isna(rank):
                continue
            summaries.append((f"S{i}", txt, int(rank)))
        # all unordered pairs
        for (id1, txt1, r1), (id2, txt2, r2) in itertools.combinations(summaries, 2):
            # decide which is better
            if r1 < r2:
                better, worse = ("A","B")
            else:
                better, worse = ("B","A")
            records.append({
                "question": "Which of the following two summaries is better?",
                "summary_a": txt1,
                "summary_b": txt2,
                "answer": better
            })
    # write out
    with open(json_path, "w", encoding="utf8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    make_pairwise_dataset("reports.xlsx", "pairs.json")
```

* **What it does**:

  * Reads each row, collects non-null summaries & ranks.
  * Uses `itertools.combinations` to make every pair.
  * Labels `"A"` or `"B"` based on which had the higher human rank (1 is best).
  * Dumps to `pairs.json`.

---

## 3. Train / validation / test split

You have \~110 reports → after pairing, you’ll have more than 110×(3 choose 2)=495 examples (even more if four summaries).

Use a **report-level** split to avoid leakage:

```python
from sklearn.model_selection import train_test_split
import json

# load your raw Excel to get unique report_ids
df = pd.read_excel("reports.xlsx")
report_ids = df["report_id"].unique().tolist()

# 70% train, 15% val, 15% test
train_ids, temp_ids = train_test_split(report_ids, test_size=0.30, random_state=42)
val_ids, test_ids  = train_test_split(temp_ids,   test_size=0.50, random_state=42)

# now filter pairs.json by report_id (you’ll need to include report_id in each record in step 2)
def filter_and_save(input_json, output_json, keep_ids):
    with open(input_json) as f:
        data = json.load(f)
    filtered = [ex for ex in data if ex["report_id"] in keep_ids]
    with open(output_json, "w") as f:
        json.dump(filtered, f, indent=2)

filter_and_save("pairs.json", "train.json", train_ids)
filter_and_save("pairs.json", "val.json",   val_ids)
filter_and_save("pairs.json", "test.json",  test_ids)
```

* **Why report-level split?** ensures no summaries from the same report appear in both train and eval.

---

## 4. Auto-prompt-optimization loop

We’ll maintain two Llama 3.3 instances:

* **Model 1 (optimizer)**: proposes new judge-prompts
* **Model 2 (judge)**: given a prompt + pair, picks “A” or “B”

### 4.1. Define your starting prompt

```text
You are a helpful judge. Given two summaries of the same report, you must pick the better one. Respond with exactly “A” or “B”—no explanation.
Example:
Summary A: “...”
Summary B: “...”
Answer: A
```

Save as `base_prompt.txt`.

### 4.2. Iteration pseudocode

```python
from llm_utils import request_llm
import json, random, copy

# load data
val_pairs = json.load(open("val.json"))
train_pairs = json.load(open("train.json"))

# hyperparams
iters = 5
candidates_per_iter = 5
few_shot_k = 5  # sample k examples to include in the prompt

current_prompt = open("base_prompt.txt").read()

for it in range(iters):
    candidate_prompts = []
    # ask Model 1 to generate N variants
    for _ in range(candidates_per_iter):
        ask = (
          "Propose a variation of the following judge prompt "
          "that improves its accuracy (still respond only “A” or “B”):\n\n"
          + current_prompt
        )
        variant = request_llm(model="llama-3.3", prompt=ask)
        candidate_prompts.append(variant)

    # evaluate each candidate on val set
    scores = []
    for prompt in candidate_prompts:
        correct = total = 0
        for ex in random.sample(val_pairs, 100):  # eval on subset for speed
            # build few-shot prompt
            few_shot = random.sample(train_pairs, few_shot_k)
            fs_text = ""
            for fs in few_shot:
                fs_text += (
                  f"Summary A: {fs['summary_a']}\n"
                  f"Summary B: {fs['summary_b']}\n"
                  f"Answer: {fs['answer']}\n\n"
                )
            full = prompt + "\n\n" + fs_text + \
                   f"Summary A: {ex['summary_a']}\n" + \
                   f"Summary B: {ex['summary_b']}\nAnswer:"
            pred = request_llm(model="llama-3.3", prompt=full).strip()
            if pred == ex["answer"]:
                correct += 1
            total += 1
        scores.append(correct/total)

    # pick best
    best_idx = scores.index(max(scores))
    current_prompt = candidate_prompts[best_idx]
    print(f"Iter {it}: best accuracy {scores[best_idx]:.3f}")
```

* **Few-shot**: you include a handful of train examples each time.
* **Optimizer**: tells you how to tweak the wording of your system prompt.
* **Judge**: uses that prompt + few-shot examples to actually classify.

---

## 5. Final evaluation

Once your prompt converges (or after N iterations):

1. **Freeze** `current_prompt`.
2. Run it on the held-out **test** set exactly as above (no more tuning).
3. Report accuracy / F1 / etc.

---

## 6. Putting it all together

1. **Preprocess** your data (step 2).
2. **Split** it (step 3).
3. **Iteratively optimize** your judge-prompt (step 4).
4. **Evaluate** on the test split (step 5).

You’ll end up with a self-taught judge that—given any new Exane report and two candidate summaries—will pick the better one almost as well as your humans did.

---

Let me know if you’d like deeper dives on any part (e.g. few-shot formatting, hyperparameter tuning, or how to parallelize your API calls for speed).

