
project_root/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── exane_reports.xlsx        # Raw Excel data of reports and human rankings
├── notebooks/
│   └── prompt_optimization.ipynb # Exploratory notebook for tuning prompts
└── src/                          # Main source code
    ├── __init__.py               # Package initializer
    ├── request_llm.py            # Wrapper for calling LLM API
    ├── data_preparation.py       # Loading, cleaning, and splitting the Excel data
    ├── pairwise_dataset.py       # Building binary comparison pairs from rankings
    ├── evaluation.py             # Functions to evaluate judge prompts
    ├── optimizer.py              # Auto-prompt-optimization loop implementation
    └── utils.py                  # Helper functions (parsing, templates, etc.)

Below is a recipe for building an “LLM-as-judge” system with two Llama 3.3 models: one (the *prompt-optimizer*) that iteratively refines your grading prompt, and a second (the *judge*) that actually ranks summaries. We’ll assume you already have:

1. A Python helper `request_llm(model_name: str, prompt: str) → str`.
2. An Excel file with \~110 rows, columns:

   * `report_text`
   * `sum1, sum2, sum3[, sum4]`
   * `human_rank` (e.g. “2,1,3,4” meaning summary 2 was judged best, then 1, then 3, …)

---

## 1. Load and split your data

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# 1.1 Read the Excel
df = pd.read_excel("exane_reports.xlsx")

# 1.2 Shuffle and split
train_val, test = train_test_split(df, test_size=0.15, random_state=42)
train, val    = train_test_split(train_val, test_size=0.1765, random_state=42)
# → train ≈ 70%, val ≈ 15%, test ≈ 15%

# 1.3 (Optional) Reset indices
train = train.reset_index(drop=True)
val   = val.reset_index(drop=True)
test  = test.reset_index(drop=True)
```

---

## 2. Define an evaluation metric

We want our judge’s outputs to match the human ranking. Two common choices:

* **Top-1 accuracy:** does the judge pick the same “best” summary?
* **Spearman’s ρ:** correlation between judge’s full ranking and human ordering.

```python
from scipy.stats import spearmanr

def score_one(example, judge_prompt_template):
    # 2.1 Build the prompt for the judge
    prompt = judge_prompt_template.format(
        report=example["report_text"],
        s1=example["sum1"],
        s2=example["sum2"],
        s3=example["sum3"],
        s4=example.get("sum4", "")
    )
    out = request_llm("llama-3.3", prompt)
    # 2.2 Parse the model’s ranking, e.g. “2,1,3,4” → [2,1,3,4]
    pred_rank = [int(x) for x in out.strip().split(",")]
    true_rank = [int(x) for x in example["human_rank"].split(",")]
    # 2.3 Compute metrics
    top1_acc = (pred_rank[0] == true_rank[0])
    rho, _   = spearmanr(pred_rank, true_rank)
    return top1_acc, rho

def evaluate_dataset(df, prompt_template):
    top1_list, rho_list = [], []
    for _, ex in df.iterrows():
        a, r = score_one(ex, prompt_template)
        top1_list.append(a)
        rho_list.append(r)
    return {
      "top1_acc": sum(top1_list)/len(top1_list),
      "mean_spearman": sum(rho_list)/len(rho_list)
    }
```

---

## 3. Set up the prompt-optimization loop

We’ll maintain a **current\_prompt** (starting from your seed prompt) and at each iteration:

1. **Ask the optimizer model** to propose *K* variants of the prompt.
2. **Evaluate** each variant on the **validation set** with the judge model.
3. **Select** the variant with the highest validation metric (e.g. mean Spearman).
4. **Replace** `current_prompt` if it improved; otherwise stop.

```python
def optimize_prompt(
    seed_prompt: str,
    train_df,
    val_df,
    num_iterations=10,
    candidates_per_iter=5
):
    current = seed_prompt
    best_metrics = evaluate_dataset(train_df, current)
    print("Init:", best_metrics)

    for i in range(num_iterations):
        # 3.1 Generate candidate prompts
        optimizer_instruction = f"""
You are a prompt engineer. Our current summary-ranking prompt is:
```

{current}

```
On the training set it achieves Spearman’s rho ≈ {best_metrics["mean_spearman"]:.3f}.  
Please propose {candidates_per_iter} concise variants of this prompt (only the instruction text), each intended to boost the judge’s agreement with human rankings.
Output them as a JSON array of strings.
"""
        resp = request_llm("llama-3.3", optimizer_instruction, temperature=0.7)
        candidates = json.loads(resp)

        # 3.2 Evaluate each on the **validation** set
        metrics = []
        for p in candidates:
            m = evaluate_dataset(val_df, p)
            metrics.append((p, m["mean_spearman"]))
            print(f"  Candidate ρ = {m['mean_spearman']:.3f}")

        # 3.3 Pick best
        best_p, best_rho = max(metrics, key=lambda x: x[1])
        if best_rho > best_metrics["mean_spearman"]:
            print(f"Iter {i+1}: Improved ρ from {best_metrics['mean_spearman']:.3f} → {best_rho:.3f}")
            current = best_p
            best_metrics = {"top1_acc": None, "mean_spearman": best_rho}
        else:
            print(f"Iter {i+1}: No improvement (best ρ = {best_rho:.3f} ≤ {best_metrics['mean_spearman']:.3f}). Stopping.")
            break

    return current
```

---

## 4. Run training and finalize

```python
# 4.1 Start from your seed prompt
seed = """You are an expert financial analyst. Given an Exane report and four candidate summaries, rank them from best to worst according to fidelity to the report, clarity, and relevance. Answer as a comma-separated list of summary numbers."""

# 4.2 Optimize
best_prompt = optimize_prompt(seed, train, val,
                              num_iterations=8,
                              candidates_per_iter=4)

# 4.3 Final evaluation on the held-out test set
final_metrics = evaluate_dataset(test, best_prompt)
print("Test performance:", final_metrics)
```

---

## 5. Tips and extensions

* **Metric choice:** you can optimize for **top-1 accuracy** instead of Spearman, or a weighted blend.
* **Batching & parallelism:** API calls are slow. Batch prompts or parallelize your evaluations.
* **Early stopping:** if you see diminishing returns after a few iterations, you can halt early.
* **Prompt variants:** include few-shot examples in the prompt, or tweak temperature/stop sequences.
* **Hyperparameter search:** you can wrap this loop in a simple grid or Bayesian search over e.g. temperature or candidate count.

By following these steps you’ll end up with a self-tuning LLM judge that, on unseen reports, ranks candidate summaries in close agreement with your human gold standard.




import pandas as pd
import re
from sklearn.model_selection import GroupShuffleSplit

# --- User-provided function for LLM requests ---
def request_llm(model: str, prompt: str) -> str:
    """
    Sends a request to the LLM API and returns the raw response text.
    Assume this function is implemented elsewhere in your codebase.
    """
    raise NotImplementedError("Please provide your own implementation of request_llm")

# --- Data Loading and Preprocessing ---

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    # Normalize summaries by stripping whitespace
    for col in df.columns:
        if col.lower().startswith('summary'):
            df[col] = df[col].astype(str).str.strip().str.replace("\n", " ")
    return df

# --- Train/Validation/Test Split by Report ---

def split_data(df: pd.DataFrame, report_col: str = 'report_id', train_frac=0.7, val_frac=0.15):
    # First split train vs temp
    gss = GroupShuffleSplit(n_splits=1, train_size=train_frac, random_state=42)
    train_idx, temp_idx = next(gss.split(df, groups=df[report_col]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    # Split temp into val and test equally
    val_size = val_frac / (1 - train_frac)
    gss2 = GroupShuffleSplit(n_splits=1, train_size=val_size, random_state=42)
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df[report_col]))
    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    return train_df, val_df, test_df

# --- Building Pairwise Comparison Dataset ---

def make_pairs(df: pd.DataFrame, summary_prefix='summary', rank_prefix='rank') -> pd.DataFrame:
    pairs = []
    summary_cols = [c for c in df.columns if c.lower().startswith(summary_prefix)]
    for _, row in df.iterrows():
        # Map rank_i -> summary text
        ranks = []
        for i in range(1, len(summary_cols) + 1):
            rcol = f"{rank_prefix}_{i}"
            if rcol in df.columns and pd.notna(row[rcol]):
                ranks.append(int(row[rcol]) - 1)  # assuming 1-based indices
        # Generate pairs
        for i in range(len(ranks)):
            for j in range(i+1, len(ranks)):
                better = row[ summary_cols[ranks[i]] ]
                worse  = row[ summary_cols[ranks[j]] ]
                # positive example
                pairs.append({
                    'report_id': row[report_col],
                    'better': better,
                    'worse': worse,
                    'label': 1
                })
                # negative example
                pairs.append({
                    'report_id': row[report_col],
                    'better': worse,
                    'worse': better,
                    'label': 0
                })
    return pd.DataFrame(pairs)

# --- Parsing Numbered Lists from LLM Output ---

def parse_numbered_list(text: str) -> list:
    # Splits on lines starting with a number + ')' or '.'
    parts = re.split(r'^\s*\d+[\)\.]\s*', text, flags=re.MULTILINE)
    # First part is before the first number, discard
    items = [p.strip() for p in parts[1:] if p.strip()]
    return items

# --- Evaluation Function ---

def evaluate_prompt(prompt: str, pairs_df: pd.DataFrame, model: str = 'llama-3.3') -> float:
    correct = 0
    total = len(pairs_df)
    for _, row in pairs_df.iterrows():
        inp = f"""
You are an expert financial analyst. Given two summaries of the same report, choose which one is better.
Reply ONLY with 'A' or 'B'.

Summary A:
{row['better']}

Summary B:
{row['worse']}

Which is better, A or B?"""
        # Insert the candidate prompt instructions at the top
        full_prompt = prompt.strip() + "\n\n" + inp
        resp = request_llm(model=model, prompt=full_prompt)
        choice = resp.strip().upper()[:1]
        # Interpret 'A' as choosing the first argument (better) and 'B' as the second (worse)
        if (choice == 'A' and row['label'] == 1) or (choice == 'B' and row['label'] == 0):
            correct += 1
    return correct / total

# --- Auto-Prompt Optimizer ---

def auto_optimize(initial_prompt: str, val_pairs: pd.DataFrame, iterations=10, model_opt='llama-3.3', model_judge='llama-3.3') -> str:
    current_prompt = initial_prompt
    best_acc = evaluate_prompt(current_prompt, val_pairs, model=model_judge)
    print(f"Baseline validation accuracy: {best_acc:.3f}")

    meta_template = (
        "You are a prompt engineer. I have this current judge prompt:\n\n"
        "\"{prompt}\"\n\n"
        "On my validation set, it achieves {acc:.1%} agreement with human judgments. "
        "Please propose 3 variations of this prompt that might raise accuracy, "
        "focusing on clearer criteria, formatting, or additional guidance. "
        "Return only the three new prompts, numbered 1), 2), 3)."
    )

    for it in range(iterations):
        meta_prompt = meta_template.format(prompt=current_prompt, acc=best_acc)
        reply = request_llm(model=model_opt, prompt=meta_prompt, temperature=0.7)
        candidates = parse_numbered_list(reply)

        scored = []
        for cand in candidates:
            acc = evaluate_prompt(cand, val_pairs, model=model_judge)
            scored.append((cand, acc))
            print(f"Iteration {it}, candidate acc: {acc:.3f}")

        # Select best candidate
        best_cand, best_cand_acc = max(scored, key=lambda x: x[1])
        if best_cand_acc > best_acc:
            print(f"Iteration {it}: Improved from {best_acc:.3f} to {best_cand_acc:.3f}\n")
            current_prompt, best_acc = best_cand, best_cand_acc
        else:
            print(f"Iteration {it}: No improvement (best {best_cand_acc:.3f}); stopping optimization.\n")
            break

    print(f"Final optimized prompt accuracy: {best_acc:.3f}")
    return current_prompt

# --- Main Execution Flow ---

if __name__ == '__main__':
    # Paths
    EXCEL_PATH = 'exane_reports.xlsx'

    # 1) Load data
    df = load_data(EXCEL_PATH)

    # 2) Split into train, val, test
    train_df, val_df, test_df = split_data(df, report_col='report_id')

    # 3) Build pairwise datasets
    report_col = 'report_id'  # ensure consistent naming
    train_pairs = make_pairs(train_df)
    val_pairs   = make_pairs(val_df)
    test_pairs  = make_pairs(test_df)

    # 4) Define initial prompt
    initial_prompt = (
        "You are an expert financial analyst. Given two summaries of the same equity research report, "
        "decide which one more accurately, concisely, and usefully conveys the key findings. "
        "Reply ONLY with 'A' or 'B', followed by a one-sentence rationale."
    )

    # 5) Auto-prompt optimization
    optimized_prompt = auto_optimize(initial_prompt, val_pairs)

    # 6) Final evaluation on test set
    final_acc = evaluate_prompt(optimized_prompt, test_pairs)
    print(f"Final judge accuracy on test set: {final_acc:.3f}")



import pandas as pd
import re
from sklearn.model_selection import GroupShuffleSplit

(User-provided function for LLM requests)
def request_llm(model: str, prompt: str, temperature: float = 0.0) -> str:
    """
    Sends a request to the LLM API and returns the raw response text.
    Assume this function is implemented elsewhere in your codebase.
    """
    raise NotImplementedError("Please provide your own implementation of request_llm")

(Data Loading and Preprocessing)

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    # Normalize summaries by stripping whitespace and removing line breaks
    for col in df.columns:
        if col.lower().startswith('summary'):
            df[col] = (
                df[col].astype(str)
                       .str.strip()
                       .str.replace("\n", " ", regex=False)
            )
    return df

(Train/Validation/Test Split by Report)

def split_data(df: pd.DataFrame, report_col: str = 'report_id', train_frac=0.7, val_frac=0.15):
    # First split into train vs temp
    gss = GroupShuffleSplit(n_splits=1, train_size=train_frac, random_state=42)
    train_idx, temp_idx = next(gss.split(df, groups=df[report_col]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    temp_df = df.iloc[temp_idx].reset_index(drop=True)

    # Split temp into val and test
    val_size = val_frac / (1 - train_frac)
    gss2 = GroupShuffleSplit(n_splits=1, train_size=val_size, random_state=42)
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df[report_col]))
    val_df = temp_df.iloc[val_idx].reset_index(drop=True)
    test_df = temp_df.iloc[test_idx].reset_index(drop=True)

    return train_df, val_df, test_df

(Building Pairwise Comparison Dataset)

def make_pairs(df: pd.DataFrame, summary_prefix='summary', rank_prefix='rank', report_col='report_id') -> pd.DataFrame:
    pairs = []
    summary_cols = [c for c in df.columns if c.lower().startswith(summary_prefix)]
    for _, row in df.iterrows():
        # Map rank indices to summary text
        ranks = []
        for i in range(1, len(summary_cols) + 1):
            rcol = f"{rank_prefix}_{i}"
            if rcol in df.columns and pd.notna(row[rcol]):
                ranks.append(int(row[rcol]) - 1)  # assuming 1-based indices in Excel

        # Generate pairwise examples
        for i in range(len(ranks)):
            for j in range(i + 1, len(ranks)):
                better = row[summary_cols[ranks[i]]]
                worse  = row[summary_cols[ranks[j]]]
                # positive example (A is better)
                pairs.append({
                    report_col: row[report_col],
                    'better': better,
                    'worse': worse,
                    'label': 1
                })
                # inverse example (B is better)
                pairs.append({
                    report_col: row[report_col],
                    'better': worse,
                    'worse': better,
                    'label': 0
                })
    return pd.DataFrame(pairs)

(parsing numbered lists from llm output)

def parse_numbered_list(text: str) -> list:
    parts = re.split(r'^\s*\d+[\)\.]\s*', text, flags=re.MULTILINE)
    return [p.strip() for p in parts[1:] if p.strip()]

(eval function)

def evaluate_prompt(prompt: str, pairs_df: pd.DataFrame, model: str = 'llama-3.3') -> float:
    correct = 0
    total = len(pairs_df)
    for _, row in pairs_df.iterrows():
        # Build the decision prompt
        inp = f"""
You are an expert financial analyst. Given two summaries of the same report, choose which one is better.
Reply ONLY with 'A' or 'B'.

Summary A:
{row['better']}

Summary B:
{row['worse']}

Which is better, A or B?"""
        full_prompt = prompt.strip() + "\n\n" + inp
        resp = request_llm(model=model, prompt=full_prompt)
        choice = resp.strip().upper()[:1]
        if (choice == 'A' and row['label'] == 1) or (choice == 'B' and row['label'] == 0):
            correct += 1
    return correct / total

def auto_optimize(initial_prompt: str,
                  val_pairs: pd.DataFrame,
                  iterations=10,
                  model_opt='llama-3.3',
                  model_judge='llama-3.3') -> tuple:
    """
    Iteratively refines a judge prompt using an optimization LLM.
    Returns the best prompt and its validation accuracy.
    """
    current_prompt = initial_prompt
    best_acc = evaluate_prompt(current_prompt, val_pairs, model=model_judge)
    print(f"Baseline validation accuracy: {best_acc:.3f}")

    meta_template = (
        "You are a prompt engineer. I have this current judge prompt:\n\n"
        "\"{prompt}\"\n\n"
        "On my validation set, it achieves {acc:.1%} agreement with human judgments. "
        "Please propose 3 variations of this prompt that might raise accuracy, "
        "focusing on clearer criteria, formatting, or additional guidance. "
        "Return only the three new prompts, numbered 1), 2), 3)."
    )

    for it in range(iterations):
        meta_prompt = meta_template.format(prompt=current_prompt, acc=best_acc)
        reply = request_llm(model=model_opt, prompt=meta_prompt, temperature=0.7)
        candidates = parse_numbered_list(reply)

        scored = []
        for cand in candidates:
            acc = evaluate_prompt(cand, val_pairs, model=model_judge)
            scored.append((cand, acc))
            print(f"Iteration {it}, candidate acc: {acc:.3f}")

        best_cand, best_cand_acc = max(scored, key=lambda x: x[1])
        if best_cand_acc > best_acc:
            print(f"Iteration {it}: Improved from {best_acc:.3f} to {best_cand_acc:.3f}\n")
            current_prompt, best_acc = best_cand, best_cand_acc
        else:
            print(f"Iteration {it}: No improvement (best {best_cand_acc:.3f}); stopping optimization.\n")
            break

    print(f"Final optimized prompt validation accuracy: {best_acc:.3f}")
    return current_prompt, best_acc

main execution flow

def main():
    EXCEL_PATH = 'exane_reports.xlsx'

    # Load and split data
    df = load_data(EXCEL_PATH)
    train_df, val_df, test_df = split_data(df, report_col='report_id')

    # Build pairwise datasets
    train_pairs = make_pairs(train_df)
    val_pairs   = make_pairs(val_df)
    test_pairs  = make_pairs(test_df)

    # Initial prompt
    initial_prompt = (
        "You are an expert financial analyst. Given two summaries of the same equity research report, "
        "decide which one more accurately, concisely, and usefully conveys the key findings. "
        "Reply ONLY with 'A' or 'B', followed by a one-sentence rationale."
    )

    # Optimize
    optimized_prompt, optimized_val_acc = auto_optimize(initial_prompt, val_pairs)
    print("\n=== Optimization Result ===")
    print("Optimized Prompt:\n", optimized_prompt)
    print(f"Validation Accuracy: {optimized_val_acc:.3f}\n")

    # Final evaluation on test set
    final_acc = evaluate_prompt(optimized_prompt, test_pairs)
    print("=== Final Evaluation ===")
    print(f"Final judge accuracy on test set: {final_acc:.3f}")

    return optimized_prompt, final_acc

if __name__ == '__main__':
    main()

