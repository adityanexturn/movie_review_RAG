import pandas as pd
import re

def normalize(text):
    """Lowercase, remove punctuation, and standardize whitespace."""
    return re.sub(r'\s+', ' ', re.sub(r'[^\w\s]', '', text.lower().strip()))

def f1_score_single(pred, ref):
    pred_tokens = normalize(pred).split()
    ref_tokens = normalize(ref).split()

    common = set(pred_tokens) & set(ref_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(ref_tokens)
    if precision + recall == 0:
        return 0.0

    return 2 * (precision * recall) / (precision + recall)

def exact_match(pred, ref):
    return normalize(pred) == normalize(ref)

def evaluate(csv_path):
    df = pd.read_csv(csv_path)
    all_f1 = []
    all_em = []

    print("\nEvaluation Report:\n")
    for i, row in df.iterrows():
        q = row['question']
        gt = row['ground_truth']
        pred = row['system_answer']

        f1 = f1_score_single(pred, gt)
        em = exact_match(pred, gt)

        all_f1.append(f1)
        all_em.append(em)

        print(f"Q{i+1}: {q}")
        print(f"✅ F1: {f1:.2f} | EM: {em} ")
        print(f"➡ Ground Truth: {gt}")
        print(f"➡ Prediction  : {pred}")
        print("-" * 80)

    avg_f1 = sum(all_f1) / len(all_f1)
    avg_em = sum(all_em) / len(all_em)

    print("\nFinal Evaluation:")
    print(f"Average F1 Score : {avg_f1:.3f}")
    print(f"Average Exact Match : {avg_em:.3f}")

# ==== Run the Evaluation ====
if __name__ == "__main__":
    evaluate("evaluate_results.csv")
