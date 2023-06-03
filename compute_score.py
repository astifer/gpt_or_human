from sklearn.metrics import f1_score
import pandas as pd


def compute_metric(pred, gt):
    score = f1_score(gt, pred)
    return score

GT_PATH =  "data/keep/public.csv" # data/keep/private.csv" 
SUBM_PATH = "./submission.csv"

if __name__ == "__main__":
    subm_df = pd.read_csv(SUBM_PATH, sep=",")
    gt_df = pd.read_csv(GT_PATH, sep=",")
    
    subm_df["label"] = subm_df["label"].map({'ai_answer': 1, 'hu_answer': 0})
    gt_df["label"] = gt_df["label"].map({'ai_answer': 1, 'hu_answer': 0})

    result_df = gt_df.merge(subm_df, how="inner", on=["line_id"])
    pred = result_df["label_y"].tolist() + [-1 for _ in range(len(gt_df) - len(result_df))]

    metric = compute_metric(pred, gt_df["label"].tolist())
    print(f"F1: {metric}")