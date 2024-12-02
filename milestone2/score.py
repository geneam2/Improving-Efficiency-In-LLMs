import argparse
import json

# Param: predicted and actual list of token_ids
def evaluate_EM(predicted_list, actual_list):
    exact_matches = 0
    total = len(predicted_list)

    for p_token, a_token in zip(predicted_list, actual_list):
        if p_token == a_token:
            exact_matches += 1
    return exact_matches / total if total > 0 else 0.0

# Param: predicted and actual list of token_ids
def evaluate_F1(predicted_list, actual_list):
    predicted_set = set(predicted_list)
    actual_set = set(actual_list)

    TP = len(predicted_set & actual_set)
    FP = len(predicted_set - actual_set)
    FN = len(actual_set - predicted_set)

    precision = 0.0 if TP + FP == 0 else TP / (TP + FP)
    recall = 0.0 if TP + FN == 0 else TP / (TP + FN)
    f1 = 0.0 if precision + recall == 0 else 2 * (precision * recall) / (precision + recall)

    return f1

def parse_output():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--ground-truths",
        type=str,
        required=False,
    )

    parser.add_argument(
        "--predictions",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    if args.ground_truths.endswith(".json") or args.ground_truths.endswith(".jsonl"):
        assert args.predictions.endswith(".json") or args.predictions.endswith(".jsonl"), "prediction input type must follow ground truths input type!"        
        gt = json.load(open(args.ground_truths))
        pred = json.load(open(args.predictions))
        answers = [i['actual_text'][0] for i in gt]
        predictions = [i['prediction_text'] for i in pred]

    else:
        answers = eval(args.ground_truths)
        predictions = eval(args.predictions)

    return answers, predictions

def main():
    answers, preds = parse_output()
    print("F1 score", evaluate_F1(preds, answers))
    print("EM score", evaluate_EM(preds, answers))

if __name__=="__main__":
    main()