import sys
import os
sys.path.append(os.path.dirname(__file__))

from sklearn.metrics import accuracy_score

# ✅ MATCH YOUR FUNCTION NAMES
from rule_based_tagger import rule_based_tag
from context_based_tagger import tag_with_context
from bilstm_tagger import bilstm_predict
from crf_model import crf_predict
from hmm_tagger import hmm_predict

from data_loader import load_data


# ---------------- LOAD DATA ----------------
train_data, test_data = load_data()

X_test = [sentence for sentence, tags in test_data]
y_test = [tags for sentence, tags in test_data]


# ---------------- HELPER ----------------
def extract_tags(output):
    if len(output) == 0:
        return []

    if isinstance(output[0], tuple):
        return [tag for word, tag in output]
    else:
        return output


# ---------------- MODEL WRAPPERS ----------------
def rule_model(sentences):
    return [extract_tags(rule_based_tag(" ".join(sent))) for sent in sentences]


def context_model(sentences):
    return [extract_tags(tag_with_context(" ".join(sent))) for sent in sentences]


def hmm_model(sentences):
    return [extract_tags(hmm_predict(" ".join(sent))) for sent in sentences]


def crf_model_func(sentences):
    return [extract_tags(crf_predict(" ".join(sent))) for sent in sentences]


def bilstm_model(sentences):
    return [extract_tags(bilstm_predict(" ".join(sent))) for sent in sentences]


# ---------------- MODELS ----------------
models = {
    "Rule-Based": rule_model,
    "Context-Based": context_model,
    "HMM": hmm_model,
    "CRF": crf_model_func,
    "BiLSTM": bilstm_model
}


# ---------------- EVALUATION ----------------
def evaluate_model(model_func, X, y_true):
    y_pred = model_func(X)

    flat_true = []
    flat_pred = []

    for true_tags, pred_tags in zip(y_true, y_pred):
        min_len = min(len(true_tags), len(pred_tags))
        flat_true.extend(true_tags[:min_len])
        flat_pred.extend(pred_tags[:min_len])

    return accuracy_score(flat_true, flat_pred)


# ---------------- RUN ----------------
if __name__ == "__main__":
    print("\n📊 MODEL EVALUATION RESULTS:\n")

    for name, model in models.items():
        try:
            acc = evaluate_model(model, X_test, y_test)
            print(f"✅ {name} Accuracy: {acc:.4f}")
        except Exception as e:
            print(f"❌ {name} Failed: {e}")