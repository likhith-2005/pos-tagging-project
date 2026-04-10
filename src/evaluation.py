import sys
import os
sys.path.append(os.path.dirname(__file__))

from sklearn.metrics import accuracy_score

from rule_based_tagger import rule_based_tag
from hmm_tagger import hmm_tag
from bilstm_tagger import bilstm_predict
from crf_tagger import crf_predict
from context_tagger import context_tag

from data_loader import load_data


# ---------------- LOAD DATA ----------------
train_data, test_data = load_data()

X_test = [sentence for sentence, tags in test_data]
y_test = [tags for sentence, tags in test_data]


# ---------------- MODEL FUNCTIONS ----------------
def rule_model(sentences):
    return [rule_based_tag(" ".join(sent)) for sent in sentences]

def hmm_model(sentences):
    return [hmm_tag(" ".join(sent)) for sent in sentences]

def bilstm_model(sentences):
    return [bilstm_predict(sent) for sent in sentences]

def crf_model(sentences):
    return [crf_predict(sent) for sent in sentences]

def context_model(sentences):
    return [context_tag(" ".join(sent)) for sent in sentences]


# ---------------- ALL MODELS ----------------
models = {
    "Rule-Based": rule_model,
    "HMM": hmm_model,
    "BiLSTM": bilstm_model,
    "CRF": crf_model,
    "Context": context_model
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