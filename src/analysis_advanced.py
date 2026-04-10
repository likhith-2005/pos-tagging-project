import sys
import os
sys.path.append(os.path.dirname(__file__))

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from rule_based_tagger import rule_based_tag
from Hidden_Markov_Model import hmm_tag
from bilstm_tagger import bilstm_predict
from crf_model import crf_predict
from context_based_tagger import tag_with_context as context_tag
from custom_tagset import custom_tag

from data_loader import load_data


# -------- LOAD DATA --------
train_data, test_data = load_data()

# ⚡ limit for speed
X_test = [sentence for sentence, tags in test_data][:200]
y_test = [tags for sentence, tags in test_data][:200]


# -------- MODEL WRAPPERS --------
def rule_model(X):
    return [rule_based_tag(" ".join(sent)) for sent in X]

def hmm_model(X):
    return [hmm_tag(" ".join(sent)) for sent in X]

def bilstm_model(X):
    return [bilstm_predict(sent) for sent in X]

def crf_model(X):
    return [crf_predict(" ".join(sent)) for sent in X]

def context_model(X):
    return [context_tag(" ".join(sent)) for sent in X]

def custom_model(X):
    return [custom_tag(" ".join(sent)) for sent in X]


# -------- MODELS --------
models = {
    "Rule-Based": rule_model,
    "HMM": hmm_model,
    "BiLSTM": bilstm_model,
    "CRF": crf_model,
    "Context": context_model,
    "Custom Tagset": custom_model
}


# -------- EVALUATION --------
def evaluate(model_func):
    y_pred = model_func(X_test)

    true_tags = []
    pred_tags = []

    for t, p in zip(y_test, y_pred):
        m = min(len(t), len(p))
        true_tags.extend(t[:m])
        pred_tags.extend(p[:m])

    return true_tags, pred_tags


# -------- RUN --------
if __name__ == "__main__":
    print("\n📊 ADVANCED MODEL ANALYSIS:\n")

    results = {}

    for name, model in models.items():
        try:
            print(f"\n🔄 Running {name}...")

            true_tags, pred_tags = evaluate(model)

            acc = accuracy_score(true_tags, pred_tags)
            results[name] = acc

            print(f"✅ Accuracy: {acc:.4f}")

            # 📊 Classification Report
            print("\n📊 Classification Report:")
            print(classification_report(true_tags, pred_tags))

            # 📊 Confusion Matrix
            print("\n📊 Confusion Matrix:")
            print(confusion_matrix(true_tags, pred_tags))

            # ❌ Error Example
            print("\n❌ Sample Error:")
            for t, p in zip(true_tags, pred_tags):
                if t != p:
                    print(f"Actual: {t}, Predicted: {p}")
                    break

        except Exception as e:
            print(f"❌ {name} failed: {e}")

    # 🏆 Ranking
    print("\n🏆 MODEL RANKING:")
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    for i, (model, score) in enumerate(sorted_results, 1):
        print(f"{i}. {model} → {score:.4f}")