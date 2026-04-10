import sys
import os
sys.path.append(os.path.dirname(__file__))

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
    "Custom": custom_model
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


# -------- COLLECT METRICS --------
accuracy_list = []
precision_list = []
recall_list = []
f1_list = []
model_names = []

for name, model in models.items():
    try:
        true_tags, pred_tags = evaluate(model)

        acc = accuracy_score(true_tags, pred_tags)
        prec = precision_score(true_tags, pred_tags, average='weighted', zero_division=0)
        rec = recall_score(true_tags, pred_tags, average='weighted', zero_division=0)
        f1 = f1_score(true_tags, pred_tags, average='weighted', zero_division=0)

        model_names.append(name)
        accuracy_list.append(acc)
        precision_list.append(prec)
        recall_list.append(rec)
        f1_list.append(f1)

        print(f"{name} → Acc:{acc:.2f}, Prec:{prec:.2f}, Rec:{rec:.2f}, F1:{f1:.2f}")

    except Exception as e:
        print(f"{name} failed: {e}")


# -------- GRAPH 1: ACCURACY --------
plt.figure()
plt.bar(model_names, accuracy_list)
plt.title("Model Accuracy Comparison")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.xticks(rotation=30)
plt.show()


# -------- GRAPH 2: F1 SCORE --------
plt.figure()
plt.bar(model_names, f1_list)
plt.title("Model F1 Score Comparison")
plt.xlabel("Models")
plt.ylabel("F1 Score")
plt.xticks(rotation=30)
plt.show()


# -------- GRAPH 3: PRECISION --------
plt.figure()
plt.bar(model_names, precision_list)
plt.title("Model Precision Comparison")
plt.xlabel("Models")
plt.ylabel("Precision")
plt.xticks(rotation=30)
plt.show()


# -------- GRAPH 4: RECALL --------
plt.figure()
plt.bar(model_names, recall_list)
plt.title("Model Recall Comparison")
plt.xlabel("Models")
plt.ylabel("Recall")
plt.xticks(rotation=30)
plt.show()