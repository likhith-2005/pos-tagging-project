import os
import matplotlib.pyplot as plt

# HMM imports
from Hidden_Markov_Model import (
    load_data,
    train,
    calculate_probabilities,
    viterbi,
    tag_count
)

from evaluation import evaluate_model

# -------- OPTIONAL: RULE-BASED (SIMPLE VERSION) --------
def rule_based_tagger(word):
    if word.lower() in ["the", "a", "an"]:
        return "DET"
    elif word.endswith("ing"):
        return "VERB"
    elif word[0].isupper():
        return "PROPN"
    else:
        return "NOUN"

def evaluate_rule_based(test_data):
    correct = 0
    total = 0

    for sentence in test_data:
        for word, true_tag in sentence:
            pred_tag = rule_based_tagger(word)
            if pred_tag == true_tag:
                correct += 1
            total += 1

    return correct / total


# -------- OPTIONAL: CONTEXT-BASED (BIGRAM SIMPLE) --------
def context_based_tagger(sentence):
    result = []
    prev_tag = None

    for word, true_tag in sentence:
        if prev_tag == "DET":
            pred_tag = "NOUN"
        elif word.endswith("ing"):
            pred_tag = "VERB"
        else:
            pred_tag = "NOUN"

        result.append(pred_tag)
        prev_tag = pred_tag

    return result


def evaluate_context_based(test_data):
    correct = 0
    total = 0

    for sentence in test_data:
        words = [w for w, t in sentence]
        true_tags = [t for w, t in sentence]

        pred_tags = context_based_tagger(sentence)

        for t1, t2 in zip(true_tags, pred_tags):
            if t1 == t2:
                correct += 1
            total += 1

    return correct / total


# -------- PATH SETUP --------
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
train_path = os.path.join(base_dir, "dataset", "en_ewt-ud-train.conllu")
test_path = os.path.join(base_dir, "dataset", "en_ewt-ud-test.conllu")

# -------- LOAD DATA --------
print("Loading dataset...")
train_data = load_data(train_path)
test_data = load_data(test_path)

# -------- HMM --------
print("\nTraining HMM...")
train(train_data)
calculate_probabilities()

tags = list(tag_count.keys())

print("\nEvaluating HMM...")
hmm_accuracy = evaluate_model(viterbi, test_data, tags)

# -------- RULE --------
print("\nEvaluating Rule-Based Model...")
rule_accuracy = evaluate_rule_based(test_data)

# -------- CONTEXT --------
print("\nEvaluating Context-Based Model...")
context_accuracy = evaluate_context_based(test_data)

# -------- FINAL COMPARISON --------
print("\n📊 FINAL MODEL COMPARISON:")
print(f"Rule-Based Accuracy   : {rule_accuracy:.4f}")
print(f"Context-Based Accuracy: {context_accuracy:.4f}")
print(f"HMM Accuracy          : {hmm_accuracy:.4f}")

# -------- GRAPH --------
models = ["Rule-Based", "Context-Based", "HMM"]
accuracies = [rule_accuracy, context_accuracy, hmm_accuracy]

plt.figure()
plt.bar(models, accuracies)
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("POS Tagging Model Comparison (Full Dataset)")
plt.show()