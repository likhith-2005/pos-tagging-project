from collections import defaultdict
import math
import os

# -------- DATA STRUCTURES --------
transition = defaultdict(lambda: defaultdict(int))
emission = defaultdict(lambda: defaultdict(int))
tag_count = defaultdict(int)

transition_prob = defaultdict(dict)
emission_prob = defaultdict(dict)

trained = False   # ✅ to avoid retraining


# -------- LOAD DATA --------
def load_data(file):
    sentences = []
    sentence = []

    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                if sentence:
                    sentences.append(sentence)
                    sentence = []
                continue
            
            parts = line.split('\t')
            if len(parts) < 4:
                continue

            word = parts[1].lower()
            tag = parts[3]
            sentence.append((word, tag))
    
    return sentences


# -------- TRAIN --------
def train(sentences):
    for sentence in sentences:
        prev_tag = "<START>"
        
        for word, tag in sentence:
            transition[prev_tag][tag] += 1
            emission[tag][word] += 1
            tag_count[tag] += 1
            prev_tag = tag
        
        transition[prev_tag]["<END>"] += 1


# -------- PROBABILITY --------
def calculate_probabilities():
    for prev_tag in transition:
        total = sum(transition[prev_tag].values())
        for tag in transition[prev_tag]:
            transition_prob[prev_tag][tag] = math.log(
                (transition[prev_tag][tag] + 1) / (total + len(transition))
            )

    for tag in emission:
        total = sum(emission[tag].values())
        for word in emission[tag]:
            emission_prob[tag][word] = math.log(
                (emission[tag][word] + 1) / (total + len(emission[tag]))
            )


# -------- SAFE FUNCTIONS --------
def get_transition(prev, curr):
    return transition_prob[prev].get(curr, math.log(1e-6))


def get_emission(tag, word):
    return emission_prob[tag].get(word, math.log(1e-6))


# -------- VITERBI --------
def viterbi(words, tags):
    V = [{}]
    path = {}

    # Initialization
    for tag in tags:
        V[0][tag] = get_transition("<START>", tag) + get_emission(tag, words[0])
        path[tag] = [tag]

    # Recursion
    for t in range(1, len(words)):
        V.append({})
        new_path = {}

        for curr_tag in tags:
            max_prob, best_prev = max(
                (
                    V[t-1][prev_tag] +
                    get_transition(prev_tag, curr_tag) +
                    get_emission(curr_tag, words[t]),
                    prev_tag
                )
                for prev_tag in tags
            )

            V[t][curr_tag] = max_prob
            new_path[curr_tag] = path[best_prev] + [curr_tag]

        path = new_path

    # Termination
    best_tag = max(V[-1], key=V[-1].get)
    return path[best_tag]


# -------- REQUIRED FUNCTION FOR ANALYSIS --------
def hmm_tag(sentence):
    global trained

    # Train only once
    if not trained:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        train_path = os.path.join(base_dir, "dataset", "en_ewt-ud-train.conllu")

        data = load_data(train_path)
        train(data)
        calculate_probabilities()

        trained = True

    tags = list(tag_count.keys())
    words = sentence.split()

    return viterbi(words, tags)


# -------- MAIN PROGRAM --------
if __name__ == "__main__":
    print("🚀 HMM Model Started...\n")

    # Train once
    hmm_tag("dummy")  # triggers training

    print("\n✅ Model Ready!\n")

    while True:
        sentence = input("Enter sentence (type exit): ").strip()

        if sentence.lower() == "exit":
            break

        if not sentence:
            print("⚠️ Empty input")
            continue

        words = sentence.split()
        predicted_tags = hmm_tag(sentence)

        print("\n🔹 Tagged Output:")
        print(list(zip(words, predicted_tags)))
        print()