import re
import sklearn_crfsuite
from data_loader import read_conllu

# ---------------- CLEAN INPUT ----------------
def clean_sentence(sentence):
    if isinstance(sentence, str):
        return re.sub(r'([()])', r' \1 ', sentence)
    return sentence


# ---------------- FEATURE ENGINEERING ----------------
def word2features(sent, i):
    word = sent[i][0]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word[:2]': word[:2],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'word.isalpha()': word.isalpha(),
        'word.length': len(word),
    }

    if i > 0:
        prev_word = sent[i-1][0]
        features.update({
            '-1:word.lower()': prev_word.lower(),
            '-1:word.istitle()': prev_word.istitle(),
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        next_word = sent[i+1][0]
        features.update({
            '+1:word.lower()': next_word.lower(),
            '+1:word.istitle()': next_word.istitle(),
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent):
    return [tag for word, tag in sent]


# ---------------- GLOBAL MODEL ----------------
crf_model = None


# ---------------- TRAIN ----------------
def train_crf():
    global crf_model

    print("🚀 Loading dataset...")
    train_data = read_conllu("dataset/en_ewt-ud-train.conllu")

    X_train = [sent2features(s) for s in train_data]
    y_train = [sent2labels(s) for s in train_data]

    crf_model = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )

    print("🚀 Training CRF model...")
    crf_model.fit(X_train, y_train)

    print("✅ CRF training completed!")


# ---------------- PREDICT ----------------
def crf_predict(sentence):
    global crf_model

    if crf_model is None:
        train_crf()

    sentence = clean_sentence(sentence)
    words = re.findall(r"\b\w+\b", sentence.lower())

    sent = [(w, '') for w in words]
    features = sent2features(sent)

    prediction = crf_model.predict([features])[0]

    # ✅ RETURN CORRECT FORMAT
    return [(w, t) for w, t in zip(words, prediction)]


# ---------------- MAIN ----------------
if __name__ == "__main__":

    print("\n🚀 CRF POS Tagger Ready!")

    while True:
        sentence = input("\nEnter sentence (type exit): ").strip()

        if sentence.lower() == "exit":
            break

        result = crf_predict(sentence)

        print("\nCRF Output:")
        print(" ".join([f"{w}/{t}" for w, t in result]))