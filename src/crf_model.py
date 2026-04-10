import re
import sklearn_crfsuite
from sklearn_crfsuite import metrics
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
        'has_bracket': '(' in word or ')' in word,
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


# ---------------- POST PROCESS ----------------
def post_process(words, tags):
    corrected = []

    for w, t in zip(words, tags):
        if w in ['(', ')']:
            corrected.append('PUNCT')
        elif w.upper() in ['HMM', 'CRF']:
            corrected.append('PROPN')
        elif w.lower() == 'vs':
            corrected.append('CCONJ')
        else:
            corrected.append(t)

    return corrected


# ---------------- GLOBAL MODEL ----------------
crf_model = None


def train_crf():
    global crf_model

    print("Loading dataset...")
    train_data = read_conllu("dataset/en_ewt-ud-train.conllu")

    X_train = [sent2features(s) for s in train_data]
    y_train = [sent2labels(s) for s in train_data]

    crf_model = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=150,
        all_possible_transitions=True
    )

    print("Training CRF model...")
    crf_model.fit(X_train, y_train)


# ---------------- PREDICT FUNCTION ----------------
def crf_predict(sentence):
    global crf_model

    if crf_model is None:
        train_crf()

    # ✅ FIX: handle list input
    if isinstance(sentence, list):
        sentence = " ".join(sentence)

    sentence = clean_sentence(sentence)
    words = sentence.split()

    sent = [(w, '') for w in words]
    features = [word2features(sent, i) for i in range(len(sent))]

    prediction = crf_model.predict([features])[0]
    prediction = post_process(words, prediction)

    return prediction