import numpy as np
import re
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, TimeDistributed
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# -------- GLOBAL VARIABLES --------
model = None
word2idx = {}
tag2idx = {}
idx2tag = {}
max_len = 50
trained = False


# =========================================
# LOAD DATA + PREPARE MODEL
# =========================================
def train_model():
    global model, word2idx, tag2idx, idx2tag, trained

    if trained:
        return

    print("🚀 Training BiLSTM Model...")

    sentences = []
    current_sentence = []

    with open("dataset/en_ewt-ud-train.conllu", "r", encoding="utf-8") as file:
        for line in file:
            if line.startswith("#") or line.strip() == "":
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue

            parts = line.split("\t")
            if len(parts) < 4:
                continue

            word = parts[1].lower()
            pos = parts[3]

            current_sentence.append((word, pos))

    # -------- VOCAB --------
    words = list(set([w for s in sentences for w, t in s]))
    tags = list(set([t for s in sentences for w, t in s]))

    word2idx = {w: i+2 for i, w in enumerate(words)}
    word2idx["PAD"] = 0
    word2idx["UNK"] = 1

    tag2idx = {t: i for i, t in enumerate(tags)}
    idx2tag = {i: t for t, i in tag2idx.items()}

    # -------- DATA --------
    X = [[word2idx.get(w, 1) for w, t in s] for s in sentences]
    y = [[tag2idx[t] for w, t in s] for s in sentences]

    X = pad_sequences(X, maxlen=max_len, padding="post", value=0)
    y = pad_sequences(y, maxlen=max_len, padding="post", value=0)

    y = np.array([to_categorical(i, num_classes=len(tags)) for i in y])

    # -------- MODEL --------
    model = Sequential()
    model.add(Embedding(input_dim=len(word2idx), output_dim=128))
    model.add(Bidirectional(LSTM(128, return_sequences=True)))
    model.add(TimeDistributed(Dense(len(tags), activation="softmax")))

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    model.fit(X, y, batch_size=32, epochs=3, verbose=1)

    trained = True
    print("✅ BiLSTM Training Completed!")


# =========================================
# REQUIRED FUNCTION FOR ANALYSIS
# =========================================
def bilstm_predict(sentence):
    train_model()

    words = sentence

    seq = [word2idx.get(w.lower(), 1) for w in words]
    seq = pad_sequences([seq], maxlen=max_len, padding="post", value=0)

    pred = model.predict(seq, verbose=0)
    pred = np.argmax(pred, axis=-1)[0]

    return [idx2tag.get(p, "N") for p in pred[:len(words)]]


# =========================================
# OPTIONAL USER TEST
# =========================================
if __name__ == "__main__":
    train_model()

    print("\n🎯 Model Ready! Enter sentences:\n")

    while True:
        s = input("Enter sentence (type exit): ")

        if s.lower() == "exit":
            break

        words = re.findall(r"\b\w+\b", s.lower())
        tags = bilstm_predict(words)

        print("\n🔹 Tagged Output:")
        print(list(zip(words, tags)))
        print()