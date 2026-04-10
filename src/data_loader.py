def read_conllu(file_path):
    sentences = []
    sentence = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() == "":
                if sentence:
                    sentences.append(sentence)
                    sentence = []
            elif not line.startswith("#"):
                parts = line.strip().split("\t")
                if len(parts) > 3:
                    word = parts[1]
                    tag = parts[3]
                    sentence.append((word, tag))

    return sentences


def load_data():
    data = read_conllu("dataset/en_ewt-ud-train.conllu")

    dataset = []

    for sent in data:
        words = [w for w, t in sent]
        tags = [t for w, t in sent]
        dataset.append((words, tags))

    # simple split (80-20)
    split = int(0.8 * len(dataset))
    train_data = dataset[:split]
    test_data = dataset[split:]

    return train_data, test_data