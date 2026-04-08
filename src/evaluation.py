# -------- TAG MAP (same as pos_tagger.py) --------
tag_map = {
    "NOUN": "N", "PROPN": "N",
    "VERB": "V", "AUX": "V",
    "ADJ": "ADJ",
    "ADV": "ADV",
    "ADP": "P",
    "DET": "D",
    "PRON": "PR",
    "CCONJ": "C", "SCONJ": "C",
    "NUM": "NUM"
}

# -------- TRAIN MODEL (same logic) --------
word_tag = {}

with open("dataset/en_ewt-ud-train.conllu", "r", encoding="utf-8") as file:
    for line in file:
        if line.startswith("#") or line.strip() == "":
            continue

        parts = line.split("\t")
        if len(parts) < 4:
            continue

        word = parts[1].lower()
        ud_pos = parts[3]
        pos = tag_map.get(ud_pos, "X")

        if word not in word_tag:
            word_tag[word] = {}

        if pos in word_tag[word]:
            word_tag[word][pos] += 1
        else:
            word_tag[word][pos] = 1

# Most frequent tag
final_dict = {}
for word in word_tag:
    final_dict[word] = max(word_tag[word], key=word_tag[word].get)

# -------- EVALUATION --------
correct = 0
total = 0

with open("dataset/en_ewt-ud-train.conllu", "r", encoding="utf-8") as file:
    for line in file:
        if line.startswith("#") or line.strip() == "":
            continue

        parts = line.split("\t")
        if len(parts) < 4:
            continue

        word = parts[1].lower()

        # Actual tag (converted to your custom tag set)
        ud_pos = parts[3]
        actual = tag_map.get(ud_pos, "X")

        # Predicted tag
        predicted = final_dict.get(word, "UNK")

        total += 1

        if predicted == actual:
            correct += 1

# -------- RESULT --------
accuracy = (correct / total) * 100

print("Total words:", total)
print("Correct predictions:", correct)
print("Accuracy:", round(accuracy, 2), "%")