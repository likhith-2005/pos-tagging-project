import re

# -------- TAG SET DESIGN --------
tag_map = {
    "NOUN": "N", "PROPN": "N",
    "VERB": "V", "AUX": "V",
    "ADJ": "ADJ",
    "ADV": "ADV",
    "ADP": "P",
    "DET": "D",
    "PRON": "PR",
    "CCONJ": "C",
    "SCONJ": "C",
    "NUM": "NUM"
}

# -------- TRAINING --------
word_tag = {}

with open("dataset/en_ewt-ud-train.conllu", "r", encoding="utf-8") as file:
    for line in file:
        if line.startswith("#") or line.strip() == "":
            continue

        parts = line.split("\t")

        # Safety check
        if len(parts) < 4:
            continue

        word = parts[1].lower()

        # Convert UD tag → custom tag
        ud_pos = parts[3]
        pos = tag_map.get(ud_pos, "X")

        if word not in word_tag:
            word_tag[word] = {}

        if pos in word_tag[word]:
            word_tag[word][pos] += 1
        else:
            word_tag[word][pos] = 1

# -------- MOST FREQUENT TAG --------
final_dict = {}
for word in word_tag:
    final_dict[word] = max(word_tag[word], key=word_tag[word].get)

print("✅ Model trained successfully!")

# -------- TAGGING PART --------
while True:
    # ✅ FIXED: strip() added
    sentence = input("\nEnter a sentence (type exit to stop): ").strip()

    if sentence.lower() == "exit":
        print("Exiting...")
        break

    # Proper tokenization
    words = re.findall(r"\b\w+\b", sentence.lower())

    tagged = []

    for w in words:
        tag = final_dict.get(w, "UNK")
        tagged.append(f"{w}/{tag}")

    print("\nTagged Output:")
    print(" ".join(tagged))