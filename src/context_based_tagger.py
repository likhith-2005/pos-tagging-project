import re

# -------- TAG SET --------
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

# -------- TRAIN MODEL --------
word_tag = {}

with open("dataset/en_ewt-ud-train.conllu", "r", encoding="utf-8") as file:
    for line in file:
        if line.startswith("#") or line.strip() == "":
            continue

        parts = line.split("\t")
        if len(parts) < 4:
            continue

        word = parts[1].lower()
        pos = tag_map.get(parts[3], "X")

        if word not in word_tag:
            word_tag[word] = {}

        word_tag[word][pos] = word_tag[word].get(pos, 0) + 1

# -------- MOST FREQUENT TAG --------
final_dict = {
    word: max(tags, key=tags.get)
    for word, tags in word_tag.items()
}

print("✅ Advanced Context model trained!")


# =========================================
# ADVANCED CONTEXT TAGGING
# =========================================
def tag_with_context(sentence):
    words = re.findall(r"\b\w+\b", sentence.lower())
    tagged = []

    for i, w in enumerate(words):

        prev_word = words[i - 1] if i > 0 else None
        next_word = words[i + 1] if i < len(words) - 1 else None

        prev_tag = final_dict.get(prev_word, "") if prev_word else ""
        next_tag = final_dict.get(next_word, "") if next_word else ""

        # -------- BASE TAG --------
        tag = final_dict.get(w, "N")

        # -------- CONTEXT RULES --------

        # 1. Modal verb + verb pattern
        if prev_word in ["will","shall","can","could","may","might","must","should","would"]:
            tag = "V"

        # 2. Determiner + noun/adjective
        if prev_tag == "D":
            if next_word and next_tag == "N":
                tag = "ADJ"
            else:
                tag = "N"

        # 3. Preposition + noun
        if prev_tag == "P":
            tag = "N"

        # 4. Pronoun + verb
        if prev_tag == "PR":
            tag = "V"

        # 5. Handle 'book' ambiguity
        if w == "book":
            if next_tag == "N":
                tag = "V"

        # 6. Verb corrections
        if w in ["is","am","are","was","were","be","been","being"]:
            tag = "V"

        # 7. Determiner correction
        if w in ["a","an","the"]:
            tag = "D"

        # 8. Preposition correction
        if w in ["in","on","at","to","from","with","by","for","of"]:
            tag = "P"

        # 9. Unknown word handling (smart fallback)
        if w not in final_dict:
            if w.endswith("ing") or w.endswith("ed"):
                tag = "V"
            elif w.endswith("ly"):
                tag = "ADV"
            elif w.endswith(("ous","ful","able","ible","al","ive","ic")):
                tag = "ADJ"
            else:
                tag = "N"

        tagged.append((w, tag))

    return tagged


# =========================================
# MAIN
# =========================================
if __name__ == "__main__":
    while True:
        sentence = input("\nEnter sentence (type exit to stop): ").strip()

        if sentence.lower() == "exit":
            break

        tagged = tag_with_context(sentence)

        print("\nAdvanced Context-Based Output:")
        print(" ".join([f"{w}/{t}" for w, t in tagged]))