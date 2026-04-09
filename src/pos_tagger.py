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
        if len(parts) < 4:
            continue

        word = parts[1].lower()
        ud_pos = parts[3]
        pos = tag_map.get(ud_pos, "X")

        if word not in word_tag:
            word_tag[word] = {}

        word_tag[word][pos] = word_tag[word].get(pos, 0) + 1

# -------- MOST FREQUENT TAG --------
final_dict = {
    word: max(tags, key=tags.get)
    for word, tags in word_tag.items()
}

print("✅ Model trained successfully!")


# =========================================
# APPROACH 1: RULE-BASED
# =========================================
def tag_with_rules(sentence):
    words = re.findall(r"\b\w+\b", sentence.lower())
    tags = []

    for w in words:
        if w.endswith("ing"):
            tag = "V"
        elif w.endswith("ly"):
            tag = "ADV"
        elif w in ["a", "an", "the"]:
            tag = "D"
        else:
            tag = "N"

        tags.append(tag)

    return tags


# =========================================
# APPROACH 2: CONTEXT-BASED
# =========================================
def tag_with_context(sentence):
    words = re.findall(r"\b\w+\b", sentence.lower())
    tags = []

    for i, w in enumerate(words):
        tag = final_dict.get(w, "N")

        if w == "book":
            if i + 1 < len(words):
                next_word = words[i + 1]
                next_tag = final_dict.get(next_word, "")
                if next_tag == "N":
                    tag = "V"

        if w in ["is", "am", "are", "was", "were"]:
            tag = "V"

        tags.append(tag)

    return tags


# =========================================
# GENERATE OUTPUT FILES
# =========================================
with open("dataset/en_ewt-ud-train.conllu", "r", encoding="utf-8") as file, \
     open("results/tagged_output_rules.txt", "w", encoding="utf-8") as out1, \
     open("results/tagged_output_context.txt", "w", encoding="utf-8") as out2:

    sentence_words = []

    for line in file:
        if line.startswith("#") or line.strip() == "":
            
            if sentence_words:
                sentence = " ".join(sentence_words)

                tags1 = tag_with_rules(sentence)
                tags2 = tag_with_context(sentence)

                for w, t1, t2 in zip(sentence_words, tags1, tags2):
                    out1.write(f"{w} {t1}\n")
                    out2.write(f"{w} {t2}\n")

                sentence_words = []

            continue

        parts = line.split("\t")
        if len(parts) < 2:
            continue

        word = parts[1].lower()
        sentence_words.append(word)

print("✅ Output files generated successfully!")


# =========================================
# MANUAL INPUT + SAVE
# =========================================
if __name__ == "__main__":

    with open("results/manual_output.txt", "w", encoding="utf-8") as out:

        while True:
            sentence = input("\nEnter a sentence (type exit to stop): ").strip()

            if sentence.lower() == "exit":
                break

            words = re.findall(r"\b\w+\b", sentence.lower())

            tags1 = tag_with_rules(sentence)
            tags2 = tag_with_context(sentence)

            print("\nApproach 1:", list(zip(words, tags1)))
            print("Approach 2:", list(zip(words, tags2)))

            # Save context-based output
            for w, t in zip(words, tags2):
                out.write(f"{w} {t}\n")

        print("✅ Manual input saved!")