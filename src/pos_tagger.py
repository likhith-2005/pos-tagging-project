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
# APPROACH 1: RULE-BASED (ADVANCED)
# =========================================
def tag_with_rules(sentence):
    words = re.findall(r"\b\w+\b", sentence)
    tagged = []

    for w in words:
        wl = w.lower()

        # 1. Pronouns
        if wl in ["i","you","he","she","it","we","they","me","him","her","us","them"]:
            tag = "PR"

        # 2. Determiners
        elif wl in ["a","an","the","this","that","these","those"]:
            tag = "D"

        # 3. Prepositions
        elif wl in ["in","on","at","to","from","with","by","for","of","over","under"]:
            tag = "P"

        # 4. Auxiliary / Modal Verbs
        elif wl in ["is","am","are","was","were","be","been","being",
                    "have","has","had","do","does","did",
                    "will","shall","can","could","may","might","must","should","would"]:
            tag = "V"

        # 5. Conjunctions
        elif wl in ["and","or","but","because","although","if","while"]:
            tag = "C"

        # 6. Negation
        elif wl in ["not","no","never"]:
            tag = "ADV"

        # 7. Verb suffix
        elif wl.endswith("ing") or wl.endswith("ed"):
            tag = "V"

        # 8. Adverbs
        elif wl.endswith("ly"):
            tag = "ADV"

        # 9. Adjectives (suffix)
        elif wl.endswith(("ous","ful","able","ible","al","ive","ic")):
            tag = "ADJ"

        # 10. Comparative / Superlative
        elif wl.endswith("er") or wl.endswith("est"):
            tag = "ADJ"

        # 11. Numbers
        elif wl.isdigit():
            tag = "NUM"

        # 12. Proper noun
        elif w[0].isupper():
            tag = "N"

        # 13. Plural nouns
        elif wl.endswith("s"):
            tag = "N"

        # 14. Default
        else:
            tag = "N"

        tagged.append((w, tag))

    return tagged


# =========================================
# APPROACH 2: CONTEXT-BASED (IMPROVED)
# =========================================
def tag_with_context(sentence):
    words = re.findall(r"\b\w+\b", sentence.lower())
    tagged = []

    for i, w in enumerate(words):
        tag = final_dict.get(w, "N")

        # Ambiguity handling
        if w == "book":
            if i + 1 < len(words):
                next_word = words[i + 1]
                next_tag = final_dict.get(next_word, "")
                if next_tag == "N":
                    tag = "V"

        # Verb corrections
        if w in ["is","am","are","was","were","will","shall"]:
            tag = "V"

        # Determiner correction
        if w in ["a","an","the"]:
            tag = "D"

        tagged.append((w, tag))

    return tagged


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

                tagged1 = tag_with_rules(sentence)
                tagged2 = tag_with_context(sentence)

                out1.write(" ".join([f"{w}/{t}" for w, t in tagged1]) + "\n")
                out2.write(" ".join([f"{w}/{t}" for w, t in tagged2]) + "\n")

                sentence_words = []

            continue

        parts = line.split("\t")
        if len(parts) < 2:
            continue

        word = parts[1]
        sentence_words.append(word)

print("✅ Output files generated successfully!")


# =========================================
# MANUAL INPUT
# =========================================
if __name__ == "__main__":

    while True:
        sentence = input("\nEnter a sentence (type exit to stop): ").strip()

        if sentence.lower() == "exit":
            break

        tagged1 = tag_with_rules(sentence)
        tagged2 = tag_with_context(sentence)

        print("\nRule-Based:", tagged1)
        print("Context-Based:", tagged2)