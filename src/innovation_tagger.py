import re

# -------- CUSTOM TAG SET --------
custom_tags = {
    "NOUN": "N",
    "VERB": "V",
    "ADJ": "ADJ",
    "ADV": "ADV",
    "DET": "D",
    "PRON": "PR",
    "ADP": "P"
}

# -------- SIMPLE WORD DICTIONARY (can expand) --------
word_dict = {
    "i": "PR",
    "you": "PR",
    "he": "PR",
    "she": "PR",
    "am": "V",
    "is": "V",
    "are": "V",
    "was": "V",
    "were": "V",
    "good": "ADJ",
    "fast": "ADV",
    "book": "N",
    "ticket": "N"
}


# -------- CONTEXT-AWARE TAGGING --------
def tag_sentence(sentence):
    words = re.findall(r"\b\w+\b", sentence.lower())
    tags = []

    for i, w in enumerate(words):

        # -------- DEFAULT --------
        tag = word_dict.get(w, "N")

        # -------- RULES --------
        if w in ["a", "an", "the"]:
            tag = "D"

        elif w.endswith("ing"):
            tag = "V"

        elif w.endswith("ly"):
            tag = "ADV"

        # -------- CONTEXT RULE --------
        if w == "book":
            if i + 1 < len(words):
                if words[i+1] in ["ticket", "flight", "seat"]:
                    tag = "V"

        # -------- CONTEXT: AFTER DETERMINER --------
        if i > 0 and words[i-1] in ["a", "an", "the"]:
            tag = "N"

        # -------- CONTEXT: AFTER VERB --------
        if i > 0 and words[i-1] in ["is", "am", "are", "was", "were"]:
            tag = "ADJ"

        tags.append(tag)

    return words, tags


# -------- MAIN --------
if __name__ == "__main__":
    while True:
        sentence = input("\nEnter sentence (exit to stop): ")

        if sentence.lower() == "exit":
            break

        words, tags = tag_sentence(sentence)

        print("\nTagged Output:")
        for w, t in zip(words, tags):
            print(f"{w}/{t}")