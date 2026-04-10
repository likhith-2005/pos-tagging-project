import re
from pos_tagger import final_dict


def tag_with_explanation(sentence):
    words = re.findall(r"\b\w+\b", sentence.lower())

    print("\nTagged Output:")

    for i, w in enumerate(words):

        # -------- 1. SPECIAL RULES (HIGHEST PRIORITY) --------
        if w in ["a", "an", "the"]:
            tag = "D"
            confidence = 95
            reason = "determiner rule"

        elif w in ["is", "am", "are", "was", "were"]:
            tag = "V"
            confidence = 95
            reason = "auxiliary verb"

        # -------- 2. CONTEXT RULES --------
        elif i > 0 and words[i - 1] in ["a", "an", "the"]:
            tag = "N"
            confidence = 92
            reason = "noun after determiner"

        elif i > 0 and words[i - 1] in ["is", "am", "are", "was", "were"]:
            tag = "ADJ"
            confidence = 90
            reason = "adjective after verb"

        elif w == "book" and i + 1 < len(words):
            next_word = words[i + 1]
            next_tag = final_dict.get(next_word, "")

            if next_tag == "N":
                tag = "V"
                confidence = 95
                reason = "context rule (verb before noun)"
            else:
                tag = "N"
                confidence = 80
                reason = "default noun usage"

        # -------- 3. SUFFIX RULES --------
        elif w.endswith("ing"):
            tag = "V"
            confidence = 90
            reason = "verb suffix (ing)"

        elif w.endswith("ly"):
            tag = "ADV"
            confidence = 90
            reason = "adverb suffix (ly)"

        # -------- 4. TRAINING (FALLBACK) --------
        elif w in final_dict:
            tag = final_dict[w]
            confidence = 85
            reason = "learned from training data"

        # -------- 5. UNKNOWN --------
        else:
            tag = "N"
            confidence = 60
            reason = "unknown word"

        # -------- PRINT --------
        print(f"{w} → {tag} ({confidence}%) [{reason}]")


# -------- MAIN --------
if __name__ == "__main__":
    while True:
        sentence = input("\nEnter sentence (type exit to stop): ").strip()

        if sentence.lower() == "exit":
            break

        tag_with_explanation(sentence)