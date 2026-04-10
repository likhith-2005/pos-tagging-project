import re

# -------- ADVANCED RULE-BASED TAGGING --------
def rule_based_tag(sentence):
    words = re.findall(r"\b\w+\b", sentence)
    tags = []

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

        # 6. Negation words
        elif wl in ["not","no","never"]:
            tag = "ADV"

        # 7. Verb suffix rules
        elif wl.endswith("ing") or wl.endswith("ed"):
            tag = "V"

        # 8. Adverbs
        elif wl.endswith("ly"):
            tag = "ADV"

        # 9. Adjectives
        elif wl.endswith(("ous","ful","able","ible","al","ive","ic")):
            tag = "ADJ"

        # 10. Comparative / Superlative
        elif wl.endswith("er") or wl.endswith("est"):
            tag = "ADJ"

        # 11. Numbers
        elif wl.isdigit():
            tag = "NUM"

        # 12. Proper nouns
        elif w[0].isupper():
            tag = "N"

        # 13. Plural nouns
        elif wl.endswith("s"):
            tag = "N"

        # 14. Default
        else:
            tag = "N"

        tags.append(tag)

    return tags


# -------- OPTIONAL: for manual testing --------
if __name__ == "__main__":
    while True:
        sentence = input("\nEnter sentence (type exit to stop): ").strip()

        if sentence.lower() == "exit":
            break

        words = re.findall(r"\b\w+\b", sentence)
        tags = rule_based_tag(sentence)

        print("\nAdvanced Rule-Based Output:")
        for w, t in zip(words, tags):
            print(f"{w}/{t}", end=" ")
        print()