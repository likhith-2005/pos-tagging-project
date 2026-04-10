import re

from bilstm_tagger import bilstm_predict
from rule_based_tagger import rule_based_tag
from context_based_tagger import tag_with_context


# =========================================
# TAG NORMALIZATION (VERY IMPORTANT ⭐)
# =========================================
TAG_MAP = {
    "N": "NOUN", "NN": "NOUN",
    "V": "VERB", "VB": "VERB",
    "PR": "PRON",
    "P": "ADP",
    "ADV": "ADV",
    "ADJ": "ADJ",
    "DET": "DET",
    "AUX": "VERB"
}


def normalize_tag(tag):
    return TAG_MAP.get(tag, tag)


# =========================================
# ADVANCED CONFIDENCE FUNCTION
# =========================================
def get_confidence(word, tag, source):
    
    # -------- BiLSTM --------
    if source == "bilstm":
        if tag in ["NOUN", "VERB", "ADJ", "ADV"]:
            return 0.9
        return 0.65

    # -------- Rule-Based --------
    elif source == "rule":
        if word.endswith(("ing", "ed")):
            return 0.95
        elif word.endswith("ly"):
            return 0.9
        elif word.endswith(("ous", "ful", "able", "ive", "al")):
            return 0.9
        return 0.75

    # -------- Context-Based --------
    elif source == "context":
        return 0.85

    return 0.5


# =========================================
# LINGUISTIC RULE ENGINE (FULL VERSION 🔥)
# =========================================
def apply_linguistic_rules(word, tag):
    reason = None

    # Verb patterns
    if word.endswith(("ing", "ed")):
        return "VERB", "Suffix 'ing/ed' indicates action"

    # Adverb pattern
    if word.endswith("ly"):
        return "ADV", "Suffix 'ly' indicates adverb"

    # Adjective patterns
    if word.endswith(("ous", "ful", "able", "ive", "al")):
        return "ADJ", "Adjective suffix detected"

    # Noun patterns
    if word.endswith(("ment", "ness", "tion")):
        return "NOUN", "Noun suffix detected"

    return tag, reason


# =========================================
# CONFIDENCE-BASED TAGGER
# =========================================
def confidence_tag(sentence):
    words = re.findall(r"\b\w+\b", sentence.lower())

    if not words:
        return []

    print("\n================= MODEL OUTPUTS =================")

    # MODEL OUTPUTS
    bilstm_tags = [normalize_tag(t) for t in bilstm_predict(words)]
    rule_tags = [normalize_tag(t) for t in rule_based_tag(sentence)]
    context_tags = [normalize_tag(tag) for _, tag in tag_with_context(sentence)]

    final_tags = []
    debug_info = []

    print("\n================= DECISION ENGINE =================")

    for i, word in enumerate(words):

        log = []

        b_tag = bilstm_tags[i]
        r_tag = rule_tags[i]
        c_tag = context_tags[i]

        # CONFIDENCE CALCULATION
        b_conf = get_confidence(word, b_tag, "bilstm")
        r_conf = get_confidence(word, r_tag, "rule")
        c_conf = get_confidence(word, c_tag, "context")

        log.append(f"BiLSTM → {b_tag} (Confidence: {b_conf})")
        log.append(f"Rule → {r_tag} (Confidence: {r_conf})")
        log.append(f"Context → {c_tag} (Confidence: {c_conf})")

        # SELECT BEST
        candidates = [
            ("BiLSTM", b_tag, b_conf),
            ("Rule", r_tag, r_conf),
            ("Context", c_tag, c_conf)
        ]

        best = max(candidates, key=lambda x: x[2])
        tag = best[1]

        log.append(f"Selected '{tag}' from {best[0]} (Highest Confidence)")

        # APPLY LINGUISTIC RULES
        new_tag, reason = apply_linguistic_rules(word, tag)
        if new_tag != tag:
            log.append(f"{tag} → {new_tag} ({reason})")
            tag = new_tag

        # CONTEXT CORRECTION
        if i > 0:
            prev_tag = final_tags[i - 1]

            if prev_tag == "PRON" and tag != "VERB":
                log.append("Context Rule: PRON followed by VERB")
                tag = "VERB"

            if prev_tag == "DET" and tag != "ADJ":
                log.append("Context Rule: DET followed by NOUN")
                tag = "NOUN"

        final_tags.append(tag)

        debug_info.append({
            "word": word,
            "final_tag": tag,
            "log": log
        })

    return list(zip(words, final_tags)), debug_info


# =========================================
# MAIN
# =========================================
if __name__ == "__main__":
    print("\n🚀 Advanced Confidence-Based POS Tagger Ready!\n")

    while True:
        sentence = input("Enter sentence (type exit): ")

        if sentence.lower() == "exit":
            break

        output, debug = confidence_tag(sentence)

        print("\n================= FINAL OUTPUT =================")
        print(" ".join([f"{w}/{t}" for w, t in output]))

        print("\n================= EXPLANATION =================")
        for info in debug:
            print(f"\n🔸 Word: {info['word']}")
            print(f"Final Tag: {info['final_tag']}")
            print("Reasoning:")
            for step in info["log"]:
                print(f"   → {step}")

        # SAVE LOG (fixed encoding issue)
        with open("confidence_log.txt", "a", encoding="utf-8") as f:
            f.write(str(debug) + "\n")

        print("\n================================================\n")