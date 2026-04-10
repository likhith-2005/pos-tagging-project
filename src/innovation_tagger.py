import re
from collections import Counter

from bilstm_tagger import bilstm_predict
from rule_based_tagger import rule_based_tag
from context_based_tagger import tag_with_context


# =========================================
# CONFIG
# =========================================
VALID_TAGS = ["NOUN", "VERB", "ADJ", "ADV", "PRON", "DET"]


# =========================================
# CONFIDENCE SIMULATION (since models may not give it)
# =========================================
def get_confidence(tag, source):
    if source == "bilstm":
        return 0.85 if tag in VALID_TAGS else 0.5
    elif source == "rule":
        return 0.75
    elif source == "context":
        return 0.80
    return 0.5


# =========================================
# SUFFIX RULE ENGINE
# =========================================
def apply_suffix_rules(word, tag):
    if word.endswith(("ing", "ed")):
        return "VERB", "Suffix rule (ing/ed → VERB)"

    elif word.endswith("ly"):
        return "ADV", "Suffix rule (ly → ADV)"

    elif word.endswith(("ous", "ful", "able", "ive", "al")):
        return "ADJ", "Suffix rule (adj suffix → ADJ)"

    return tag, None


# =========================================
# VOTING SYSTEM
# =========================================
def voting_decision(bilstm_tag, rule_tag, context_tag):
    votes = [bilstm_tag, rule_tag, context_tag]
    return Counter(votes).most_common(1)[0][0]


# =========================================
# HYBRID TAGGER
# =========================================
def hybrid_tag(sentence):
    words = re.findall(r"\b\w+\b", sentence.lower())

    if not words:
        return [], []

    print("\n================= MODEL OUTPUTS =================")

    # STEP 1: MODELS
    bilstm_tags = bilstm_predict(words)
    rule_tags = rule_based_tag(sentence)
    context_tags = [tag for _, tag in tag_with_context(sentence)]

    print("\n🔹 BiLSTM:")
    for w, t in zip(words, bilstm_tags):
        print(f"{w:10} → {t}")

    print("\n🔹 Rule-Based:")
    for w, t in zip(words, rule_tags):
        print(f"{w:10} → {t}")

    print("\n🔹 Context-Based:")
    for w, t in zip(words, context_tags):
        print(f"{w:10} → {t}")

    print("\n================= DECISION ENGINE =================")

    final_tags = []
    debug_info = []

    stats = {
        "bilstm": 0,
        "rule": 0,
        "context": 0,
        "suffix": 0
    }

    for i, word in enumerate(words):
        log = []

        b_tag = bilstm_tags[i]
        r_tag = rule_tags[i]
        c_tag = context_tags[i]

        # CONFIDENCE
        b_conf = get_confidence(b_tag, "bilstm")
        r_conf = get_confidence(r_tag, "rule")
        c_conf = get_confidence(c_tag, "context")

        log.append(f"BiLSTM → {b_tag} ({b_conf})")
        log.append(f"Rule → {r_tag} ({r_conf})")
        log.append(f"Context → {c_tag} ({c_conf})")

        # VOTING
        tag = voting_decision(b_tag, r_tag, c_tag)
        log.append(f"Voting result → {tag}")

        # SUFFIX RULE
        new_tag, reason = apply_suffix_rules(word, tag)
        if new_tag != tag:
            log.append(f"{tag} → {new_tag} ({reason})")
            tag = new_tag
            stats["suffix"] += 1

        # CONTEXT RELATION
        if i > 0:
            prev_tag = final_tags[i - 1]

            if prev_tag == "PRON" and tag != "VERB":
                log.append("After PRON → forcing VERB")
                tag = "VERB"
                stats["context"] += 1

            if prev_tag == "DET" and tag != "ADJ":
                log.append("After DET → forcing NOUN")
                tag = "NOUN"
                stats["context"] += 1

        # FINAL SOURCE TRACKING
        if tag == b_tag:
            stats["bilstm"] += 1
        elif tag == r_tag:
            stats["rule"] += 1
        elif tag == c_tag:
            stats["context"] += 1

        final_tags.append(tag)

        debug_info.append({
            "word": word,
            "final_tag": tag,
            "log": log
        })

    return list(zip(words, final_tags)), debug_info, stats


# =========================================
# MAIN
# =========================================
if __name__ == "__main__":
    print("\n🚀 PRO MAX Hybrid POS Tagger Ready!\n")

    while True:
        sentence = input("Enter sentence (type exit): ")

        if sentence.lower() == "exit":
            break

        output, debug, stats = hybrid_tag(sentence)

        print("\n================= FINAL OUTPUT =================")
        print(" ".join([f"{w}/{t}" for w, t in output]))

        print("\n================= EXPLANATION =================")
        for info in debug:
            print(f"\n🔸 Word: {info['word']}")
            print(f"Final Tag: {info['final_tag']}")
            print("Steps:")
            for step in info["log"]:
                print(f"   → {step}")

        print("\n================= STATS =================")
        total = sum(stats.values())
        for k, v in stats.items():
            percent = (v / total * 100) if total > 0 else 0
            print(f"{k.upper():10}: {v} ({percent:.1f}%)")

        # SAVE LOG
        with open("tagger_log.txt", "a") as f:
            f.write(str(debug) + "\n")

        print("\n================================================\n")