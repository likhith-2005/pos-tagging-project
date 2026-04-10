import spacy
import re

# -------- LOAD MODEL --------
nlp = spacy.load("en_core_web_sm")


# ---------------- ADVANCED CUSTOM TAG SET ----------------
CUSTOM_TAGS = {
    "NOUN": ["N_COMMON", "N_PROPER", "N_ABSTRACT"],
    "VERB": ["V_MAIN", "V_AUX", "V_MODAL", "V_GERUND"],
    "PRON": ["PR_SUBJ", "PR_OBJ", "PR_POS"],
    "ADJ": ["ADJ_QUAL", "ADJ_COMP", "ADJ_SUP"],
    "ADV": ["ADV_MANNER", "ADV_TIME", "ADV_FREQ"],
    "DET": ["DET_DEF", "DET_INDEF"],
    "ADP": ["PREP"],
    "CONJ": ["CONJ_COORD", "CONJ_SUB"],
    "NUM": ["NUM_CARD", "NUM_ORD"]
}


# ---------------- RULE SET ----------------
AUX_VERBS = {"is","am","are","was","were","be","been","being","do","does","did","have","has","had"}
MODAL_VERBS = {"can","could","may","might","must","shall","should","will","would"}

SUBJECT_PRON = {"i","he","she","we","they"}
OBJECT_PRON = {"me","him","her","us","them"}
POSSESSIVE_PRON = {"my","your","his","her","its","our","their"}

TIME_ADVERBS = {"now","today","yesterday","tomorrow"}
FREQ_ADVERBS = {"always","often","sometimes","never"}

DEFINITE_DET = {"the"}
INDEFINITE_DET = {"a","an"}

COORD_CONJ = {"and","or","but"}
SUB_CONJ = {"because","although","if","while"}

ABSTRACT_HINT = {"idea","love","hate","freedom","happiness"}


# ---------------- MAPPING FUNCTION ----------------
def map_to_custom(token):
    word = token.text
    w = word.lower()
    upos = token.pos_
    dep = token.dep_

    # -------- NOUNS --------
    if upos in ["NOUN", "PROPN"]:
        if word[0].isupper():
            return "N_PROPER"
        elif w in ABSTRACT_HINT:
            return "N_ABSTRACT"
        else:
            return "N_COMMON"

    # -------- VERBS --------
    elif upos in ["VERB", "AUX"]:
        if w in AUX_VERBS:
            return "V_AUX"
        elif w in MODAL_VERBS:
            return "V_MODAL"
        elif w.endswith("ing"):
            return "V_GERUND"
        else:
            return "V_MAIN"

    # -------- PRONOUNS --------
    elif upos == "PRON":
        if w in SUBJECT_PRON or dep == "nsubj":
            return "PR_SUBJ"
        elif w in POSSESSIVE_PRON or dep == "poss":
            return "PR_POS"
        else:
            return "PR_OBJ"

    # -------- ADJECTIVES --------
    elif upos == "ADJ":
        if w.endswith("est"):
            return "ADJ_SUP"
        elif w.endswith("er"):
            return "ADJ_COMP"
        else:
            return "ADJ_QUAL"

    # -------- ADVERBS --------
    elif upos == "ADV":
        if w in TIME_ADVERBS:
            return "ADV_TIME"
        elif w in FREQ_ADVERBS:
            return "ADV_FREQ"
        else:
            return "ADV_MANNER"

    # -------- DETERMINERS --------
    elif upos == "DET":
        return "DET_DEF" if w in DEFINITE_DET else "DET_INDEF"

    # -------- PREPOSITIONS --------
    elif upos == "ADP":
        return "PREP"

    # -------- CONJUNCTIONS --------
    elif upos in ["CCONJ", "SCONJ"]:
        return "CONJ_COORD" if w in COORD_CONJ else "CONJ_SUB"

    # -------- NUMBERS --------
    elif upos == "NUM":
        if re.match(r"\d+", w):
            return "NUM_CARD"
        elif w.endswith(("st","nd","rd","th")):
            return "NUM_ORD"
        else:
            return "NUM_CARD"

    return upos


# ---------------- REQUIRED FUNCTION (IMPORTANT) ----------------
def custom_tag(sentence):
    doc = nlp(sentence)

    tags = []
    for token in doc:
        custom = map_to_custom(token)
        tags.append(custom)

    return tags


# ---------------- OPTIONAL (FOR TESTING) ----------------
def process_sentence(sentence):
    doc = nlp(sentence)

    print("\nWord\tUPOS\tDEP\tCUSTOM_TAG")
    print("--------------------------------------------------")

    for token in doc:
        word = token.text
        upos = token.pos_
        dep = token.dep_
        custom = map_to_custom(token)

        print(f"{word}\t{upos}\t{dep}\t{custom}")


# ---------------- RUN ----------------
if __name__ == "__main__":
    while True:
        sentence = input("\nEnter sentence (or type 'exit'): ")

        if sentence.lower() == "exit":
            break

        process_sentence(sentence)