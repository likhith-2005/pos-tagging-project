from pos_tagger import tag_with_rules, tag_with_context, tag_map

# -------- LOAD DATA --------
correct_a1 = 0
correct_a2 = 0
total = 0

# For Ground Truth Table (store sample)
gt_words = []
gt_actual = []
gt_pred_a1 = []
gt_pred_a2 = []

with open("dataset/en_ewt-ud-train.conllu", "r", encoding="utf-8") as file:
    sentence_words = []
    sentence_tags = []

    for line in file:
        if line.startswith("#") or line.strip() == "":
            
            if sentence_words:
                sentence = " ".join(sentence_words)

                # Predictions
                a1_tags = tag_with_rules(sentence)
                a2_tags = tag_with_context(sentence)

                for i in range(len(sentence_tags)):
                    
                    word = sentence_words[i]
                    actual = sentence_tags[i]

                    # Approach 1
                    if i < len(a1_tags):
                        pred1 = a1_tags[i]
                        if pred1 == actual:
                            correct_a1 += 1
                    else:
                        pred1 = "NA"

                    # Approach 2
                    if i < len(a2_tags):
                        pred2 = a2_tags[i]
                        if pred2 == actual:
                            correct_a2 += 1
                    else:
                        pred2 = "NA"

                    total += 1

                    # Store only first 20 rows (for display)
                    if len(gt_words) < 20:
                        gt_words.append(word)
                        gt_actual.append(actual)
                        gt_pred_a1.append(pred1)
                        gt_pred_a2.append(pred2)

                sentence_words = []
                sentence_tags = []

            continue

        parts = line.split("\t")
        if len(parts) < 4:
            continue

        word = parts[1].lower()
        ud_pos = parts[3]
        pos = tag_map.get(ud_pos, "X")

        sentence_words.append(word)
        sentence_tags.append(pos)


# -------- RESULTS --------
a1_acc = (correct_a1 / total) * 100
a2_acc = (correct_a2 / total) * 100

print("\n===== EVALUATION RESULTS =====")
print("Total Words:", total)

print("\nApproach 1 (Rule-Based):")
print("Correct:", correct_a1)
print("Accuracy:", round(a1_acc, 2), "%")

print("\nApproach 2 (Context-Based):")
print("Correct:", correct_a2)
print("Accuracy:", round(a2_acc, 2), "%")

print("\nImprovement:", round(a2_acc - a1_acc, 2), "%")


# -------- GROUND TRUTH TABLE --------
print("\n===== GROUND TRUTH TABLE (Sample) =====")
print("Word\tActual\tA1_Pred\tA2_Pred")

for w, a, p1, p2 in zip(gt_words, gt_actual, gt_pred_a1, gt_pred_a2):
    print(f"{w}\t{a}\t{p1}\t{p2}")