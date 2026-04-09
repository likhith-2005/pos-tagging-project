from pos_tagger import tag_with_rules, tag_with_context

# POS Tag Analysis + Testing Comparison

pos_count = {}
total_words = 0

# -------------------------------
# PART 1: POS DISTRIBUTION
# -------------------------------
with open("dataset/en_ewt-ud-train.conllu", "r", encoding="utf-8") as file:
    
    for line in file:
        if line.startswith("#") or line.strip() == "":
            continue
        
        parts = line.split("\t")
        
        if len(parts) < 4:
            continue
        
        pos = parts[3]
        total_words += 1
        
        pos_count[pos] = pos_count.get(pos, 0) + 1


print("\nPOS Tag Distribution:\n")
for tag in sorted(pos_count):
    print(f"{tag} : {pos_count[tag]}")

print("\nTotal Words:", total_words)


# -------------------------------
# PART 2: TEST DATA
# -------------------------------
test_data = [
    ("Book a ticket", ["V", "D", "N"]),
    ("She reads a book", ["PR", "V", "D", "N"]),
    ("He is running fast", ["PR", "V", "V", "ADV"])
]


# -------------------------------
# PART 3: COMPARISON + ACCURACY
# -------------------------------
correct_a1 = 0
correct_a2 = 0

print("\n--- Comparison Output ---\n")

for sentence, expected in test_data:
    
    a1_tags = tag_with_rules(sentence)
    a2_tags = tag_with_context(sentence)
    
    print("Sentence:", sentence)
    print("Expected:", expected)
    print("A1:", a1_tags)
    print("A2:", a2_tags)
    print("-" * 40)
    
    if a1_tags == expected:
        correct_a1 += 1
    
    if a2_tags == expected:
        correct_a2 += 1


# -------------------------------
# PART 4: ACCURACY
# -------------------------------
total = len(test_data)

a1_acc = (correct_a1 / total) * 100
a2_acc = (correct_a2 / total) * 100

print("\n--- Accuracy ---")
print(f"Approach 1 Accuracy: {a1_acc:.2f}%")
print(f"Approach 2 Accuracy: {a2_acc:.2f}%")
print(f"Improvement: {a2_acc - a1_acc:.2f}%")