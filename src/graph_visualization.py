import matplotlib.pyplot as plt
from collections import Counter

# -------- READ FILE --------
def read_tags(file_path):
    tags = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                tags.append(parts[1])
    return tags


# -------- LOAD DATA --------
a1_tags = read_tags("results/tagged_output_rules.txt")
a2_tags = read_tags("results/tagged_output_context.txt")

# -------- TAG DISTRIBUTION --------
a1_count = Counter(a1_tags)
a2_count = Counter(a2_tags)

# -------- GRAPH 1: TAG DISTRIBUTION --------
plt.figure()
plt.bar(a1_count.keys(), a1_count.values())
plt.title("Rule-Based Tag Distribution (A1)")
plt.xlabel("Tags")
plt.ylabel("Count")
plt.show()

plt.figure()
plt.bar(a2_count.keys(), a2_count.values())
plt.title("Context-Based Tag Distribution (A2)")
plt.xlabel("Tags")
plt.ylabel("Count")
plt.show()


# -------- ACCURACY (example values, replace if needed) --------
# You can copy from your evaluation.py output
a1_acc = 25.0
a2_acc = 47.0

# -------- GRAPH 2: ACCURACY COMPARISON --------
plt.figure()
plt.bar(["A1 (Rule-Based)", "A2 (Context-Based)"], [a1_acc, a2_acc])
plt.title("Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.show()


# -------- GRAPH 3: IMPROVEMENT --------
improvement = a2_acc - a1_acc

plt.figure()
plt.bar(["Improvement"], [improvement])
plt.title("Performance Improvement")
plt.ylabel("Increase in Accuracy (%)")
plt.show()