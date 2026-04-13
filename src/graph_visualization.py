import matplotlib.pyplot as plt
from collections import Counter

# -------- READ FILE (FIXED) --------
def read_tags(file_path):
    tags = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            words = line.strip().split()
            for w in words:
                if "/" in w:
                    tag = w.split("/")[-1]   # ✅ extract only tag
                    tags.append(tag)
    return tags


# -------- LOAD DATA --------
a1_tags = read_tags("results/tagged_output_rules.txt")
a2_tags = read_tags("results/tagged_output_context.txt")


# -------- FUNCTION TO PLOT CLEAN GRAPH --------
def plot_distribution(tags, title):
    tag_count = Counter(tags)

    # 🔹 take top 10 tags only (clean graph)
    most_common = tag_count.most_common(10)
    labels = [x[0] for x in most_common]
    values = [x[1] for x in most_common]

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)
    plt.title(title)
    plt.xlabel("Tags")
    plt.ylabel("Count")

    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


# -------- GRAPH 1: TAG DISTRIBUTION --------
plot_distribution(a1_tags, "Rule-Based Tag Distribution (A1)")
plot_distribution(a2_tags, "Context-Based Tag Distribution (A2)")


# -------- ACCURACY (update with real values) --------
a1_acc = 25.0
a2_acc = 47.0


# -------- GRAPH 2: ACCURACY COMPARISON --------
plt.figure(figsize=(6, 4))
plt.bar(["Rule-Based", "Context-Based"], [a1_acc, a2_acc])
plt.title("Accuracy Comparison")
plt.ylabel("Accuracy (%)")
plt.tight_layout()
plt.show()


# -------- GRAPH 3: IMPROVEMENT --------
improvement = a2_acc - a1_acc

plt.figure(figsize=(5, 4))
plt.bar(["Improvement"], [improvement])
plt.title("Performance Improvement")
plt.ylabel("Increase in Accuracy (%)")
plt.tight_layout()
plt.show()