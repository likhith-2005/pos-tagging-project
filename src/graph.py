import matplotlib.pyplot as plt

tags = []
counts = []

with open("dataset/en_ewt-ud-train.conllu", "r", encoding="utf-8") as file:

    pos_count = {}

    for line in file:

        if line.startswith("#") or line.strip() == "":
            continue

        parts = line.split("\t")
        pos = parts[3]

        if pos in pos_count:
            pos_count[pos] += 1
        else:
            pos_count[pos] = 1


for tag, count in pos_count.items():
    tags.append(tag)
    counts.append(count)

plt.bar(tags, counts)

plt.title("POS Tag Distribution")
plt.xlabel("POS Tags")
plt.ylabel("Frequency")

plt.xticks(rotation=45)

plt.show()