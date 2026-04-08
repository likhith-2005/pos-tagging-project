# POS Tag Analysis for Universal Dependencies Dataset

pos_count = {}

# Open the dataset file
with open("dataset/en_ewt-ud-train.conllu", "r", encoding="utf-8") as file:

    for line in file:

        # Skip comment lines and empty lines
        if line.startswith("#") or line.strip() == "":
            continue

        # Split the columns using tab
        parts = line.split("\t")

        # Extract POS tag (Column 4)
        pos = parts[3]

        # Count frequency of each POS tag
        if pos in pos_count:
            pos_count[pos] += 1
        else:
            pos_count[pos] = 1


# Print results
print("\nPOS Tag Distribution:\n")

for tag in sorted(pos_count):
    print(f"{tag} : {pos_count[tag]}")