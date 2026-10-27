from src.data.loaders import load_all


# Load the unified HaluEval + TruthfulQA dataset
df = load_all(source="hf")


print("\n" + "=" * 70)
print("1. SUMMARIZATION DUPLICATE GROUPS")
print("=" * 70)

summ = df[df["subset"] == "summarization"].copy()

duplicate_groups = (
    summ.groupby("norm_key")
    .filter(lambda g: len(g) > 1)
    .sort_values("norm_key")
)

if duplicate_groups.empty:
    print("No duplicate summarization groups found.")
else:
    for key, group in duplicate_groups.groupby("norm_key"):

        print("\nDuplicate group:")
        print("-" * 70)

        for _, row in group.iterrows():
            print(f"UID: {row['uid']}")
            print(f"Source index: {row['source_idx']}")
            print(
                f"Context length: "
                f"{len(str(row['context']).split())} words"
            )

            print("\nContext:")
            print(row["context"][:1000])

            print("\nFaithful reference:")
            print(row["reference_correct"][0])

            print("\nHallucinated reference:")
            print(row["reference_incorrect"][0])

            print()


print("\n" + "=" * 70)
print("2. TRUTHFULQA IDENTICAL CORRECT/INCORRECT REFERENCES")
print("=" * 70)

truthful = df[df["subset"] == "truthfulqa"].copy()

conflicts = []

for _, row in truthful.iterrows():

    correct = {
        str(x).strip().lower()
        for x in row["reference_correct"]
        if str(x).strip()
    }

    incorrect = {
        str(x).strip().lower()
        for x in row["reference_incorrect"]
        if str(x).strip()
    }

    overlap = correct.intersection(incorrect)

    if overlap:
        conflicts.append((row, overlap))


if not conflicts:

    print("No conflicting references found.")

else:

    print(f"Found {len(conflicts)} conflicting questions.")

    for row, overlap in conflicts:

        print("\nQuestion:")
        print(row["prompt"])

        print("\nUID:")
        print(row["uid"])

        print("\nIdentical answer(s) appearing in BOTH lists:")

        for answer in overlap:
            print(f"- {answer}")

        print("\nCorrect references:")

        for answer in row["reference_correct"]:
            print(f"- {answer}")

        print("\nIncorrect references:")

        for answer in row["reference_incorrect"]:
            print(f"- {answer}")

        print("\n" + "-" * 70)