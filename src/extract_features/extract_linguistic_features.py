from pathlib import Path
import re
import pandas as pd


# ============================================================
# 1. FILE PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "generation_qwen_calibration.parquet"
OUTPUT_FILE = PROJECT_ROOT / "data" / "a3" / "linguistic_features.parquet"


# ============================================================
# 2. REGULAR EXPRESSIONS
# ============================================================

WORD_PATTERN = re.compile(r"\b[\w'-]+\b")

NUMBER_PATTERN = re.compile(
    r"(?<!\w)[+-]?\d+(?:,\d{3})*(?:\.\d+)?%?(?!\w)"
)


# ============================================================
# 3. FEATURE 1: HEDGING FREQUENCY
# ============================================================

HEDGING_WORDS = {
    "maybe",
    "perhaps",
    "possibly",
    "probably",
    "apparently",
    "likely",
    "unlikely",
    "arguably",
    "seemingly",
    "reportedly",
    "approximately",
    "roughly",
    "typically",
    "potentially",
    "generally",
    "presumably",
    "ostensibly",
    "plausibly",
    "conceivably",
}

HEDGING_PHRASES = [
    "to the best of my knowledge",
    "as far as i know",
    "as far as i can tell",
    "based on the information provided",
    "based on available information",
    "there is a possibility that",
    "there is a chance that",
    "i am unable to verify",
    "i'm unable to verify",
    "i cannot be certain",
    "i can't be certain",
    "i do not have access to",
    "i don't have access to",
    "i do not know",
    "i don't know",
    "i cannot confirm",
    "i can't confirm",
    "i cannot verify",
    "i can't verify",
    "i may be mistaken",
    "i could be wrong",
    "i am uncertain",
    "i'm uncertain",
    "i am not sure",
    "i'm not sure",
    "based on my knowledge",
    "it is conceivable that",
    "it is possible",
    "it is probable",
    "it is likely",
    "it is unlikely",
    "my guess is",
    "i would guess",
    "i suspect",
    "i suppose",
    "i believe",
    "i think",
    "it appears",
    "it seems",
    "might be",
    "could be",
    "may be",
]


def calculate_hedging(text):
    """
    Count hedging words and phrases.

    Returns:
        hedge_count: Number of detected hedging expressions.
        hedge_rate: Hedge count divided by total word count.
    """

    if not isinstance(text, str):
        text = ""

    text_lower = text.lower()
    word_count = len(WORD_PATTERN.findall(text_lower))

    if word_count == 0:
        return 0, 0.0

    # Find phrases, longest first, so overlapping expressions
    # are not counted more than once.
    matched_positions = []
    hedge_count = 0

    for phrase in sorted(HEDGING_PHRASES, key=len, reverse=True):
        pattern = re.compile(
            r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"
        )

        for match in pattern.finditer(text_lower):
            start, end = match.span()

            overlaps = any(
                start < existing_end and end > existing_start
                for existing_start, existing_end in matched_positions
            )

            if not overlaps:
                matched_positions.append((start, end))
                hedge_count += 1

    # Count individual hedge words not already included in phrases.
    for word in HEDGING_WORDS:
        pattern = re.compile(
            r"(?<!\w)" + re.escape(word) + r"(?!\w)"
        )

        for match in pattern.finditer(text_lower):
            start, end = match.span()

            overlaps = any(
                start < existing_end and end > existing_start
                for existing_start, existing_end in matched_positions
            )

            if not overlaps:
                matched_positions.append((start, end))
                hedge_count += 1

    hedge_rate = hedge_count / word_count

    return hedge_count, hedge_rate


# ============================================================
# 4. FEATURE 2: ENTITY / NUMERAL DENSITY
# ============================================================

def calculate_entity_numeral_density(text):
    """
    Estimate named entities using capitalization rules and
    count numeric expressions using a regular expression.

    This is a heuristic approximation, not true NER.
    """

    if not isinstance(text, str):
        text = ""

    words = WORD_PATTERN.findall(text)
    word_count = len(words)

    if word_count == 0:
        return 0, 0, 0.0

    # Estimate entity mentions from capitalized words.
    # Skip the first word of each sentence to reduce the
    # chance of counting sentence-initial capitalization.
    entity_count = 0

    for sentence in re.split(r"[.!?]+", text):
        sentence_words = WORD_PATTERN.findall(sentence)

        for word in sentence_words[1:]:
            if word[:1].isupper() and word.isalpha():
                entity_count += 1

    # Count numeric expressions, including decimals and percentages.
    numeral_count = len(NUMBER_PATTERN.findall(text))

    entity_numeral_density = (
        entity_count + numeral_count
    ) / word_count

    return (
        entity_count,
        numeral_count,
        entity_numeral_density,
    )


# ============================================================
# 5. FEATURE 3: ASSERTION DENSITY
# ============================================================

ASSERTION_CUES = {
    "is",
    "are",
    "was",
    "were",
    "has",
    "have",
    "had",
    "does",
    "did",
    "means",
    "shows",
    "states",
    "contains",
    "includes",
    "causes",
    "indicates",
    "proves",
    "founded",
    "established",
    "occurred",
    "happened",
}


def calculate_assertion_density(text):
    """
    Estimate assertion density using predefined cue words.

    A sentence is counted as an assertion sentence when it
    contains at least one assertion cue.
    """

    if not isinstance(text, str):
        text = ""

    sentences = [
        sentence.strip()
        for sentence in re.split(r"[.!?]+", text)
        if sentence.strip()
    ]

    if not sentences:
        return 0, 0.0

    assertion_sentence_count = 0

    for sentence in sentences:
        words = {
            word.lower()
            for word in WORD_PATTERN.findall(sentence)
        }

        if words.intersection(ASSERTION_CUES):
            assertion_sentence_count += 1

    assertion_density = (
        assertion_sentence_count / len(sentences)
    )

    return assertion_sentence_count, assertion_density


# ============================================================
# 6. FEATURE 4: REPETITION
# ============================================================

def calculate_repetition(text):
    """
    Calculate repeated word count and repetition rate.

    Repeated word count = total tokens - unique tokens.
    """

    if not isinstance(text, str):
        text = ""

    words = [
        word.lower()
        for word in WORD_PATTERN.findall(text)
    ]

    total_words = len(words)

    if total_words == 0:
        return 0, 0.0

    unique_words = len(set(words))

    repeated_word_count = total_words - unique_words
    repetition_rate = repeated_word_count / total_words

    return repeated_word_count, repetition_rate


# ============================================================
# 7. EXTRACT ALL FOUR FEATURE GROUPS
# ============================================================

def extract_features(text):
    """
    Extract all four B2 linguistic feature groups
    from one generated answer.
    """

    if not isinstance(text, str):
        text = ""

    word_count = len(WORD_PATTERN.findall(text))

    # Feature 1: Hedging
    hedge_count, hedge_rate = calculate_hedging(text)

    # Feature 2: Entity / numeral density
    (
        entity_mention_count_approx,
        numeral_count,
        entity_numeral_density_approx,
    ) = calculate_entity_numeral_density(text)

    # Feature 3: Assertion density
    (
        assertion_sentence_count_approx,
        assertion_density_approx,
    ) = calculate_assertion_density(text)

    # Feature 4: Repetition
    repeated_word_count, repetition_rate = (
        calculate_repetition(text)
    )

    return {
        "word_count": word_count,
        "hedge_count": hedge_count,
        "hedge_rate": hedge_rate,
        "entity_mention_count_approx": entity_mention_count_approx,
        "numeral_count": numeral_count,
        "entity_numeral_density_approx": entity_numeral_density_approx,
        "assertion_sentence_count_approx": assertion_sentence_count_approx,
        "assertion_density_approx": assertion_density_approx,
        "repeated_word_count": repeated_word_count,
        "repetition_rate": repetition_rate,
    }


# ============================================================
# 8. MAIN PIPELINE
# ============================================================

def main():

    print("=" * 65)
    print("B2 LINGUISTIC FEATURE EXTRACTION")
    print("=" * 65)

    # Check the input file.
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    print(f"\nReading input: {INPUT_FILE}")

    df = pd.read_parquet(INPUT_FILE)

    print(f"Total rows: {len(df)}")

    # Check required columns.
    required_columns = {
        "uid",
        "generated_answer",
    }

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    # Prevent duplicate IDs from silently entering the output.
    if df["uid"].duplicated().any():
        duplicate_count = int(df["uid"].duplicated().sum())

        raise ValueError(
            f"Found {duplicate_count} duplicate UIDs."
        )

    # Keep only metadata columns that exist in the input.
    metadata_columns = ["uid"]

    for column in ["dataset", "subset"]:
        if column in df.columns:
            metadata_columns.append(column)

    metadata = df[metadata_columns].copy()

    # Handle missing answers.
    answers = (
        df["generated_answer"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    # Create the output directory.
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Process every answer.
    feature_rows = []

    total = len(answers)

    print("\nExtracting linguistic features...")

    for i, answer in enumerate(answers, start=1):
        features = extract_features(answer)
        feature_rows.append(features)

        if i % 2000 == 0 or i == total:
            print(f"Processed {i}/{total} answers")

    # Convert extracted features to a DataFrame.
    features_df = pd.DataFrame(feature_rows)

    # Combine metadata and feature columns.
    output_df = pd.concat(
        [
            metadata.reset_index(drop=True),
            features_df.reset_index(drop=True),
        ],
        axis=1,
    )

    # Validate output.
    if len(output_df) != len(df):
        raise ValueError(
            "Output row count does not match input row count."
        )

    if output_df["uid"].duplicated().any():
        raise ValueError(
            "Duplicate UIDs detected in output."
        )

    # Save output.
    output_df.to_parquet(
        OUTPUT_FILE,
        index=False,
    )

    print("\n" + "=" * 65)
    print("FEATURE EXTRACTION COMPLETED")
    print("=" * 65)

    print(f"Input rows: {len(df)}")
    print(f"Output rows: {len(output_df)}")
    print(f"Output columns: {len(output_df.columns)}")
    print(f"\nSaved to:\n{OUTPUT_FILE}")

    print("\nFeature columns:")
    for column in features_df.columns:
        print(f"  - {column}")

    print("\nFirst five rows:")
    print(output_df.head().to_string(index=False))

    print("\nMissing values per column:")
    print(output_df.isna().sum())


# ============================================================
# 9. RUN SCRIPT
# ============================================================

if __name__ == "__main__":
    main()