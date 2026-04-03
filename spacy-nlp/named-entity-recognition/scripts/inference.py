#!/usr/bin/env python3
"""
Inference Script for Brand Extraction
Uses trained spaCy model or pattern matching for brand extraction
"""

import re
import time
import warnings
from typing import List

import pandas as pd
import spacy
from tqdm import tqdm

warnings.filterwarnings("ignore")

# Config
CSV_IN = "data/marketing_sample.csv"
CSV_OUT = "data/marketing_sample_with_brands_inference.csv"
TEXT_COL = "asin_name"
MODEL_DIR = "models/spacy_brand_model"
MODEL_PATH = f"{MODEL_DIR}/model/model-last"

# Brand patterns (same as training)
BRAND_PATTERNS = {
    "amazon_private": [
        r"Amazon\s+\w+",
        r"Solimo\b",
        r"Goodthreads\b",
        r"Lark\s+&\s+Ro\b",
        r"Stone\s+&\s+Beam\b",
        r"Daily\s+Ritual\b",
    ],
    "leading_brand": [
        r"^([A-Z][a-zA-Z0-9&\'\.\-\s]{1,30})\s+(?:by\s+)?[A-Z]",
        r"^([A-Z][a-zA-Z0-9&\'\.\-\s]{1,20})\s+(?:Set|Kit|Pack|Bundle)",
    ],
    "by_brand": [
        r"\bby\s+([A-Z][a-zA-Z0-9&\'\.\-\s]{1,30})\b",
        r"\bfrom\s+([A-Z][a-zA-Z0-9&\'\.\-\s]{1,30})\b",
    ],
    "trademark": [
        r"([A-Za-z0-9&\'\.\-\s]+)\s?[®™]",
    ],
    "gaming": [
        r"\b(?:Nintendo|PlayStation|Xbox|Steam|Epic\s+Games)\b",
        r"\b(?:Pokemon|Pokémon|Mario|Zelda|Sonic)\b",
    ],
    "fashion": [
        r"\b(?:Nike|Adidas|Puma|Reebok|Under\s+Armour)\b",
        r"\b(?:Gap|H&M|Zara|Uniqlo|Forever\s+21)\b",
    ],
}

# Common non-brand words
COMMON_NON_BRAND = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "from",
    "official",
    "kids",
    "child",
    "children",
    "men",
    "women",
    "adult",
    "jr",
    "jr.",
    "boys",
    "girls",
    "set",
    "pack",
    "bundle",
    "kit",
    "toy",
    "game",
    "book",
    "dvd",
    "cd",
    "blu-ray",
    "digital",
    "download",
    "online",
    "free",
    "sale",
    "discount",
    "clearance",
    "new",
    "used",
    "refurbished",
    "vintage",
}


def load_trained_model():
    """Load trained spaCy model if available"""
    try:
        nlp = spacy.load(MODEL_PATH)
        print(f"Loaded trained model from {MODEL_PATH}")
        return nlp
    except OSError:
        print(f"No trained model found at {MODEL_PATH}")
        print("Using pattern-based extraction instead...")
        return None


def extract_brand_patterns(title: str) -> str:
    """Extract brand using pattern matching"""
    if not title or not isinstance(title, str):
        return ""

    title_clean = title.strip()
    brands_found = []

    # Apply all patterns
    for _, patterns in BRAND_PATTERNS.items():
        for pattern in patterns:
            matches = re.findall(pattern, title_clean, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match = match[0]  # Take first group

                brand = match.strip().strip("®™\"'")
                if (
                    brand
                    and len(brand) > 1
                    and not brand.isdigit()
                    and brand.lower() not in COMMON_NON_BRAND
                    and brand not in brands_found
                ):
                    brands_found.append(brand)

    # Return the first (most likely) brand
    return brands_found[0] if brands_found else ""


def extract_brand_spacy(title: str, nlp) -> str:
    """Extract brand using trained spaCy model"""
    if not title or not isinstance(title, str):
        return ""

    doc = nlp(title)
    brand_entities = [ent.text for ent in doc.ents if ent.label_ == "BRAND"]

    return brand_entities[0] if brand_entities else ""


def extract_brands_hybrid(df: pd.DataFrame, nlp=None) -> List[str]:
    """Extract brands using hybrid approach (spaCy + patterns)"""
    results = []

    print("Extracting brands...")
    for text in tqdm(df[TEXT_COL], desc="Processing titles"):
        if not text or pd.isna(text):
            results.append("")
            continue

        # Try spaCy model first if available
        if nlp is not None:
            brand = extract_brand_spacy(str(text), nlp)
            if brand:
                results.append(brand)
                continue

        # Fallback to pattern matching
        brand = extract_brand_patterns(str(text))
        results.append(brand)

    return results


def evaluate_results(df: pd.DataFrame, brands: List[str]):
    """Evaluate and display results"""
    df["brand"] = brands

    # Statistics
    total_records = len(df)
    non_empty_brands = sum(bool(b) for b in brands)
    blank_rate = (total_records - non_empty_brands) / total_records

    print("\nRESULTS:")
    print(f"Total records: {total_records:,}")
    print(f"Brands extracted: {non_empty_brands:,}")
    print(f"Blank rate: {blank_rate:.1%}")
    print(f"Success rate: {(1 - blank_rate):.1%}")

    # Show sample results
    print("\nSAMPLE RESULTS:")
    sample_df = df[df["brand"] != ""].head(15)
    for idx, row in sample_df.iterrows():
        title = str(row[TEXT_COL])
        brand = row["brand"]
        print(f"Title: {title[:70]}{'...' if len(title) > 70 else ''}")
        print(f"Brand: {brand}")

    # Show most common brands
    brand_counts = df[df["brand"] != ""]["brand"].value_counts().head(10)
    print("\nTOP 10 BRANDS:")
    for brand, count in brand_counts.items():
        print(f"{brand}: {count}")


def main():
    """Main inference function"""
    print("=== Brand Extraction Inference ===")

    # Load data
    print("Loading data...")
    df = pd.read_csv(CSV_IN)
    if TEXT_COL not in df.columns:
        raise ValueError(
            f"Column '{TEXT_COL}' not found. Available: {list(df.columns)}"
        )

    print(f"Processing {len(df):,} records")

    # Load model
    nlp = load_trained_model()

    # Extract brands
    start_time = time.time()
    brands = extract_brands_hybrid(df, nlp)
    elapsed_time = time.time() - start_time

    print(f"Processing completed in {elapsed_time:.1f} seconds")
    print(f"Speed: {len(df) / elapsed_time:.1f} records/second")

    # Evaluate results
    evaluate_results(df, brands)

    # Save results
    df.to_csv(CSV_OUT, index=False)
    print(f"\nResults saved to {CSV_OUT}")


if __name__ == "__main__":
    main()
