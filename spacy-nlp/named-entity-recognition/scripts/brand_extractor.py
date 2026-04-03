#!/usr/bin/env python3
"""
Brand Extractor using Hybrid Approach
Combines pre-trained spaCy model with pattern matching for better coverage
No manual annotation required
"""

import re
from typing import Optional

import pandas as pd
import spacy

# Curated list of real brands
KNOWN_BRANDS = {}


def extract_brand_hybrid(text: str, nlp) -> Optional[str]:
    # sourcery skip: low-code-quality
    """
    Extract brand using hybrid approach: spaCy ORG + pattern matching + known brands
    """
    if not text or pd.isna(text):
        return None

    text_lower = text.lower()

    # Method 1: Check known brands first (most reliable)
    for brand in sorted(KNOWN_BRANDS, key=len, reverse=True):  # Longest first
        if brand in text_lower:
            pos = text_lower.find(brand)
            if pos != -1:
                # Extract with original case
                brand_text = text[pos : pos + len(brand)]
                return brand_text.title()

    # Method 2: Use spaCy ORG entities
    doc = nlp(text)
    if orgs := [ent.text for ent in doc.ents if ent.label_ == "ORG"]:
        # Filter and score ORG entities
        potential_brands = []
        for org in orgs:
            if len(org) < 2 or len(org) > 30:
                continue

            # Skip common non-brand patterns
            skip_patterns = [
                r"\b(and|or|the|of|for|with|by|in|on|at|to|from)\b",
                r"\b(design|studio|designs?|collection|series|set|kit|pack|bundle)\b",
                r"\b(inc|corp|ltd|llc|co|company|corporation)\b",
                r"\b(limited|unlimited|international|global|worldwide)\b",
                r"^\d+",  # Starts with number
                r"\d+$",  # Ends with number
            ]

            should_skip = False
            for pattern in skip_patterns:
                if re.search(pattern, org, re.IGNORECASE):
                    should_skip = True
                    break

            if should_skip:
                continue

            # Score the potential brand
            score = 0

            # Higher score if at beginning of text
            if text.lower().startswith(org.lower()):
                score += 10

            # Higher score for known brand patterns
            known_brands = [
                "apple",
                "samsung",
                "nike",
                "sony",
                "microsoft",
                "canon",
                "nintendo",
                "disney",
                "lego",
                "marvel",
            ]
            if any(brand in org.lower() for brand in known_brands):
                score += 15

            # Lower score if too long (likely not a brand)
            if len(org) > 20:
                score -= 5

            potential_brands.append((org, score))

        if potential_brands:
            # Return highest scoring brand
            best_brand = max(potential_brands, key=lambda x: x[1])
            return best_brand[0]

    # Method 3: Simple pattern matching as fallback
    # Pattern 1: Brand at the start followed by product
    pattern1 = r"^([A-Z][a-zA-Z0-9&\'\.\-\s]{1,15})\s+(?:by\s+)?[A-Z]"
    if match1 := re.search(pattern1, text):
        potential_brand = match1[1].strip()
        if len(potential_brand) > 1 and len(potential_brand) < 20:
            return potential_brand

    # Pattern 2: Brand + product type
    pattern2 = r"^([A-Z][a-zA-Z0-9&\'\.\-\s]{1,15})\s+(?:Set|Kit|Pack|Bundle|Console|Camera|Phone)"
    if match2 := re.search(pattern2, text):
        potential_brand = match2[1].strip()
        if len(potential_brand) > 1 and len(potential_brand) < 20:
            return potential_brand

    return None


def main():
    print("HYBRID BRAND EXTRACTION")

    # Load pre-trained model
    print("Loading pre-trained model...")
    nlp = spacy.load("en_core_web_sm")
    print("Model loaded!")

    # Load data
    print("Loading data...")
    df = pd.read_csv("data/marketing_sample.csv")
    print(f"Loaded {len(df)} records")

    # Extract brands
    print("\nExtracting brands...")
    brands = []

    for i, text in enumerate(df["asin_name"]):
        if i % 1000 == 0:
            print(f"Processed {i}/{len(df)} records...")

        brand = extract_brand_hybrid(text, nlp)
        brands.append(brand)

    # Add brands to dataframe
    df["brand"] = brands

    # Save results
    output_file = "data/marketing_sample_with_brands.csv"
    df.to_csv(output_file, index=False)

    # Show results
    print(f"\nResults saved to {output_file}")

    # Show statistics
    total_records = len(df)
    successful_extractions = len(df[df["brand"].notna()])
    success_rate = (successful_extractions / total_records) * 100

    print("\nEXTRACTION STATISTICS:")
    print(f"Total records: {total_records:,}")
    print(f"Successful extractions: {successful_extractions:,}")
    print(f"Success rate: {success_rate:.1f}%")

    # Show top brands
    print("\nTOP 15 BRANDS:")
    brand_counts = df["brand"].value_counts().head(15)
    for i, (brand, count) in enumerate(brand_counts.items(), 1):
        print(f"{i:2d}. {brand}: {count}")

    # Show sample extractions
    print("\nSAMPLE EXTRACTIONS:")
    sample_df = df[df["brand"].notna()].head(15)
    for i, row in sample_df.iterrows():
        print(f"Title: {row['asin_name'][:60]}...")
        print(f"Brand: {row['brand']}")


if __name__ == "__main__":
    main()
