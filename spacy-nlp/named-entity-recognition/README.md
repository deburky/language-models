# Brand Extractor

Extracts brand names from e-commerce product titles using spaCy's pre-trained NER model, with a regex fallback for titles following a "Brand Product" structure. No manual annotation required.

## Usage

```bash
# Download spaCy model
uv run python -m spacy download en_core_web_sm

# Extract brands
uv run python scripts/brand_extractor.py

# View results
uv run python scripts/inference.py
```

## How It Works

The extractor uses spaCy's `en_core_web_sm` model to identify ORG entities in product titles, filters out common false positives (legal suffixes, overly long spans), and falls back to a capitalization pattern match when NER finds nothing. Results are scored by position and entity length.

## Configuration

Edit the config section in `scripts/brand_extractor.py`:

```python
CSV_IN = "data/marketing_sample.csv"
CSV_OUT = "data/marketing_sample_with_brands.csv"
TEXT_COL = "asin_name"
```

## Example

Input:
```
LEGO Minecraft Creeper BigFig and Ocelot Characters 21156 Building Kit
Disney Princess Belle Deluxe Costume for Girls
Fisher-Price Laugh & Learn Smart Stages Chair
```

Output:
```
Brand: LEGO Minecraft Creeper BigFig
Brand: Disney Princess
Brand: Fisher-Price
```

## Project Structure

```
├── scripts/
│   ├── brand_extractor.py    # Main extraction script
│   └── inference.py          # View and test results
└── data/
    └── marketing_sample.csv  # Input data
```

## Further Reading

A similar approach is described in Souames and Mohammedi, "An end to end approach to brand recognition in product titles using BI-LSTM-CRF" (2022), available via the [brand-ner](https://github.com/annis-souames/brand-ner) repository. Their paper extends the heuristic baseline with a trained CRF layer for sequence labeling.
