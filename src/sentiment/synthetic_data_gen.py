import pandas as pd
import random
from pathlib import Path
import sys
import os

# Resolve paths locally (src.utils.paths module does not exist as a standalone file)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

def generate_synthetic_data():
    print("Generating synthetic Hindi/Hinglish financial dataset...")

    # Define core components for combinatorics
    nifty_stocks = [
        "Reliance", "TCS", "HDFC Bank", "ICICI Bank", "Infosys",
        "SBI", "Bharti Airtel", "ITC", "L&T", "Bajaj Finance",
        "Adani Enterprises", "Tata Motors", "Mahindra & Mahindra",
        "Maruti Suzuki", "Sun Pharma", "रिलायंस", "इन्फोसिस", "एसबीआई",
        "एचडीएफसी", "टाटा मोटर्स"
    ]

    # POSITIVE FRAGMENTS (Hindi & Hinglish)
    positive_phrases = [
        "के मुनाफे में भारी उछाल", "में जबरदस्त तेजी देखने को मिली", "के शेयर रॉकेट बन गए",
        "ने निवेशकों को मालामाल किया", "के Q3 नतीजे शानदार रहे", "को बड़ा प्रोजेक्ट मिला",
        "का मुनाफा 20% बढ़ा", "में आज अच्छी खरीददारी दिखी", "ने बाजार को सपोर्ट किया",
        "ke profit me acchi growth aayi", "me jabardast rally dekhi gayi", "ke results expectation se better aaye",
        "ko naya contract mila hai", "stock uppar gaya hai aaj", "me strong buying momentum hai",
        "ne 52-week high touch kiya", "ki aamdani me bhot badhotri hui"
    ]

    # NEGATIVE FRAGMENTS (Hindi & Hinglish)
    negative_phrases = [
        "में भारी गिरावट दर्ज की गई", "के शेयर लुढ़क गए", "को भारी नुकसान हुआ",
        "के तिमाही नतीजे बेहद खराब रहे", "पर सेबी ने जुर्माना लगाया", "का स्टॉक धड़ाम हो गया",
        "में बिकवाली का दबाव दिखा", "का मुनाफा 15% घट गया", "को बड़ा घाटा हुआ",
        "ke shares crash ho gaye", "me bohot bekwali hui aaj", "ke results disappointing the",
        "par RBI ne strict action liya", "stock me lower circuit lag gaya", "ki rating downgrade ki gayi",
        "ke profits gir gaye hain is quarter me", "me panic selling dekhne ko mili"
    ]

    # MIXED / GENERIC FRAGMENTS (to create variety)
    prefixes = ["आज ", "मार्केट ओपन होते ही ", "रिपोर्ट्स के अनुसार, ", "Experts ka manna hai ki ", "Breaking: ", ""]
    suffixes = ["।", "!", "...", "", "."]

    dataset = []

    # Generate Positives
    for _ in range(500):
        stock = random.choice(nifty_stocks)
        phrase = random.choice(positive_phrases)
        prefix = random.choice(prefixes)
        suffix = random.choice(suffixes)
        headline = f"{prefix}{stock} {phrase}{suffix}".strip()
        dataset.append({"text": headline, "label": 1})

    # Generate Negatives
    for _ in range(500):
        stock = random.choice(nifty_stocks)
        phrase = random.choice(negative_phrases)
        prefix = random.choice(prefixes)
        suffix = random.choice(suffixes)
        headline = f"{prefix}{stock} {phrase}{suffix}".strip()
        dataset.append({"text": headline, "label": 0})

    # Shuffle the dataset to ensure random distribution during training
    random.shuffle(dataset)

    # Convert to DataFrame
    df = pd.DataFrame(dataset)
    
    # Save to CSV
    output_path = RAW_DATA_DIR / "synthetic_hindi_financial_train.csv"
    df.to_csv(output_path, index=False)
    
    print(f"Successfully generated {len(df)} synthetic headlines.")
    print(f"Saved dataset to: {output_path}")
    
    print("\nSample Data:")
    print(df.head(10).to_string(index=False))

if __name__ == "__main__":
    generate_synthetic_data()
