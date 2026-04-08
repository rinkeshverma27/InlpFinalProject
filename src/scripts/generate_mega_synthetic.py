import pandas as pd
import random
import os
from pathlib import Path

def generate_mega_dataset():
    tickers = [
        "Reliance", "TCS", "HDFC Bank", "ICICI Bank", "Infosys", "SBI", "ITC", "L&T", 
        "Airtel", "Tata Motors", "रिलायंस", "इन्फोसिस", "एसबीआई", "टाटा", "अडानी"
    ]
    
    positives = [
        "{t} का मुनाफा {n}% बढ़ा", "{t} ke results bhot acche rahe", "{t} net profit jumps to {n} crore",
        "{t} को मिला {n} करोड़ का बड़ा प्रोजेक्ट", "{t} wins major order in USA", "FIIs increase stake in {t}",
        "{t} के शेयर रॉकेट बन गए", "{t} stock touches 52-week high", "{t} stock upgrade by Morgan Stanley",
        "{t} ने किया धमाकेदार डिविडेंड का ऐलान", "{t} announces bonus issue", "{t} board approves buyback",
        "गिरावट के बीच {t} ने बाजार को संभाला", "{t} management predicts strong growth next year",
        "{t} launches new AI product, stock bullish", "Major recovery seen in {t} shares today",
        "{t} signs strategic partnership with Apple", "{t} to enter 1 trillion club soon",
        "Breaking: Govt approves tax holiday for {t}", "Record production achieved by {t} in Q4",
        "Huge delivery volume seen in {t} counter", "{t} ke shares me breakout ki tayari",
        "Analysts give 'Strong Buy' rating to {t}", "{t} business expansion in Middle East",
        "Bull run continues for {t} stocks", "{t} reports zero debt status"
    ]
    
    negatives = [
        "{t} के मुनाफे में {n}% की गिरावट", "{t} profit falls below market expectation", "{t} reports disappointing loss",
        "SEBI ने {t} पर लगाया {n} करोड़ का जुर्माना", "RBI action against {t} creates panic", "{t} fraud investigation",
        "{t} के शेयर धड़ाम, 5% की गिरावट", "{t} stock crash worries retail investors", "{t} lower circuit locked",
        "{t} के सीईओ ने दिया इस्तीफा", "{t} factory fire causes major loss", "{t} rating downgraded to Sell",
        "{t} shares fall due to tech selloff", "Negative outlook for {t} in next quarter",
        "{t} margin pressure increases due to inflation", "{t} faces labor strike in major plant",
        "{t} to shut down {n} non-profitable stores", "Institutional selling seen in {t}",
        "{t} stock breaks critical support level", "Heavy selloff in {t} as promoters exit",
        "{t} stock price hits multi-year low", "Default risk warning for {t} bonds",
        "Panic selling in {t} after policy change", "{t} miss key ESG targets",
        "Weak demand for {t} products globally", "{t} audit shows accounting irregularities"
    ]

    # Mix and Match for variation
    data = []
    for _ in range(4000): # 4000 Positives
        t = random.choice(tickers)
        text = random.choice(positives).format(t=t, n=random.randint(5, 500))
        data.append({"text": text, "label": 1})
        
    for _ in range(4000): # 4000 Negatives
        t = random.choice(tickers)
        text = random.choice(negatives).format(t=t, n=random.randint(5, 500))
        data.append({"text": text, "label": 0})

    df = pd.DataFrame(data).sample(frac=1).reset_index(drop=True)
    out_path = Path("data/inputs/mega_synthetic_hindi_train.csv")
    os.makedirs(out_path.parent, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"✅ Generated {len(df)} samples at {out_path}")

if __name__ == "__main__":
    generate_mega_dataset()
