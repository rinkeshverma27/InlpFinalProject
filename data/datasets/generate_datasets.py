"""
generate_datasets.py
--------------------
Reproducible generator for two financial sentiment datasets:
  1. hindi_hinglish_financial.csv  — 5000+ samples (Hindi / Hinglish / code-switched)
  2. english_financial.csv         — 5000+ samples (English financial news)

Labels:  positive=1  |  neutral=0  |  negative=-1

Run:  python data/datasets/generate_datasets.py
"""

import csv
import itertools
import random
import pathlib
import hashlib

SEED = 42
random.seed(SEED)

OUT_DIR = pathlib.Path(__file__).parent
TARGET_PER_LABEL = 1700   # ~5100 total per dataset (slight over-shoot for variety)

# ─────────────────────────────────────────────────────────────────────────────
# SHARED VOCABULARY
# ─────────────────────────────────────────────────────────────────────────────
NIFTY_STOCKS_EN = [
    "Reliance Industries", "TCS", "HDFC Bank", "Infosys", "ICICI Bank",
    "Hindustan Unilever", "Bajaj Finance", "Wipro", "Maruti Suzuki", "Axis Bank",
    "Kotak Mahindra Bank", "L&T", "Asian Paints", "HCL Technologies", "SBI",
    "Sun Pharma", "Titan Company", "UltraTech Cement", "Nestle India",
    "JSW Steel", "Power Grid", "Tata Motors", "NTPC", "Bharti Airtel",
    "Adani Ports", "Adani Enterprises", "Divis Laboratories", "Dr Reddys",
    "Cipla", "Eicher Motors", "BPCL", "IndusInd Bank", "Coal India",
    "Tech Mahindra", "Grasim Industries", "ONGC", "Hindalco", "Hero MotoCorp",
    "Bajaj Auto", "Tata Consumer", "Shree Cement", "SBI Life", "HDFC Life",
    "Apollo Hospitals", "Tata Steel", "Britannia", "ITC", "Pidilite Industries",
]

NIFTY_STOCKS_HI = [
    "रिलायंस", "टीसीएस", "एचडीएफसी बैंक", "इन्फोसिस", "आईसीआईसीआई बैंक",
    "हिंदुस्तान यूनिलीवर", "बजाज फाइनेंस", "विप्रो", "मारुति सुजुकी", "एक्सिस बैंक",
    "कोटक महिंद्रा बैंक", "एलएंडटी", "एशियन पेंट्स", "एचसीएल टेक", "एसबीआई",
    "सन फार्मा", "टाइटन", "अल्ट्राटेक सीमेंट", "नेस्ले", "जेएसडब्ल्यू स्टील",
    "पावर ग्रिड", "टाटा मोटर्स", "एनटीपीसी", "भारती एयरटेल", "अदानी पोर्ट्स",
    "अदानी एंटर्प्राइजेज", "डॉ. रेड्डी", "सिप्ला", "आईटीसी", "ओएनजीसी",
]

PCTS_UP   = ["2.1%", "3.4%", "1.8%", "4.2%", "5.0%", "2.7%", "6.1%", "1.5%", "3.9%", "2.3%",
             "7.2%", "4.8%", "1.1%", "8.3%", "3.2%"]
PCTS_DOWN = ["2.4%", "3.1%", "1.6%", "4.5%", "5.3%", "2.9%", "6.3%", "1.2%", "3.7%", "2.6%",
             "7.5%", "4.1%", "1.4%", "8.6%", "3.3%"]
QUARTERS  = ["Q1", "Q2", "Q3", "Q4"]
FY        = ["FY23", "FY24", "FY25", "FY26"]
SECTORS   = ["IT sector", "banking sector", "pharma sector", "auto sector",
             "FMCG sector", "metals sector", "energy sector", "cement sector",
             "renewable energy sector", "telecom sector", "financial services sector"]
SECTORS_HI= ["आईटी सेक्टर", "बैंकिंग सेक्टर", "फार्मा सेक्टर", "ऑटो सेक्टर",
              "एफएमसीजी सेक्टर", "मेटल सेक्टर", "एनर्जी सेक्टर", "टेलीकॉम सेक्टर"]

# Extra variation slots
DEAL_SIZES = ["₹500 crore", "₹1,200 crore", "₹750 crore", "$800 million", "$1.2 billion",
              "₹2,000 crore", "₹300 crore", "$500 million", "₹1,500 crore"]
PROFIT_DROPS = ["15%", "22%", "18%", "30%", "12%", "25%", "35%", "10%"]
PROFIT_RISES = ["18%", "25%", "32%", "40%", "15%", "28%", "12%", "45%"]
VOLGROWTH   = ["8%", "12%", "15%", "20%", "6%", "10%", "18%"]
MARGINBPS   = ["150 bps", "200 bps", "300 bps", "80 bps", "120 bps", "250 bps"]
TIMEREFS    = ["this quarter", "this week", "in morning trade", "on Wednesday",
               "in Thursday session", "year-to-date", "in a month"]
MONTHS_EN   = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]
MONTHS_HI   = ["जनवरी", "फरवरी", "मार्च", "अप्रैल", "मई", "जून",
               "जुलाई", "अगस्त", "सितंबर", "अक्टूबर", "नवंबर", "दिसंबर"]


# ─────────────────────────────────────────────────────────────────────────────
# ENGLISH DATASET
# ─────────────────────────────────────────────────────────────────────────────
EN_POSITIVE_TEMPLATES = [
    "{stock} surges {pct} after strong {qtr} {fy} earnings beat estimates",
    "{stock} shares jump {pct} as net profit rises significantly year-on-year",
    "Bullish outlook for {stock} as revenue growth exceeds analyst expectations",
    "{stock} hits 52-week high after robust quarterly results",
    "FIIs pour money into {stock}; stock rallies {pct}",
    "{stock} wins major government contract, shares soar {pct}",
    "Strong demand outlook lifts {stock} to record high",
    "{stock} reports highest ever quarterly revenue, stock up {pct}",
    "Analysts upgrade {stock} to 'buy' after management guidance raise",
    "{stock} announces dividend payout, stock gains {pct}",
    "RBI rate cut boosts {stock} and broader {sector}",
    "{stock} exports hit all-time high; market sentiment very bullish",
    "Market rally: {stock} leads Nifty 50 gains with {pct} jump",
    "{stock} global expansion plan cheers investors; stock surges",
    "Strong GST collection data boosts {sector} stocks led by {stock}",
    "SEBI approves {stock} buyback plan; shares rally {pct}",
    "DII inflows push {stock} above key resistance; bullish momentum intact",
    "{stock} secures $1 billion deal; shares hit upper circuit",
    "Inflation falls to 4-year low, {sector} surges on positive macro data",
    "{stock} beats street estimates on all key metrics, stock up {pct}",
    "India GDP growth surprises positively, lifting {stock} and Nifty",
    "{stock} margin expansion beats expectations; brokerages raise targets",
    "Rebound in global markets drives {stock} up {pct}",
    "{stock} subsidiary IPO oversubscribed 40x; parent stock rallies",
    "Credit rating upgrade for {stock} lifts debt and equity",
    "{stock} posts lowest ever NPA ratio; banking sector sentiment strong",
    "Budget highlights boost infrastructure — {stock} gains {pct}",
    "Recovery in rural demand lifts {stock} FMCG volumes by 12%",
    "{stock} capacity expansion on track; production numbers beat forecast",
    "India PMI at 5-year high; {sector} stocks including {stock} surge",
]

EN_NEGATIVE_TEMPLATES = [
    "{stock} tanks {pct} after disappointing {qtr} {fy} results miss estimates",
    "{stock} shares crash on weak volume guidance and rising costs",
    "Sell-off in {stock} as management lowers earnings outlook for next quarter",
    "{stock} falls {pct} after FIIs dump shares in large block deal",
    "Weak global cues drag {stock} to 52-week low",
    "SEBI serves notice to {stock} promoters on insider trading allegations",
    "{stock} reports widening losses; analysts downgrade to 'sell'",
    "Rising raw material costs squeeze {stock} margins by 300 bps",
    "Credit downgrade hits {stock} bonds; equity follows suit",
    "Disappointing outlook from {stock} management triggers sharp selloff amid {sector} weakness",
    "RBI inflation shock sends {stock} and {sector} into sharp decline",
    "{stock} misses revenue and profit estimates; stock drops {pct}",
    "Promoter pledging in {stock} spooks investors, stock down {pct}",
    "Foreign fund outflows batter {stock}; broader market under pressure",
    "{stock} hit by regulatory action; operations in key state suspended",
    "Nifty bears take control — {stock} among top losers, down {pct}",
    "Earnings season disappointment: {stock} profit falls 25% YoY",
    "High valuations, weak demand outlook weigh on {stock}",
    "{stock} audit qualifications raise accounting concerns; stock hammered",
    "India trade deficit widens to record high, hurting {sector} and {stock}",
    "Commodity price spike eats into {stock} margins; stock slides {pct}",
    "{stock} Q4 margin shock eclipses revenue growth; stock in freefall",
    "Global recession fears trigger risk-off; {stock} sheds {pct}",
    "{stock} loses market share in key segment, volumes decline 18%",
    "Management exit at {stock} raises governance concerns, stock falls {pct}",
    "{stock} debt levels rise sharply; leverage concerns emerge",
    "GST demand notice of ₹2,000 cr shocks {stock} investors",
    "China slowdown hits {stock} export revenues; stock dips {pct}",
    "{stock} rights issue at steep discount surprises street, stock drops",
    "Pollution shutdown order hits {stock} plant; production disrupted",
]

EN_NEUTRAL_TEMPLATES = [
    "{stock} shares trade flat ahead of {qtr} {fy} results expected next week",
    "{stock} volume dull as market awaits RBI policy decision",
    "Nifty 50 range-bound; {stock} consolidates near support levels",
    "{stock} analyst meet scheduled — no forward guidance change anticipated",
    "Mixed signals from {stock}: revenue beats but margins slightly miss",
    "{stock} trading at 52-week average; market digests {qtr} numbers",
    "FIIs marginally net sellers in {sector} with {stock} stable",
    "{stock} board meeting to approve quarterly accounts on Friday",
    "India markets open flat; {stock} moves in line with benchmark",
    "{stock} ex-dividend date tomorrow — no major price move expected",
    "Nifty 50 flat amid global uncertainty; {stock} holds key support",
    "{stock} to report results on Thursday — street estimates in line",
    "{stock} gets NCLT approval for subsidiary merger; impact neutral",
    "Budget allocations for {sector} broadly in line with expectations; {stock} unmoved",
    "RBI keeps repo rate unchanged at 6.5%; {stock} sees no reaction",
    "{stock} shares consolidate at current levels pending global cues",
    "{stock} AGM resolution on dividend to be taken next week",
    "Market participants await US Fed meeting before taking fresh positions in {stock}",
    "{stock} management reiterates full-year guidance unchanged",
    "{stock} pledges remain steady at current levels, no fresh concern",
    "Moderate FII buying in {sector}; {stock} stable near 200-day moving average",
    "{stock} holds its recent range; no fresh news ahead of results",
    "{stock} moving in line with {sector} index, no stock-specific trigger",
    "India macro data awaited this week; {stock} in wait and watch mode",
    "{stock} volume 20% below 10-day average; traders await direction",
    "Credit rating of {stock} affirmed at AA+; no change in outlook",
    "{stock} share buyback window opens today — limited market impact",
    "{sector} index in consolidation phase; {stock} near price equilibrium",
    "{stock} technical charts show neutral pattern; RSI at 50",
    "Q1 results season kicks off — {stock} expected to perform in-line", 
]


def fill_en(template, positive=None):
    stock  = random.choice(NIFTY_STOCKS_EN)
    sector = random.choice(SECTORS)
    pct    = random.choice(PCTS_UP if positive else PCTS_DOWN if positive is False else PCTS_UP)
    qtr    = random.choice(QUARTERS)
    fy     = random.choice(FY)
    # extra slots (only used by some templates, ignored otherwise)
    deal   = random.choice(DEAL_SIZES)
    pdrop  = random.choice(PROFIT_DROPS)
    prise  = random.choice(PROFIT_RISES)
    vol    = random.choice(VOLGROWTH)
    mbps   = random.choice(MARGINBPS)
    tref   = random.choice(TIMEREFS)
    month  = random.choice(MONTHS_EN)
    return template.format(
        stock=stock, sector=sector, pct=pct, qtr=qtr, fy=fy,
        deal=deal, pdrop=pdrop, prise=prise, vol=vol, mbps=mbps,
        tref=tref, month=month,
    )


def generate_english(target_per_label=TARGET_PER_LABEL):
    rows = []
    seen_texts: set[str] = set()
    labels = {
        "positive": (EN_POSITIVE_TEMPLATES, 1, True),
        "neutral":  (EN_NEUTRAL_TEMPLATES,  0, None),
        "negative": (EN_NEGATIVE_TEMPLATES, -1, False),
    }
    for label_name, (templates, label_id, positivity) in labels.items():
        # generate up to 3× the target and deduplicate down to target
        generated = 0
        attempts  = 0
        tmpl_cycle = itertools.cycle(templates)
        while generated < target_per_label and attempts < target_per_label * 6:
            tmpl = next(tmpl_cycle)
            text = fill_en(tmpl, positivity)
            attempts += 1
            if text in seen_texts:
                continue
            seen_texts.add(text)
            rows.append({
                "text": text,
                "label": label_id,
                "label_name": label_name,
                "lang": "en",
                "source_type": "generated",
                "seed_template_idx": int(attempts % len(templates)),
                "id": hashlib.md5(text.encode()).hexdigest()[:12],
            })
            generated += 1
    random.shuffle(rows)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# HINDI / HINGLISH DATASET
# ─────────────────────────────────────────────────────────────────────────────

# --- PURE HINDI positive templates ---
HI_POSITIVE_PURE = [
    "{stock} के शेयर {pct} उछले, मजबूत तिमाही नतीजों ने दिया बड़ा सहारा",
    "बाजार में तेजी: {stock} ने {pct} की छलांग लगाई, निवेशकों में खुशी",
    "{stock} का शुद्ध लाभ {qtr} में {pct} बढ़ा, विश्लेषक हुए उत्साहित",
    "{stock} ने {fy} में रिकॉर्ड राजस्व दर्ज किया, शेयर नई ऊंचाई पर",
    "एफआईआई ने {stock} में बड़ी खरीदारी की, स्टॉक {pct} चढ़ा",
    "{stock} को बड़ा सरकारी अनुबंध मिला, बाजार में जोरदार उत्साह",
    "आरबीआई की दर कटौती से {stock} और {sector} में बड़ी तेजी",
    "{stock} ने {sector} में सबसे ज्यादा रिटर्न दिया इस तिमाही",
    "मजबूत जीडीपी आंकड़ों से बाजार में जोश, {stock} {pct} ऊपर",
    "ब्रोकरेज हाउस ने {stock} को 'स्ट्रॉन्ग बाय' रेटिंग दी",
    "{stock} का निर्यात इतिहास के उच्चतम स्तर पर, शेयर में जोरदार तेजी",
    "निफ्टी 50 में {stock} आज सबसे बड़ा गेनर, {pct} की भारी बढ़त",
    "डीआईआई की मजबूत खरीदारी से {stock} में बुलिश सेंटिमेंट",
    "{stock} का डिविडेंड ऐलान, निवेशकों में खुशी की लहर",
    "विदेशी निवेशकों ने {stock} में रिकॉर्ड निवेश किया इस सप्ताह",
    "अप्रैल में {stock} की बिक्री {pct} उछली, बाजार का सेंटिमेंट बुलिश",
    "{stock} का पूंजी विस्तार योजना पर काम तेज, शेयर रिकॉर्ड ऊंचाई पर",
    "बाजार विश्लेषकों ने {stock} का लक्ष्य मूल्य बढ़ाया",
    "मुद्रास्फीति में गिरावट: {sector} में {stock} सबसे बड़ा फायदेमंद",
    "शेयर बाजार में रौनक: {stock} ने ऊपरी सर्किट लगाया",
]

# --- PURE HINDI negative templates ---
HI_NEGATIVE_PURE = [
    "{stock} के शेयर {pct} टूटे, कमजोर नतीजों ने बाजार को निराश किया",
    "{stock} का मुनाफा {pct} घटा, विश्लेषकों ने 'सेल' रेटिंग दी",
    "एफआईआई की बिकवाली से {stock} में भारी गिरावट, {pct} नुकसान",
    "{stock} पर सेबी की जांच, शेयर में निवेशकों की भाग-दौड़",
    "कमजोर मांग के संकेत से {stock} का शेयर {pct} लुढ़का",
    "{sector} में मंदी की आहट, {stock} सबसे बड़ा लूजर",
    "कच्चे माल की ऊंची लागत से {stock} का मार्जिन घटा, शेयर धड़ाम",
    "आरबीआई की सख्त नीति से {stock} पर दबाव, {pct} की गिरावट",
    "{stock} प्रमोटर की हिस्सेदारी गिरवी से निवेशक चिंतित",
    "{stock} का {qtr} नतीजा बाजार की उम्मीद से काफी कमजोर",
    "विदेशी बाजारों में मंदी से {stock} में भारी बिकवाली",
    "{stock} के खिलाफ कर विभाग की कार्रवाई, शेयर निचले स्तर पर",
    "ग्लोबल मंदी की आशंका से {sector} सेक्टर में {stock} पिछड़ा",
    "{stock} का कर्ज रिकॉर्ड स्तर पर, रेटिंग एजेंसी ने आउटलुक नेगेटिव किया",
    "{stock} की बाजार हिस्सेदारी में {pct} की गिरावट",
    "निफ्टी में बड़ी गिरावट: {stock} सबसे ज्यादा टूटने वाले शेयरों में",
    "सरकारी नीति में बदलाव से {sector} पर मार, {stock} {pct} नीचे",
    "{stock} का तिमाही राजस्व अनुमान से कम, शेयर धराशाई",
    "खराब मानसून अनुमान ने {stock} की संभावनाओं पर पानी फेरा",
    "{stock} लेखापरीक्षा में अनियमितता के संकेत, निवेशक हुए सतर्क",
]

# --- PURE HINDI neutral templates ---
HI_NEUTRAL_PURE = [
    "{stock} के शेयर सपाट, बाजार {qtr} नतीजों का इंतजार कर रहा है",
    "निफ्टी 50 स्थिर: {stock} में सीमित दायरे में कारोबार",
    "आरबीआई नीतिगत बैठक से पहले {stock} में सतर्कता",
    "{stock} का बोर्ड बैठक कल, कोई बड़ा बदलाव नहीं अपेक्षित",
    "{sector} सेक्टर में मिश्रित संकेत, {stock} स्थिर",
    "वैश्विक संकेतों की प्रतीक्षा में {stock} कारोबार सीमित",
    "{stock} के रिजल्ट की तारीख घोषित, विश्लेषक अनुमान के अनुरूप",
    "बाजार में उतार-चढ़ाव के बीच {stock} 200-दिन के औसत पर टिका",
    "{stock} का तकनीकी चार्ट तटस्थ संकेत दे रहा है",
    "{stock} में संस्थागत निवेशकों की हिस्सेदारी अपरिवर्तित",
    "क्रेडिट रेटिंग {stock} का एए प्लस पर बरकरार, कोई बदलाव नहीं",
    "{stock} का वॉल्यूम 10-दिन के औसत से कम, दिशा का इंतजार",
    "बजट पूर्व सत्र में {stock} और {sector} सीमित दायरे में",
    "{stock} का लाभांश रिकॉर्ड डेट जल्द, मूल्य प्रभाव सीमित",
    "मैनेजमेंट ने {stock} का वार्षिक मार्गदर्शन अपरिवर्तित रखा",
]

# --- HINGLISH positive templates (code-switched) ---
HINGLISH_POSITIVE = [
    "{stock} का share {pct} ऊपर गया, Q3 results ने सबको चौंकाया",
    "{stock} ने इस quarter में record profit book किया, investors खुश",
    "Nifty rally में {stock} आगे, FII ने heavy buying की",
    "{stock} shares ne 52-week high छुआ, strong earnings के बाद",
    "Bullish sentiment: {stock} में {pct} की तेजी, brokerage target बढ़ाई",
    "{stock} का dividend announce हुआ, market में positivity छाई",
    "RBI rate cut से {stock} और {sector} में जोरदार rally",
    "{stock} का export revenue all-time high, investors में जोश",
    "Management ने {stock} का guidance upgrade किया, stock flew {pct}",
    "DII net buyer रहे {stock} में, price support मजबूत",
    "{stock} ने {fy} {qtr} में analyst estimates को beat किया",
    "मजबूत macro data से {stock} और पूरे {sector} में तेजी",
    "{stock} buyback approved by SEBI, shares jumped {pct}",
    "Block deal: FII ने {stock} में ₹500 crore का stake खरीदा",
    "{stock} का revenue growth double-digit, stock outperforms Nifty",
    "India PMI surge ने {sector} boost किया, {stock} {pct} चढ़ा",
    "{stock} subsidiary का IPO 30x oversubscribed, parent भी चढ़ा",
    "Strong quarterly numbers: {stock} का EPS estimate से {pct} ज़ादा",
    "Credit rating upgrade मिली {stock} को, bond और equity दोनों ऊपर",
    "Global cues positive, {stock} में buying ka momentum strong है",
]

# --- HINGLISH negative templates ---
HINGLISH_NEGATIVE = [
    "{stock} का share {pct} गिरा, weak Q3 results से investors निराश",
    "FII selling से {stock} crash, {pct} की भारी गिरावट",
    "{stock} management ने lower guidance दी, street disappointed",
    "SEBI notice के बाद {stock} में panic selling, {pct} की गिरावट",
    "Weak global cues से {stock} और {sector} दोनों पर pressure",
    "{stock} का margin miss हुआ, analysts ने 'sell' call दी",
    "Rising input costs ने {stock} को नुकसान पहुंचाया, stock slips {pct}",
    "{stock} promoter pledge बढ़ी, investor confidence हिला",
    "Disappointing earnings: {stock} revenue miss + margin miss double whammy",
    "{stock} को regulatory action का सामना, stock tanked {pct}",
    "Global recession fears से {sector} में selloff, {stock} सबसे ज्यादा टूटा",
    "{stock} का debt level record high, rating agency ने outlook negative किया",
    "Audit concerns raised for {stock}, investors को घबराहट",
    "Market मंदी में {stock} ने {pct} का नुकसान दिया इस week",
    "{stock} market share {pct} घटी, competition बढ़ा, stock दबाव में",
    "Tax demand notice: {stock} को ₹1000 crore का GST notice मिला",
    "China slowdown का असर {stock} के exports पर, share price slips",
    "RBI की hawkish policy से {sector} hit, {stock} biggest loser",
    "{stock} ने rights issue announce किया steep discount पर, stock गिरा",
    "Poor monsoon forecast ने {stock} की earnings outlook को hurt किया",
]

# --- HINGLISH neutral templates ---
HINGLISH_NEUTRAL = [
    "{stock} shares flat हैं, market {qtr} results का wait कर रहा है",
    "Nifty में no clear direction — {stock} range bound movement",
    "RBI policy meeting से पहले {sector} में caution, {stock} stable",
    "{stock} board meeting होगी Friday को, कोई major announcement नहीं",
    "Mixed signals: {stock} का revenue beat हुआ लेकिन margins थोड़े miss",
    "{stock} 200-day moving average पर support ले रहा है",
    "Market में sideways trend, {stock} में कोई fresh trigger नहीं",
    "FII ने {stock} में neutral stance लिया है, minor buying-selling",
    "{stock} का {fy} annual guidance unchanged रहा",
    "Technical charts पर {stock} neutral zone में है, RSI 50 के पास",
    "Budget expectations से {sector} flat, {stock} कोई बड़ी move नहीं",
    "Analysts का {stock} पर 'hold' rating, कोई revision नहीं",
    "Global markets uncertain — {stock} in wait-and-watch mode",
    "{stock} का trade volume below average, direction clear नहीं",
    "Credit rating affirmed: {stock} AA+ पर stable, कोई बदलाव नहीं",
]


def fill_hi(template, positivity=None):
    stock  = random.choice(NIFTY_STOCKS_HI)
    sector = random.choice(SECTORS_HI)
    pct    = random.choice(PCTS_UP if positivity else PCTS_DOWN if positivity is False else PCTS_UP)
    qtr    = random.choice(QUARTERS)
    fy     = random.choice(FY)
    deal   = random.choice(DEAL_SIZES)
    pdrop  = random.choice(PROFIT_DROPS)
    prise  = random.choice(PROFIT_RISES)
    month  = random.choice(MONTHS_HI)
    return template.format(
        stock=stock, sector=sector, pct=pct, qtr=qtr, fy=fy,
        deal=deal, pdrop=pdrop, prise=prise, month=month,
    )


def fill_hinglish(template, positivity=None):
    stock  = random.choice(NIFTY_STOCKS_EN)  # use EN ticker in Hinglish
    sector = random.choice(SECTORS)
    pct    = random.choice(PCTS_UP if positivity else PCTS_DOWN if positivity is False else PCTS_UP)
    qtr    = random.choice(QUARTERS)
    fy     = random.choice(FY)
    deal   = random.choice(DEAL_SIZES)
    pdrop  = random.choice(PROFIT_DROPS)
    prise  = random.choice(PROFIT_RISES)
    tref   = random.choice(TIMEREFS)
    return template.format(
        stock=stock, sector=sector, pct=pct, qtr=qtr, fy=fy,
        deal=deal, pdrop=pdrop, prise=prise, tref=tref,
    )


def generate_hindi_hinglish(target_per_label=TARGET_PER_LABEL):
    rows = []
    seen_texts: set[str] = set()

    def _gen_dedup(templates, fill_fn, label_id, label_name, lang, positivity, count):
        generated = 0
        attempts  = 0
        cycle_t   = itertools.cycle(templates)
        while generated < count and attempts < count * 6:
            tmpl = next(cycle_t)
            text = fill_fn(tmpl, positivity)
            attempts += 1
            if text in seen_texts:
                continue
            seen_texts.add(text)
            rows.append({
                "text": text, "label": label_id, "label_name": label_name,
                "lang": lang, "source_type": "generated",
                "seed_template_idx": int(attempts % len(templates)),
                "id": hashlib.md5(text.encode()).hexdigest()[:12],
            })
            generated += 1

    half = target_per_label // 2
    for label_name, pos_pure, neg_pure, neu_pure, pos_mix, neg_mix, neu_mix, label_id, positivity in [
        ("positive", HI_POSITIVE_PURE, None, None, HINGLISH_POSITIVE, None, None, 1, True),
        ("neutral",  None, None, HI_NEUTRAL_PURE, None, None, HINGLISH_NEUTRAL, 0, None),
        ("negative", None, HI_NEGATIVE_PURE, None, None, HINGLISH_NEGATIVE, None, -1, False),
    ]:
        # pick the relevant template pool
        pure_tmpl = pos_pure or neg_pure or neu_pure
        mix_tmpl  = pos_mix  or neg_mix  or neu_mix
        _gen_dedup(pure_tmpl, fill_hi,       label_id, label_name, "hi",       positivity, half)
        _gen_dedup(mix_tmpl,  fill_hinglish, label_id, label_name, "hinglish", positivity, half)

    random.shuffle(rows)
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# WRITER
# ─────────────────────────────────────────────────────────────────────────────
FIELDNAMES = ["id", "text", "label", "label_name", "lang", "source_type", "seed_template_idx"]


def write_csv(path: pathlib.Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    counts = {}
    for r in rows:
        counts[r["label_name"]] = counts.get(r["label_name"], 0) + 1
    print(f"  Wrote {len(rows):,} rows → {path}")
    for k, v in sorted(counts.items()):
        print(f"    {k:10s}: {v:,}")


if __name__ == "__main__":
    print("Generating datasets …\n")

    en_path = OUT_DIR / "english_financial.csv"
    hi_path = OUT_DIR / "hindi_hinglish_financial.csv"

    if en_path.exists():
        print(f"[SKIP] {en_path} already exists. Pass --force to regenerate.")
    else:
        en_rows = generate_english()
        write_csv(en_path, en_rows)

    if hi_path.exists():
        print(f"[SKIP] {hi_path} already exists. Pass --force to regenerate.")
    else:
        hi_rows = generate_hindi_hinglish()
        write_csv(hi_path, hi_rows)

    import sys
    if "--force" in sys.argv:
        print("\nForce mode: regenerating both …")
        write_csv(en_path, generate_english())
        write_csv(hi_path, generate_hindi_hinglish())

    print("\nDone.")
