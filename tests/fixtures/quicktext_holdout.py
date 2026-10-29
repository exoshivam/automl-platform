"""Held-out eval for quick text labels (independent of quicktext_eval.py).

Same three labels as quicktext_eval.py: refund / delivery late / praise. Mixed Hinglish (roman script), Hindi
(Devanagari) and English. ON_TOPIC = (text, expected_label); OFF_TOPIC texts should come back as "Other".
Written fresh, not paraphrased from quicktext_eval.py (whose sentences were not read).
"""

LABELS = ["refund", "delivery late", "praise"]

ON_TOPIC: list[tuple[str, str]] = [
    # refund
    ("galat size aaya tha, return kar diya, ab tak paise account mein nahi aaye", "refund"),
    ("cancel karne ke baad bhi amount katt gaya, wapas kab milega", "refund"),
    ("मेरा रिफंड दस दिन से अटका हुआ है", "refund"),
    ("I returned the shoes last week and still haven't got my money back", "refund"),
    ("UPI se pay kiya tha, refund ka status pending dikha raha hai", "refund"),
    ("double payment ho gaya, ek wala lauta do please", "refund"),
    ("please process the refund for order 4471, it was damaged", "refund"),
    # delivery late
    ("expected date kal thi, aaj tak courier wala nahi aaya", "delivery late"),
    ("order dispatch hua hi nahi abhi tak, kitna time lagega", "delivery late"),
    ("पार्सल अभी तक नहीं पहुँचा, दो दिन लेट हो गया", "delivery late"),
    ("my package has been stuck at the hub for four days", "delivery late"),
    ("shipment 'out for delivery' dikha raha tha phir bhi nahi aaya", "delivery late"),
    ("bhai birthday gift tha, time pe nahi pahuncha", "delivery late"),
    ("where is my order? it was supposed to arrive on Monday", "delivery late"),
    # praise
    ("delivery boy bahut polite tha, khush kar diya", "praise"),
    ("product photo jaisa hi nikla, full paisa vasool", "praise"),
    ("आपकी सर्विस से मैं बहुत संतुष्ट हूँ", "praise"),
    ("great experience, support team solved it in minutes", "praise"),
    ("dusri baar order kiya, phir se ekdum perfect", "praise"),
    ("really impressed with how fresh the vegetables were", "praise"),
]

OFF_TOPIC: list[str] = [
    "kal raat match dekha? last over mein kya scene tha",
    "mummy ne aaj rajma chawal banaye hain",
    "office ki meeting teen baje shift ho gayi",
    "बारिश की वजह से आज स्कूल बंद है",
    "can you recommend a good book on history",
    "mera phone ka charger kahin kho gaya",
    "weekend pe Goa jaane ka plan hai",
    "the train to Pune leaves at 6:40 tomorrow",
    "gym jaana shuru kiya hai, legs dukh rahe hain",
    "आज बहुत ठंड है, रजाई से निकलने का मन नहीं",
    "what's the capital of Australia",
    "bhai exam ki date aa gayi kya",
    "neighbour ka kutta raat bhar bhonkta raha",
    "I'm learning to play the guitar",
    "chai mein cheeni kam daalna",
    "sunday ko cousin ki shaadi hai",
    "मेरा लैपटॉप बहुत धीमा चल रहा है",
    "the sunset at the beach was beautiful",
    "kal se Hindi movie ki shooting shuru hogi",
    "plants ko paani dena bhool gaya",
]

assert len(ON_TOPIC) == 20 and len(OFF_TOPIC) == 20
assert {lbl for _, lbl in ON_TOPIC} == set(LABELS)
