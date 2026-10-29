"""Fixed eval for quick-text 'Other' decisions (regression test + tuning). 3 labels x 6 examples, 20 on-topic
paraphrases (Hinglish / Hindi / English) and 20 off-topic everyday texts that are NOT in the background set."""

EXAMPLES = {
    "refund": ["mujhe refund chahiye", "paise wapas karo", "I want my money back", "refund kab milega",
               "paisa wapas kab milega", "cancel karke refund do"],
    "delivery late": ["order abhi tak nahi aaya", "delivery late hai", "my parcel is delayed", "kab tak deliver hoga",
                      "courier nahi aaya", "parcel kab aayega"],
    "praise": ["bahut accha service", "great job team", "thank you, very helpful", "aap log best ho",
               "mast experience tha", "love your app"],
}
ON_TOPIC = [  # (text, expected label)
    ("bhai mera paisa kab wapas aayega", "refund"), ("refund please, paisa wapas chahiye", "refund"),
    ("mera refund abhi tak nahi mila", "refund"), ("money back kab aayega", "refund"),
    ("paisa return karo jaldi", "refund"), ("मुझे रिफंड चाहिए", "refund"), ("please refund my payment", "refund"),
    ("mera order 5 din se nahi pahucha", "delivery late"), ("parcel abhi tak deliver nahi hua", "delivery late"),
    ("where is my package, it's been a week", "delivery late"), ("mera order kab aayega bhai", "delivery late"),
    ("मेरा ऑर्डर अभी तक नहीं आया", "delivery late"), ("delivery bahut slow hai", "delivery late"),
    ("courier wala aaya hi nahi", "delivery late"),
    ("aapki team bahut acchi hai", "praise"), ("superb service, thank you", "praise"), ("bahut badhiya kaam kiya", "praise"),
    ("really happy with your support", "praise"), ("आपकी सेवा बहुत अच्छी है", "praise"), ("you guys are awesome", "praise"),
]
OFF_TOPIC = [
    "aaj mausam bahut accha hai", "mujhe cricket dekhna pasand hai", "kal office jaana hai",
    "mera bhai college mein padhta hai", "dinner mein kya banaya", "mobile ka cover chahiye", "kya haal hai bhai",
    "this song is really good", "what is the capital of France", "मुझे किताबें पढ़ना पसंद है", "train kitne baje hai",
    "mera phone charge nahi ho raha", "weekend pe movie chalein", "how do I cook pasta", "baby so gaya hai",
    "petrol ka rate badh gaya", "kal exam hai mera", "मेरी बहन की शादी है", "gym jaana start kiya", "is it going to snow",
]
