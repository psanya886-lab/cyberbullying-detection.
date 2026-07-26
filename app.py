import pickle
import gradio as gr
import re

# Load model and TF-IDF vectorizer
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("tfidfvectorizer.pkl", "rb") as f:
    tfidfvector = pickle.load(f)


# Known abusive words
toxic_words = {
    "ass", "arse", "asshole", "bastard", "bitch",
    "bollocks", "bullshit", "cock", "crap", "cunt",
    "dick", "dildo", "douche", "douchebag",
    "dumbass", "fag", "faggot", "fuck", "fucking",
    "fugly", "goddamn", "idiot", "jackass", "jerk",
    "kike", "lodu", "madarchod", "moron",
    "nigger", "pedo", "penis", "piss", "prick",
    "pussy", "rape", "retard", "scum", "scumbag",
    "shit", "slut", "stupid", "suck", "tard",
    "tits", "twat", "wanker", "whore",
    "chink", "chutiya", "chutiyapa", "chutiye",
    "bhenchod", "bhosda", "bsdk", "gaand",
    "gand", "laude", "lavde", "randi"
}


def predict(text):
    text_lower = text.lower()

    # Direct abusive-word detection
    words = set(re.findall(r"[a-zA-Z]+", text_lower))

    if words.intersection(toxic_words):
        return "Toxic"

    # ML model for other sentences
    text_vector = tfidfvector.transform([text])
    score = model.decision_function(text_vector)[0]

    # Higher threshold reduces false positives
    if score > 1.0:
        return "Toxic"
    else:
        return "Non-Toxic"


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        lines=5,
        placeholder="Enter a comment..."
    ),
    outputs=gr.Textbox(label="Prediction"),
    title="Toxic Comment Detection",
    description="Enter a comment to detect whether it is toxic or non-toxic."
)

demo.launch()