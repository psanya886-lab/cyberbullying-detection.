import pickle
import gradio as gr

with open(r"C:\Users\sanya\model.pkl", "rb") as f:
    model = pickle.load(f)

with open(r"C:\Users\sanya\tfidfvectorizer.pkl", "rb") as f:
    tfidfvector = pickle.load(f)

def predict(text):
    text_vector = tfidfvector.transform([text])
    prediction = model.predict(text_vector)[0]

    if prediction == -1:
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

demo.launch(server_name="0.0.0.0", server_port=7860)
