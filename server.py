from flask import Flask, request, jsonify, render_template
import spacy

app = Flask(__name__)

#load fine tuned model
nlp = spacy.load("output/fine_tuned_ner_model")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    doc = nlp(text)
    results = [{"text": ent.text, "start": ent.start_char, "end": ent.end_char, "label": ent.label_} for ent in doc.ents]
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
