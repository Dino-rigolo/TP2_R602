from flask import Flask, render_template, request, jsonify
import pickle

app = Flask(__name__)

# Charger les modèles pré-entraînés
cv = pickle.load(open("models/cv.pkl", 'rb')) 
clf = pickle.load(open("models/clf.pkl", 'rb'))

@app.route('/')
def home():
    return render_template('spam_classifier.html')

@app.route('/predict', methods=['POST'])
def predict():
    email = request.form.get('email', '')
    
    # Transformer l'email en vecteur de tokens
    tokenized_email = cv.transform([email])
    
    # Faire la prédiction
    prediction = clf.predict(tokenized_email)[0]
    
    # Interpréter la prédiction
    if prediction == 1:
        result = "SPAM"
    else:
        result = "NON SPAM"
    
    return render_template('spam_classifier.html', 
                         email=email,
                         result=result,
                         prediction_made=True)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json(force=True)
    email = data.get('email', '')
    
    # Transformer l'email en vecteur de tokens
    tokenized_email = cv.transform([email])
    
    # Faire la prédiction
    prediction = clf.predict(tokenized_email)[0]
    
    # Interpréter la prédiction
    if prediction == 1:
        result = "SPAM"
    else:
        result = "NON SPAM"
    
    return jsonify({
        'success': True,
        'email': email,
        'prediction': result,
        'prediction_value': int(prediction)
    })

if __name__ == '__main__':
    app.run(debug=True)
