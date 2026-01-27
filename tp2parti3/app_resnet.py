from flask import Flask, render_template, request
from flask_uploads import UploadSet, configure_uploads, IMAGES
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import os


#image=keras.utils.load_img('./static/img'+filename, target_size=(224, 224) )
#image=preprocess_input(image)
#prediction=model.predict(image)
#label=decode_predictions(prediction)

app = Flask(__name__)

# Configuration des uploads d'images
photos = UploadSet('photos', IMAGES)
app.config['UPLOADED_PHOTOS_DEST'] = './static/img'
configure_uploads(app, photos)

# Charger le modèle ResNet50 pré-entraîné
model = ResNet50(weights='imagenet')

@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Vérifier qu'une image a été uploadée
        if 'photo' not in request.files:
            return render_template('upload.html', erreur="Veuillez sélectionner une image")
        
        # Sauvegarder l'image
        filename = photos.save(request.files['photo'])
        filepath = os.path.join(app.config['UPLOADED_PHOTOS_DEST'], filename)
        
        # Charger et pré-traiter l'image
        # ResNet50 attend des images de taille 224x224
        image = load_img(filepath, target_size=(224, 224))
        image_array = img_to_array(image)
        image_array = image_array.reshape((1, image_array.shape[0], image_array.shape[1], image_array.shape[2]))
        image_array = preprocess_input(image_array)
        
        # Faire la prédiction
        prediction = model.predict(image_array, verbose=0)
        
        # Décoder la prédiction (convertir du one-hot encoding)
        label = decode_predictions(prediction)
        
        # Extraire les 3 meilleures prédictions
        top_predictions = label[0][:3]
        
        # Formater les résultats
        results = []
        for pred in top_predictions:
            class_name = pred[1]
            confidence = pred[2] * 100
            results.append({
                'name': class_name,
                'confidence': confidence
            })
        
        # Prédiction principale
        main_class = top_predictions[0][1]
        main_confidence = top_predictions[0][2] * 100
        
        # Construire le chemin relatif pour l'affichage de l'image
        image_url = f'/static/img/{filename}'
        
        return render_template('upload.html', 
                             image_url=image_url,
                             classe=main_class, 
                             confiance=f"{main_confidence:.2f}",
                             predictions=results)
    
    except Exception as e:
        return render_template('upload.html', erreur=f"Erreur lors du traitement : {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
