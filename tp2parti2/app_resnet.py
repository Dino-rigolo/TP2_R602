from flask import Flask, render_template, request
from flask_uploads import UploadSet, configure_uploads, IMAGES
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import os

app = Flask(__name__)


photos = UploadSet('photos', IMAGES)
app.config['UPLOADED_PHOTOS_DEST'] = './static/img'
configure_uploads(app, photos)

model = ResNet50(weights='imagenet')

@app.route('/')
def home():
    return render_template('upload.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'photo' not in request.files:
            return render_template('upload.html', erreur="Veuillez sélectionner une image")
        
 
        filename = photos.save(request.files['photo'])
        filepath = os.path.join(app.config['UPLOADED_PHOTOS_DEST'], filename)
        image = load_img(filepath, target_size=(224, 224))
        image_array = img_to_array(image)
        image_array = image_array.reshape((1, image_array.shape[0], image_array.shape[1], image_array.shape[2]))
        image_array = preprocess_input(image_array)
        prediction = model.predict(image_array, verbose=0)
        label = decode_predictions(prediction)
        
        # Prédiction principale
        main_prediction = label[0][0]
        main_class = main_prediction[1]
        main_confidence = main_prediction[2] * 100
        
        image_url = f'/static/img/{filename}'
        
        return render_template('upload.html', 
                             image_url=image_url,
                             classe=main_class, 
                             confiance=f"{main_confidence:.2f}")
    
    except Exception as e:
        return render_template('upload.html', erreur=f"Erreur lors du traitement : {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
