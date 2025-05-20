from flask import Flask, render_template, request, redirect, url_for
from flask_pymongo import PyMongo
from datetime import datetime
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
import json
from bson.objectid import ObjectId
import logging
from PIL import Image

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE = 8 * 1024 * 1024  # 8MB
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# MongoDB Configuration
app.config["MONGO_URI"] = "mongodb://localhost:27017/hydroponics_db"
mongo = PyMongo(app)

# Complete Treatment Database (ensure these match your model's class names)
COMPLETE_TREATMENTS = {
    "Apple___Apple_scab": {
        "name": "Apple Scab",
        "treatment": """1. Apply sulfur sprays every 7-10 days\n2. Remove infected leaves\n3. Adjust pH to 6.0-6.5""",
        "prevention": """1. Plant resistant varieties\n2. Disinfect tools\n3. Avoid overhead watering""",
        "severity": "Moderate"
    },
    "Apple___Black_rot": {
        "name": "Apple Black Rot",
        "treatment": """1. Prune infected branches\n2. Apply copper fungicide\n3. Remove mummified fruits""",
        "prevention": """1. Use drip irrigation\n2. Ensure good air circulation""",
        "severity": "High"
    },
    "Tomato___Early_blight": {
        "name": "Tomato Early Blight",
        "treatment": """1. Apply chlorothalonil\n2. Remove lower leaves\n3. Increase magnesium""",
        "prevention": """1. Sterilize system\n2. Maintain pH 5.8-6.2\n3. Avoid leaf wetness""",
        "severity": "Moderate"
    },
    "Tomato___Late_blight": {
        "name": "Tomato Late Blight",
        "treatment": """1. Quarantine plants\n2. Apply phosphorous acid\n3. Increase air temp""",
        "prevention": """1. Use closed systems\n2. Install UV sterilization\n3. Select resistant cultivars""",
        "severity": "High"
    },
    "unknown": {
        "name": "Unknown Disease",
        "treatment": "Consult a plant specialist for diagnosis",
        "prevention": "Maintain good plant hygiene",
        "severity": "Unknown"
    },
    "error": {
        "name": "Diagnosis Error",
        "treatment": "Please try again with a clearer photo",
        "prevention": "Ensure good lighting and focus",
        "severity": "N/A"
    }
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_image(file_stream):
    try:
        img = Image.open(file_stream)
        img.verify()
        file_stream.seek(0)
        return True
    except Exception as e:
        logger.error(f"Invalid image file: {e}")
        return False

def preprocess_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(256, 256))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        return None

def load_plant_model():
    """Load model from SavedModel or .keras format"""
    try:
        # Try SavedModel format (directory)
        if os.path.exists('models/plantdoc_model'):
            model = tf.keras.models.load_model('models/plantdoc_model')
            logger.info("Model loaded successfully from SavedModel format")
            return model
        
        # Try .keras format
        if os.path.exists('models/plantdoc_model.keras'):
            model = tf.keras.models.load_model('models/plantdoc_model.keras')
            logger.info("Model loaded successfully from .keras format")
            return model
        
        logger.error("No model files found in models/ directory")
        return None
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        return None

# Load model and class names
model = load_plant_model()

try:
    with open('models/class_indices.json') as f:
        class_indices = json.load(f)
    CLASS_NAMES = list(class_indices.keys())
    logger.info(f"Loaded {len(CLASS_NAMES)} class names")
except Exception as e:
    logger.error(f"Error loading class indices: {e}")
    CLASS_NAMES = list(COMPLETE_TREATMENTS.keys())
    CLASS_NAMES.remove('unknown')
    CLASS_NAMES.remove('error')

def predict_disease(img_path):
    """Returns list of all possible predictions with confidence scores"""
    if model is None:
        # Return all diseases with equal probability when model fails
        all_results = []
        for disease_key, disease_info in COMPLETE_TREATMENTS.items():
            if disease_key not in ["unknown", "error"]:
                all_results.append((
                    disease_info,
                    1.0/len(COMPLETE_TREATMENTS),  # Equal probability
                    disease_key
                ))
        return all_results
    
    try:
        img_array = preprocess_image(img_path)
        if img_array is None:
            # Return all diseases with equal probability when image processing fails
            all_results = []
            for disease_key, disease_info in COMPLETE_TREATMENTS.items():
                if disease_key not in ["unknown", "error"]:
                    all_results.append((
                        disease_info,
                        1.0/len(COMPLETE_TREATMENTS),  # Equal probability
                        disease_key
                    ))
            return all_results
        
        pred = model.predict(img_array)[0]  # Get predictions for the image
        
        # Get all predictions with confidence scores
        all_results = []
        for i, confidence in enumerate(pred):
            if i < len(CLASS_NAMES):
                disease_key = CLASS_NAMES[i]
                disease_info = COMPLETE_TREATMENTS.get(disease_key, COMPLETE_TREATMENTS["unknown"])
                all_results.append((
                    disease_info,
                    float(confidence),
                    disease_key
                ))
        
        # Sort by confidence (highest first)
        all_results.sort(key=lambda x: x[1], reverse=True)
        
        return all_results
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        # Return all diseases with equal probability when prediction fails
        all_results = []
        for disease_key, disease_info in COMPLETE_TREATMENTS.items():
            if disease_key not in ["unknown", "error"]:
                all_results.append((
                    disease_info,
                    1.0/len(COMPLETE_TREATMENTS),  # Equal probability
                    disease_key
                ))
        return all_results

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('error.html', message="No file uploaded")
            
        file = request.files['file']
        if file.filename == '':
            return render_template('error.html', message="No file selected")
            
        if not allowed_file(file.filename):
            return render_template('error.html', message="Invalid file type")
            
        if not validate_image(file.stream):
            return render_template('error.html', message="Invalid image file")
            
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            predictions = predict_disease(filepath)
            top_prediction = predictions[0]  # Get the highest confidence prediction
            disease_info, confidence, disease_key = top_prediction
            
            # Store all predictions in database
            diagnosis = {
                "image_path": filepath,
                "disease": disease_info["name"],
                "scientific_name": disease_key,
                "confidence": confidence,
                "all_predictions": [
                    {
                        "disease": p[0]["name"],
                        "confidence": p[1],
                        "scientific_name": p[2]
                    } for p in predictions
                ],
                "treatment": disease_info["treatment"],
                "prevention": disease_info["prevention"],
                "severity": disease_info.get("severity", "Unknown"),
                "timestamp": datetime.now()
            }
            mongo.db.diagnoses.insert_one(diagnosis)
            
            return render_template('result.html',
                                disease_name=disease_info["name"],
                                scientific_name=disease_key,
                                confidence=round(confidence*100, 2),
                                all_predictions=predictions[:5],  # Show top 5 predictions
                                treatment=disease_info["treatment"],
                                prevention=disease_info["prevention"],
                                severity=disease_info.get("severity", "Unknown"),
                                image_path=filepath)
            
        except Exception as e:
            logger.error(f"Upload error: {e}")
            return render_template('error.html', message="Processing error")
    
    return render_template('index.html')

@app.route('/history')
def history():
    try:
        diagnoses = list(mongo.db.diagnoses.find().sort("timestamp", -1).limit(20))
        # Convert ObjectId to string for template rendering
        for diagnosis in diagnoses:
            diagnosis['_id'] = str(diagnosis['_id'])
        return render_template('history.html', diagnoses=diagnoses)
    except Exception as e:
        logger.error(f"History error: {e}")
        return render_template('error.html', message="Could not retrieve history")

@app.route('/diagnosis/<id>')
def view_diagnosis(id):
    try:
        diagnosis = mongo.db.diagnoses.find_one({"_id": ObjectId(id)})
        if not diagnosis:
            return render_template('error.html', message="Diagnosis record not found")
        return render_template('view_diagnosis.html', diagnosis=diagnosis)
    except Exception as e:
        logger.error(f"View diagnosis error: {e}")
        return render_template('error.html', message="Invalid diagnosis ID")

@app.route('/delete/<id>', methods=['POST'])
def delete_diagnosis(id):
    try:
        # Remove the file from uploads
        diagnosis = mongo.db.diagnoses.find_one({"_id": ObjectId(id)})
        if diagnosis and os.path.exists(diagnosis['image_path']):
            os.remove(diagnosis['image_path'])
        
        # Remove from database
        mongo.db.diagnoses.delete_one({"_id": ObjectId(id)})
        return redirect(url_for('history'))
    except Exception as e:
        logger.error(f"Deletion error: {e}")
        return render_template('error.html', message=f"Deletion failed: {str(e)}")

@app.errorhandler(413)
def request_entity_too_large(error):
    return render_template('error.html', message="File too large (max 8MB)"), 413

@app.errorhandler(404)
def page_not_found(error):
    return render_template('error.html', message="Page not found"), 404

if __name__ == '__main__':
    if model is None:
        logger.error("Could not load model - running with limited functionality")
    app.run(debug=True)