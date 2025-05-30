from django.shortcuts import render
from django.conf import settings
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_input_RN
import os
import numpy as np
import cv2

# Define paths to the models
model_CM_path = os.path.join(settings.MEDIA_ROOT, 'CM_weights-010-0.3063.hdf5')
model_RN_path = os.path.join(settings.MEDIA_ROOT, 'RN_weights-009-0.3958.hdf5')

# Load models function
def load_models():
    try:
        model_CM = load_model(model_CM_path)
        print("CM model loaded successfully.")
    except Exception as e:
        print(f"Error loading CM model: {e}")
        model_CM = None

    try:
        model_RN = load_model(model_RN_path)
        print("RN model loaded successfully.")
    except Exception as e:
        print(f"Error loading RN model: {e}")
        model_RN = None

    return model_CM, model_RN

# Process image and predict function
def process_image_and_predict(image_path, model, model_type):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Image not found or could not be read.")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (48, 48))
        image = img_to_array(image)

        if model_type == 'CM':
            image /= 255.0
        elif model_type == 'RN':
            image = preprocess_input_RN(image)
        
        image = np.expand_dims(image, axis=0)

        # Make prediction
        predictions = model.predict(image)[0]
        print(f"Predictions: {predictions}")
        return predictions
    except Exception as e:
        print(f"Error predicting with model: {e}")
        return None

# Test view
def index(request):
    if request.method=='POST':
        print()
    else:
        return render(request,'index.html')

def test(request):
    context = {}
    
    if request.method == 'POST' and 'file' in request.FILES:
        uploaded_file = request.FILES['file']
        
        # Save uploaded file
        file_path = os.path.join(settings.MEDIA_ROOT, uploaded_file.name)
        with open(file_path, 'wb+') as destination:
            for chunk in uploaded_file.chunks():
                destination.write(chunk)

        imagePath = file_path
        
        # Load models
        model_CM, model_RN = load_models()
        
        if model_CM and model_RN:
            # Process image and make predictions
            predictions_CM = process_image_and_predict(imagePath, model_CM, 'CM')
            predictions_RN = process_image_and_predict(imagePath, model_RN, 'RN')
            
            if predictions_CM is not None and predictions_RN is not None:
                benign_CM, malignant_CM = predictions_CM
                benign_RN, malignant_RN = predictions_RN

                label_CM = "{}: {:.2f}%".format("benign" if benign_CM > malignant_CM else "malignant", max(benign_CM, malignant_CM) * 100)
                label_RN = "{}: {:.2f}%".format("benign" if benign_RN > malignant_RN else "malignant", max(benign_RN, malignant_RN) * 100)

                context['label_CM'] = label_CM
                context['label_RN'] = label_RN
                context['uploaded_file_url'] = settings.MEDIA_URL + uploaded_file.name
            else:
                context['error'] = "Error making predictions."
        else:
            context['error'] = "Error loading models."
    
    return render(request, 'test.html', context)

# Other views
def index(request):
    return render(request, 'index.html')

def services(request):
    return render(request, 'services.html')

def contact_us(request):
    return render(request, 'contact_us.html')

def blog(request):
    return render(request, 'blog.html')

def about_us(request):
    return render(request, 'about_us.html')
