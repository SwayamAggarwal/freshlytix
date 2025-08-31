from flask import Flask, render_template, request, redirect, url_for, session
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
import os
import numpy as np

app = Flask(__name__)
app.secret_key = 'your_secret_key' 


MODEL_PATH = "model/vgg16_model.pth" 
model_state = torch.load(MODEL_PATH, map_location=torch.device("cpu"))


state_dict = model_state["state_dict"]
nutrition_data = model_state["nutrition_data"]
class_names = model_state["class_names"]


model = models.vgg16(pretrained=False)
model.classifier[6] = nn.Linear(model.classifier[6].in_features, len(class_names))
model.load_state_dict(state_dict)
model.eval()  


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


transformer = transforms.Compose([
    transforms.Resize((128, 128)),  
    transforms.ToTensor()
])

# Define upload folder
UPLOAD_FOLDER = "static/uploads/"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to clean class label
def clean_class_label(class_label):
    """ Extracts fruit/vegetable name (e.g., 'Carrot__Rotten' -> 'Carrot'). """
    return class_label.split("__")[0].strip()


def predict_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = transformer(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        _, predicted = torch.max(output, 1)

    predicted_class = class_names[predicted.item()]
    fruit_or_veg_name = clean_class_label(predicted_class)
    is_healthy = "Fresh" if "Healthy" in predicted_class else "Rotten"

    if is_healthy == "Rotten":
        return fruit_or_veg_name, is_healthy, None

    nutrition = pd.DataFrame(nutrition_data)
    nutrition = nutrition[nutrition["name"].str.contains(fruit_or_veg_name, case=False, na=False)]
    
    if not nutrition.empty:
        nutrition = nutrition.iloc[0]
        nutrition_values = {
            "Energy (kJ)": float(nutrition["energy (kJ)"]),
            "Water (g)": float(nutrition["water (g)"]),
            "Protein (g)": float(nutrition["protein (g)"]),
            "Total Fat (g)": float(nutrition["total fat (g)"]),
            "Carbohydrates (g)": float(nutrition["carbohydrates (g)"]),
            "Fiber (g)": float(nutrition["fiber (g)"]),
            "Sugars (g)": float(nutrition["sugars (g)"]),
            "Calcium (mg)": float(nutrition["calcium (mg)"]),
            "Iron (mg)": float(nutrition["iron (mg)"]),
            "Magnesium (mg)": float(nutrition["magnessium (mg)"]),
            "Phosphorus (mg)": float(nutrition["phosphorus (mg)"]),
            "Potassium (mg)": float(nutrition["potassium (mg)"]),
            "Sodium (g)": float(nutrition["sodium (g)"]),
            "Vitamin A (IU)": float(nutrition["vitamin A (IU)"]),
            "Vitamin C (mg)": float(nutrition["vitamin C (mg)"]),
            "Vitamin B1 (mg)": float(nutrition["vitamin B1 (mg)"]),
            "Vitamin B2 (mg)": float(nutrition["vitamin B2 (mg)"]),
            "Vitamin B3 (mg)": float(nutrition["vitamin B3 (mg)"]),
            "Vitamin B5 (mg)": float(nutrition["vitamin B5 (mg)"]),
            "Vitamin B6 (mg)": float(nutrition["vitamin B6 (mg)"]),
            "Vitamin E (mg)": float(nutrition["vitamin E (mg)"])
        }
        # 3 VALUE DECIMAL
        nutrition_values = {k: round(v, 3) for k, v in nutrition_values.items()}
    else:
        nutrition_values = None  # No nutrition data found

    return fruit_or_veg_name, is_healthy, nutrition_values

# Routes
@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)

        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)

        if file:
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(file_path)
            fruit_or_veg_name, result, nutrition = predict_image(file_path)
            session['filename'] = file.filename
            session['name'] = fruit_or_veg_name
            session['result'] = result
            session['nutrition'] = nutrition
            return render_template("index.html", filename=file.filename, name=fruit_or_veg_name, result=result, nutrition=nutrition)

    return render_template("index.html", filename=None, name=None, result=None, nutrition=None)

@app.route('/')
def index():
    
    if 'filename' in session:
        session.pop('filename', None)
        session.pop('result', None)
        session.pop('name', None)
        session.pop('nutrition', None)
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
