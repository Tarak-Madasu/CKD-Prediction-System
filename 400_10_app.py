from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
# Load the trained Random Forest model

model = joblib.load("random_forest_ckd_model.pkl")

import random

def calculate_gfr(age, gender, serum_creatinine):
    """Calculate eGFR using CKD-EPI for adults and Schwartz for children (<13 years)."""
    
    if age < 13:
        # Generate random height based on age group (approximate range in cm)
        if age <= 1:
            height = random.uniform(50, 75)  # Infants
        elif 1 < age <= 5:
            height = random.uniform(75, 110)
        elif 5 < age <= 10:
            height = random.uniform(110, 140)
        else:  # 11-12 years
            height = random.uniform(140, 160)
        
        # Choose k based on gender
        k = 0.55 if gender.lower() == 'female' else 0.7
        
        egfr = (k * height) / serum_creatinine
        return round(egfr, 2)
    
    else:
        # CKD-EPI formula for age >= 13
        kappa = 0.7 if gender.lower() == 'female' else 0.9
        alpha = -0.241 if gender.lower() == 'female' else -0.302
        
        scr_ratio = serum_creatinine / kappa
        min_scr = min(scr_ratio, 1)
        max_scr = max(scr_ratio, 1)
        
        egfr = 142 * (min_scr ** alpha) * (max_scr ** -1.200) * (0.9938 ** age)
        
        if gender.lower() == 'female':
            egfr *= 1.012
        
        return round(egfr, 2)


def calculate_ckd_stage(gfr):
    """Determine CKD stage based on GFR value."""
    if gfr >= 90 :
        return "Stage 1 "
    elif 60 <= gfr < 90:
        return "Stage 2 "
    elif 45 <= gfr < 60:
        return "Stage 3a "
    elif 30 <= gfr < 45:
        return "Stage 3b "
    elif 15 <= gfr < 30:
        return "Stage 4 "
    else:
        return "Stage 5 "
        
@app.route('/')
def home():
    return render_template("web_400_10.html")

@app.route('/predict_400', methods=['POST'])
def predict():
    try:
        input_features = {
            'Age': int(request.form['age']),
            'Specific Gravity': float(request.form['sg']),
            'Albumin': int(request.form['al']),
            'Blood Glucose Random': int(request.form['bgr']),
            'Serum Creatinine': float(request.form['sc']),
            'Sodium': int(request.form['sod']),
            'Hemoglobin': float(request.form['hemo']),
            'Packed  Cell Volume': int(request.form['pcv']),
            'Red Blood Cell Count': float(request.form['rc']),
            'Hypertension': int(request.form['htn']),
            'Diabetes Mellitus': int(request.form['dm'])   
        }     
        age = input_features['Age']
        sc = input_features['Serum Creatinine']
        name = request.form.get('name', 'Unknown')
        gender = request.form.get('gender', 'Unknown')

        input_array = np.array(list(input_features.values())).reshape(1, -1)
        prediction = model.predict(input_array)
        result = "CKD Detected" if prediction[0] == 1 else "No CKD"

        # Calculate CKD stage if CKD is detected
        stage = None
        gfr = None
        if result == "CKD Detected":
            gfr = calculate_gfr(age, gender, sc)
            stage = calculate_ckd_stage(gfr)

        return render_template('result_400_10.html', 
                               prediction=result, 
                               input_data=input_features,
                               gfr=gfr,
                               name=name, 
                               gender=gender, 
                               stage=stage)
    
    except KeyError as ke:
        return jsonify({"error": f"Missing input field: {str(ke)}"}), 400
    
    except ValueError as ve:
        return jsonify({"error": f"Invalid input format: {str(ve)}"}), 400
    
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
