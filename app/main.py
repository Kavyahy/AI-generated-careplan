from fastapi import FastAPI
from pydantic import BaseModel
import uuid
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

app = FastAPI()

diabetes_df = pd.read_csv("dataset/Diabetes.csv")
diabetes_df.columns = diabetes_df.columns.str.strip()

X_diabetes = diabetes_df[["BMI", "Glucose"]]
y_diabetes = diabetes_df["Outcome"]

diabetes_scaler = StandardScaler()
X_diabetes_scaled = diabetes_scaler.fit_transform(X_diabetes)

diabetes_model = LogisticRegression()
diabetes_model.fit(X_diabetes, y_diabetes)

print("Diabetes model trained")
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_diabetes_scaled,
    y_diabetes,
    test_size=0.2,
    random_state=42
)

diabetes_model_eval = LogisticRegression()
diabetes_model_eval.fit(X_train_d, y_train_d)

diabetes_pred_eval = diabetes_model_eval.predict(X_test_d)

print("\nDiabetes Model Performance")
print("Accuracy:", round(accuracy_score(y_test_d, diabetes_pred_eval) * 100, 2), "%")
print("Precision:", round(precision_score(y_test_d, diabetes_pred_eval) * 100, 2), "%")
print("Recall:", round(recall_score(y_test_d, diabetes_pred_eval) * 100, 2), "%")
print("F1 Score:", round(f1_score(y_test_d, diabetes_pred_eval) * 100, 2), "%")

heart_df = pd.read_csv("dataset/heart.csv")

stroke_df = pd.read_csv("dataset/stroke.csv")
kidney_df = pd.read_csv("dataset/kidney.csv")
liver_df = pd.read_csv("dataset/liver.csv")

X_heart = heart_df[["age", "trestbps", "chol", "thalach"]]
y_heart = heart_df["target"]

heart_scaler = StandardScaler()
X_heart_scaled = heart_scaler.fit_transform(X_heart)

heart_model = LogisticRegression()
heart_model.fit(X_heart, y_heart)

print("Heart model trained")

X_stroke = stroke_df[
    [
        "age",
        "hypertension",
        "heart_disease",
        "avg_glucose_level",
        "bmi",
        "smoking_status"
    ]
]

y_stroke = stroke_df["stroke"]

stroke_model = LogisticRegression(max_iter=1000)
stroke_model.fit(X_stroke, y_stroke)

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_stroke,
    y_stroke,
    test_size=0.2,
    random_state=42
)

stroke_model_eval = LogisticRegression(max_iter=1000)
stroke_model_eval.fit(X_train_s, y_train_s)

stroke_pred_eval = stroke_model_eval.predict(X_test_s)

print("\nStroke Model Performance")
print("Accuracy:", round(accuracy_score(y_test_s, stroke_pred_eval) * 100, 2), "%")
print("Precision:", round(precision_score(y_test_s, stroke_pred_eval) * 100, 2), "%")
print("Recall:", round(recall_score(y_test_s, stroke_pred_eval) * 100, 2), "%")
print("F1 Score:", round(f1_score(y_test_s, stroke_pred_eval) * 100, 2), "%")

print("Stroke model trained")

X_kidney = kidney_df[
    [
        "age",
        "bp",
        "bgr",
        "bu",
        "sc",
        "hemo"
    ]
]

y_kidney = kidney_df["ckd"]

kidney_model = LogisticRegression(max_iter=1000)
kidney_model.fit(X_kidney, y_kidney)

X_train_k, X_test_k, y_train_k, y_test_k = train_test_split(
    X_kidney,
    y_kidney,
    test_size=0.2,
    random_state=42
)

kidney_model_eval = LogisticRegression(max_iter=1000)
kidney_model_eval.fit(X_train_k, y_train_k)

kidney_pred_eval = kidney_model_eval.predict(X_test_k)

print("\nKidney Disease Model Performance")
print("Accuracy:", round(accuracy_score(y_test_k, kidney_pred_eval) * 100, 2), "%")
print("Precision:", round(precision_score(y_test_k, kidney_pred_eval) * 100, 2), "%")
print("Recall:", round(recall_score(y_test_k, kidney_pred_eval) * 100, 2), "%")
print("F1 Score:", round(f1_score(y_test_k, kidney_pred_eval) * 100, 2), "%")

print("Kidney model trained")

X_liver = liver_df[
    [
        "age",
        "total_bilirubin",
        "direct_bilirubin",
        "alkphos",
        "sgpt",
        "sgot",
        "proteins"
    ]
]

y_liver = liver_df["liver_disease"]

liver_model = LogisticRegression(max_iter=1000)
liver_model.fit(X_liver, y_liver)

X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
    X_liver,
    y_liver,
    test_size=0.2,
    random_state=42
)

liver_model_eval = LogisticRegression(max_iter=1000)
liver_model_eval.fit(X_train_l, y_train_l)

liver_pred_eval = liver_model_eval.predict(X_test_l)

print("\nLiver Disease Model Performance")
print("Accuracy:", round(accuracy_score(y_test_l, liver_pred_eval) * 100, 2), "%")
print("Precision:", round(precision_score(y_test_l, liver_pred_eval) * 100, 2), "%")
print("Recall:", round(recall_score(y_test_l, liver_pred_eval) * 100, 2), "%")
print("F1 Score:", round(f1_score(y_test_l, liver_pred_eval) * 100, 2), "%")

print("Liver model trained")

X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
    X_heart_scaled,
    y_heart,
    test_size=0.2,
    random_state=42
)

heart_model_eval = LogisticRegression()
heart_model_eval.fit(X_train_h, y_train_h)

heart_pred_eval = heart_model_eval.predict(X_test_h)

print("\nHeart Disease Model Performance")
print("Accuracy:", round(accuracy_score(y_test_h, heart_pred_eval) * 100, 2), "%")
print("Precision:", round(precision_score(y_test_h, heart_pred_eval) * 100, 2), "%")
print("Recall:", round(recall_score(y_test_h, heart_pred_eval) * 100, 2), "%")
print("F1 Score:", round(f1_score(y_test_h, heart_pred_eval) * 100, 2), "%")

class CarePlanRequest(BaseModel):
    patientName: str
    height: float
    weight: float
    pulse: float
    o2Saturation: float

    # ML Inputs
    age: float
    trestbps: float
    chol: float
    thalach: float
    glucose: float
    # Stroke
    hypertension: float
    heartDiseaseHistory: float
    smokingStatus: float

    # Kidney
    bp: float
    bgr: float
    bu: float
    sc: float
    hemo: float

    # Liver
    total_bilirubin: float
    direct_bilirubin: float
    alkphos: float
    sgpt: float
    sgot: float
    proteins: float

    diagnosis: str
    symptoms: str

@app.get("/")
def home():
    return {"message": "AI Care Plan System Running"}

@app.post("/generate-care-plan")
def generate_care_plan(data: CarePlanRequest):

    # BMI Calculation
    bmi = data.weight / ((data.height / 100) ** 2)
    
    diagnosis_list = set()

    stroke_pred = stroke_model.predict([[
       data.age,
       data.hypertension,
       data.heartDiseaseHistory,
       data.glucose,
       bmi,
       data.smokingStatus
    ]])[0]
    print("Stroke Prediction:", stroke_pred)

    if stroke_pred == 1:
        diagnosis_list.add("stroke risk")
    
    kidney_pred = kidney_model.predict([[
        data.age,
        data.bp,
        data.bgr,
        data.bu,
        data.sc,
        data.hemo
    ]])[0]
    print("Kidney Prediction:", kidney_pred)

    if kidney_pred == 1:
       diagnosis_list.add("kidney disease")

    liver_pred = liver_model.predict([[
       data.age,
       data.total_bilirubin,
       data.direct_bilirubin,
       data.alkphos,
       data.sgpt,
       data.sgot,
       data.proteins
    ]])[0]
    print("Liver Prediction:", liver_pred)

    if liver_pred == 1:
        diagnosis_list.add("liver disease")      

    diabetes_input = diabetes_scaler.transform([[bmi, data.glucose]])
    diabetes_pred = diabetes_model.predict(diabetes_input)[0]
    
    print("Diabetes Prediction:", diabetes_pred)
   
    if diabetes_pred == 1:
        diagnosis_list.add("diabetes")

    heart_input = heart_scaler.transform([[
      data.age,
      data.trestbps,
      data.chol,
      data.thalach
    ]])

    prob = heart_model.predict_proba(heart_input)[0][1]
    print("Heart Probability:", prob)

    heart_reason = "ML Model"
    if prob > 0.3:
       diagnosis_list.add("heart disease")

    elif (
         data.age > 55 and
         data.trestbps > 150 and
         data.chol > 240
    ):
         diagnosis_list.add("heart disease")
         heart_reason = "Rule-based detection"
   
    

    if data.pulse > 100:
        diagnosis_list.add("hypertension")

    if data.o2Saturation < 95:
        diagnosis_list.add("respiratory issue")

    if bmi < 18.5:
        diagnosis_list.add("underweight")

    if not diagnosis_list:
        diagnosis_list.add("normal")

    diagnosis = ", ".join(sorted(diagnosis_list))
    print("Diagnosis List:", diagnosis_list)

    risk = "Low"
    diet = []
    exercise = []
    medication = []
    monitoring = []
    lifestyle = []

    followup = "Monthly health checkup"

    if "diabetes" in diagnosis:

        risk = "Medium"

        diet.extend([
            "Avoid sugary foods and soft drinks",
            "Eat whole grains and high-fiber foods",
            "Include vegetables and fruits in meals",
            "Drink at least 2 liters of water daily"
         ])
        exercise.extend([
            "30 minutes walking daily",
            "Light stretching exercises"
         ])
        medication.extend([
            "Take diabetes medicines regularly",
            "Consult doctor for insulin dosage if needed"
         ])
        monitoring.extend([
            "Check blood sugar levels daily",
            "Monitor fasting glucose regularly"
        ])
        lifestyle.extend([
            "Maintain healthy body weight",
            "Avoid smoking and alcohol"
         ])

    if "heart disease" in diagnosis:

        risk = "High"
        diet.extend([
            "Follow low cholesterol diet",
            "Avoid oily and fried foods",
            "Reduce salt intake",
            "Increase fruits and green vegetables"
         ])

        exercise.extend([
             "Light walking for 20 minutes",
             "Avoid heavy physical activity"
         ])

        medication.extend([
             "Take heart medications as prescribed",
            "Consult cardiologist regularly"
         ])

        monitoring.extend([
             "Monitor blood pressure twice daily",
             "Regular ECG and heart checkup"
         ])

        lifestyle.extend([
             "Reduce stress through meditation",
            "Maintain proper sleep schedule"
         ])

        followup = "Visit cardiologist within 7 days"

    if "stroke risk" in diagnosis:

        risk = "High"

        diet.extend([
            "Reduce salt intake",
            "Eat fruits and vegetables",
            "Avoid processed foods"
     ])

        monitoring.extend([
            "Monitor blood pressure regularly"
        ])

        followup = "Neurologist consultation within 7 days"

    if "kidney disease" in diagnosis:

        risk = "High"

        diet.extend([
            "Reduce sodium intake",
            "Drink adequate water",
            "Avoid excessive protein"
        ])

        monitoring.extend([
            "Monitor kidney function regularly"
        ])

        followup = "Nephrologist consultation within 7 days"

    if "liver disease" in diagnosis:

        risk = "High"

        diet.extend([
           "Avoid alcohol",
           "Eat a balanced diet",
           "Reduce fatty foods"
        ])

        monitoring.extend([
           "Liver function test monitoring"
        ])

        followup = "Hepatologist consultation within 7 days"

    if "hypertension" in diagnosis:

        diet.extend([
             "Low salt diet",
             "Avoid processed foods"
         ])

        monitoring.extend([
              "Check blood pressure daily"
         ])

    if "respiratory issue" in diagnosis:

        risk = "Critical"

        exercise = [
             "Avoid physical activity until recovery"
         ]

        monitoring.extend([
             "Monitor oxygen saturation regularly"
         ])

        medication.extend([
              "Consult pulmonologist immediately"
         ])

        lifestyle.extend([
              "Avoid dust and smoking areas"
         ])

        followup = "Emergency consultation required"

    if "underweight" in diagnosis:

        diet.extend([
             "Increase protein intake",
            "Consume nutritious meals regularly"
         ])

    if diagnosis == "normal":

        diet = [
               "Maintain balanced diet"
         ]

        exercise = [
               "Daily walking and regular exercise"
         ]

        medication = [
               "No medication required"
         ]

        monitoring = [
            "Routine health checkups"
         ]

        lifestyle = [
             "Maintain healthy lifestyle"
         ]
    
    return {
        "patientId": str(uuid.uuid4()),
        "patientName": data.patientName,
        "bmi": round(bmi, 2),
        "diagnosis": diagnosis,
        "heartRiskProbability": round(prob, 2),
        "heartDetectionMethod": heart_reason,
        "carePlan": {
            "riskLevel": risk,
            "diet": diet,
            "exercise": exercise,
            "medication": medication,
            "monitoring": monitoring,
            "lifestyle": lifestyle,
            "followUp": followup
        },
        "doctorReview": {
            "status": "Pending",
            "remarks": "Doctor will review and edit if needed"
        }
    }

    