from fastapi import FastAPI
from pydantic import BaseModel
import uuid
import pandas as pd
from sklearn.linear_model import LogisticRegression

app = FastAPI()

diabetes_df = pd.read_csv("dataset/Diabetes.csv")
diabetes_df.columns = diabetes_df.columns.str.strip()

X_diabetes = diabetes_df[["BMI", "Glucose"]]
y_diabetes = diabetes_df["Outcome"]

diabetes_model = LogisticRegression()
diabetes_model.fit(X_diabetes, y_diabetes)

print("Diabetes model trained")

heart_df = pd.read_csv("dataset/heart.csv")

X_heart = heart_df[["age", "trestbps", "chol", "thalach"]]
y_heart = heart_df["target"]

heart_model = LogisticRegression()
heart_model.fit(X_heart, y_heart)

print("Heart model trained")

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

    diagnosis: str
    symptoms: str

@app.get("/")
def home():
    return {"message": "AI Care Plan System Running"}

@app.post("/generate-care-plan")
def generate_care_plan(data: CarePlanRequest):

    # BMI Calculation
    bmi = data.weight / ((data.height / 100) ** 2)

    diagnosis_list = []

    diabetes_pred = diabetes_model.predict([[bmi, data.glucose]])[0]
    if diabetes_pred == 1:
        diagnosis_list.append("diabetes")

    prob = heart_model.predict_proba([[
       data.age,
       data.trestbps,
       data.chol,
       data.thalach
    ]])[0][1]

    heart_reason = "ML Model"
    if prob > 0.3:
       diagnosis_list.append("heart disease")

    elif (
         data.age > 55 and
         data.trestbps > 150 and
         data.chol > 240
    ):
         diagnosis_list.append("heart disease")
         heart_reason = "Rule-based detection"
   
    if prob > 0.05:
       diagnosis_list.append("heart disease")

    if data.pulse > 100:
        diagnosis_list.append("hypertension")

    if data.o2Saturation < 95:
        diagnosis_list.append("respiratory issue")

    if bmi < 18.5:
        diagnosis_list.append("underweight")

    if not diagnosis_list:
        diagnosis_list.append("normal")

    diagnosis = ", ".join(diagnosis_list)

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

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
        X_diabetes,
        y_diabetes,
        test_size=0.2,
        random_state=42
    )

    diabetes_model_eval = LogisticRegression()
    diabetes_model_eval.fit(X_train_d, y_train_d)

    diabetes_pred = diabetes_model_eval.predict(X_test_d)

    diabetes_accuracy = accuracy_score(y_test_d, diabetes_pred)

    print("Diabetes Model Accuracy:", round(diabetes_accuracy * 100, 2), "%")

    X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(
        X_heart,
        y_heart,
        test_size=0.2,
        random_state=42
    )

    heart_model_eval = LogisticRegression(max_iter=1000)
    heart_model_eval.fit(X_train_h, y_train_h)

    heart_pred = heart_model_eval.predict(X_test_h)

    heart_accuracy = accuracy_score(y_test_h, heart_pred)

    print("Heart Disease Model Accuracy:", round(heart_accuracy * 100, 2), "%")