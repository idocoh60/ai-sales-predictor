
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import os
import openai
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai.api_key)

# Setup Flask app
app = Flask(__name__, template_folder=os.path.abspath("templates"), static_folder=os.path.abspath("static"))

# Load models and scalers
logistic_model = joblib.load(os.path.join("models", "logistic_model.pkl"))
linear_model = joblib.load(os.path.join("models", "linear_regressor_model.pkl"))
logistic_scaler = joblib.load(os.path.join("models", "logistic_scaler.pkl"))
regressor_scaler = joblib.load(os.path.join("models", "regressor_scaler.pkl"))

def generate_sales_pitch(data, recommended_product):
    probability = round(float(data.get("probability", 0)) * 100)

    prompt = f"""
אתה עוזר לנציג מכירות שמתכונן לשיחת טלפון עם לקוח. אין צורך במשפטים שלמים או טקסט רשמי. 
תן רק רשימה של ביטויים, משפטונים או מילים קצרות שנציג יכול לשלב בשיחה כדי להניע את הלקוח לקנות.

- אל תנסח משפטים ארוכים או שיחה שלמה.
- אל תבטיח הנחות או תנאים שאין בידי הנציג להבטיח.
- סגנון פשוט, ישיר, משכנע, כאילו חבר מדבר לחבר.

כתיבה בעברית טבעית, כאילו מדובר בנציג מכירות ישראלי שמדבר ללקוח רגיל.
אל תשתמש בשמות של מקומות, מותגים או לקוחות דמיוניים.
כל נקודה = משפטון קצר או ביטוי בלבד.

הנתונים על הלקוח:
- מספר רכישות קודמות: {data.get("NumPurchases")}
- סך ההוצאות: {data.get("TotalSpent")}
- ימים מאז הרכישה האחרונה: {data.get("DaysSinceLastPurchase")}
- סבירות שהלקוח יקנה: {probability}%
- מוצר מומלץ: {recommended_product}

הצג את התוצאה אך ורק כנקודות ברשימה (בולטים), בלי הקדמות ובלי סיכומים.
"""



    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "system",
                "content": "אתה יועץ מכירות חכם. תן משפטים שיווקיים מבוססי נתוני הלקוח בצורה חכמה, אישית, בלי להבטיח הנחות או להמציא עובדות. כל משפט צריך להיות משכנע ומדויק."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=700,
        temperature=0.85
    )

    return response.choices[0].message.content.strip()


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    num_purchases = int(data.get("NumPurchases", 1))
    total_spent = float(data.get("TotalSpent", 100.0))
    days_since = int(data.get("DaysSinceLastPurchase", 10))
    avg_spent = data.get("AverageSpent", round(total_spent / max(num_purchases, 1), 2))
    quantity = data.get("Quantity", 1)
    hour = data.get("Hour", 14)
    day_of_week = data.get("DayOfWeek", 3)

    df_main = pd.DataFrame([{
        "NumPurchases": num_purchases,
        "TotalSpent": total_spent,
        "AverageSpent": avg_spent,
        "DaysSinceLastPurchase": days_since
    }])

    results = {}
    
    # Logistic model prediction
    X_logistic = logistic_scaler.transform(df_main)
    logistic_pred = logistic_model.predict(X_logistic)[0]
    logistic_proba = logistic_model.predict_proba(X_logistic)[0][1]
    results["willBuy"] = bool(logistic_pred)
    results["probability"] = round(logistic_proba, 2)

    # Linear regression prediction
    X_reg = regressor_scaler.transform(df_main)
    amount_pred = linear_model.predict(X_reg)[0]
    results["expectedAmount"] = round(float(amount_pred), 2)

    # KNN block
    try:
        import numpy as np
        knn_model = joblib.load(os.path.join("models", "knn_model.pkl"))
        knn_label_encoder = joblib.load(os.path.join("models", "knn_label_encoder.pkl"))
        knn_features = np.array([[quantity, total_spent, hour, day_of_week]])
        knn_encoded = knn_model.predict(knn_features)[0]
        recommended_product = knn_label_encoder.inverse_transform([knn_encoded])[0]
        results["recommendedProduct"] = recommended_product
    except Exception as e:
        results["recommendedProduct"] = None
        results["knn_error"] = str(e)

    # GPT block
    try:
        sales_pitch = generate_sales_pitch(data, results.get("recommendedProduct", ""))
        results["salesPitch"] = sales_pitch
    except Exception as e:
        results["salesPitch"] = "Pitch unavailable due to GPT error."
        results["gpt_error"] = str(e)

    return jsonify(results)

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
