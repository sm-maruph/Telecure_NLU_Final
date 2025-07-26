from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import joblib
from preprocessing import bn_preprocess, bn_tokenizer  # keep if you want for other uses

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.DEBUG)

# Load the full pipeline model
try:
    model = joblib.load("nlu_pipeline_model.joblib")
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Model load failed: {e}")
    raise RuntimeError("Cannot start API without model")

# Intent map
intent_map = {
    "find_doctor": {"response": "ডাক্তার দেখাও", "url": "/find"},
    "prescriptions": {"response": "প্রেসক্রিপশন দেখাও", "url": "/dashboard/user/pres"},
    "home": {"response": "হোমপেজে ফিরে যাও", "url": "/"},
    "appointment": {"response": "অ্যাপয়েন্টমেন্ট দিন", "url": "/dashboard/user/appointment"},
    "medicines": {"response": "ঔষধ তালিকা", "url": "/dashboard/user/medicines"},
    "my_report": {"response": "আমার রিপোর্ট", "url": "/dashboard/user/report"},
    "my_booking": {"response": "আমার বুকিং", "url": "/dashboard/user/bookings"},
    "edit_profile": {"response": "প্রোফাইল এডিট করো", "url": "/dashboard/user/settings"},
    "join_call": {"response": "কল জয়েন করো", "url": "/dashboard/user/meeting"},
    "back": {"response": "পিছনে যাও", "url": "back"}
}

@app.route("/")
def hello_world():
    return "<p>Hello world!</p>"

@app.route("/predict", methods=["POST"])
def predict_intent():
    if not request.is_json:
        return jsonify({"error": "Request must be in JSON format"}), 415

    data = request.get_json()
    message = data.get("text", "").strip()

    if not message:
        return jsonify({"error": "No input provided"}), 400

    try:
        # Directly use the pipeline to predict
        intent = model.predict([message])[0]

        print(f"📥 Received: '{message}' → 🎯 Intent: '{intent}'")

        intent_data = intent_map.get(intent)
        if not intent_data:
            return jsonify({
                "intent": "unknown",
                "message": "দুঃখিত, আমি বুঝতে পারিনি",
                "url": None
            })

        return jsonify({
            "intent": intent,
            "message": intent_data["response"],
            "url": intent_data["url"]
        })

    except Exception as e:
        print(f"⚠️ Server error: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5005)
