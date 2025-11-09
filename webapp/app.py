import os
import smtplib
from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from g4f.client import Client
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Load the best saved model
model = joblib.load('best_model.pkl')
print("Loaded best model from 'best_model.pkl'.")

# Define the 10 feature names used in the model
features = ['Machine', 'DebugSize', 'MajorImageVersion', 'ExportSize',
            'IatVRA', 'NumberOfSections', 'SizeOfStackReserve',
            'DllCharacteristics', 'ResourceSize', 'BitcoinAddresses']

# Initialize the GPT-4o-mini client
client = Client()

# Email configuration
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
EMAIL_FROM = os.getenv('daminmain@gmail.com')  # e.g., "daminmain@gmail.com"
EMAIL_PASSWORD = os.getenv('kpqtxqskedcykwjz')  # e.g., "your-app-password"
EMAIL_TO = 'bashith67@gmail.com'  # Receiver's email

def send_email(prediction, confidence, gpt_response):
    """Send an email with the prediction results."""
    if not EMAIL_FROM or not EMAIL_PASSWORD:
        print("Error: Email credentials not set in environment variables.")
        return

    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_FROM
        msg['To'] = EMAIL_TO
        msg['Subject'] = "Prediction Results"

        body = f"""
        Model Prediction: {prediction}
        Confidence: {confidence * 100:.2f}% (if available)
        AI Analysis: {gpt_response}
        """
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_FROM, EMAIL_PASSWORD)
        server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
        server.quit()

        print("Email sent successfully.")
    except Exception as e:
        print(f"Failed to send email: {str(e)}")

@app.route('/')
def landing():
    return render_template('landing.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Create a sample data dictionary using form input for all features
        sample_data = {feature: int(request.form[feature]) for feature in features}
        sample_df = pd.DataFrame([sample_data])

        # Make a prediction
        prediction = model.predict(sample_df)[0]

        try:
            probabilities = model.predict_proba(sample_df)[0]
            confidence = max(probabilities)
        except AttributeError:
            probabilities = None
            confidence = None

        # Map the prediction (assuming 1 = Benign, 0 = Malicious)
        label_mapping = {1: "Benign", 0: "Malicious"}
        predicted_label = label_mapping.get(prediction, "Unknown")

        # Get AI analysis using GPT-4o-mini
        input_text = f"The model predicted {predicted_label} with {confidence * 100:.2f}% confidence. Provide insights."
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": input_text}],
            web_search=False
        )
        gpt_response = response.choices[0].message.content

        # Send email notification with the prediction and AI analysis
        print("About to send email...")
        send_email(predicted_label, confidence, gpt_response)
        print("Email function completed.")

        return render_template('result.html', prediction=predicted_label, confidence=confidence, gpt_response=gpt_response)

    return render_template('predict.html')

@app.route('/chat', methods=['GET', 'POST'])
def chat():
    if request.method == 'POST':
        user_input = request.form['user_input']

        # Prepare input for GPT-4o-mini with a focus on cyber network solutions
        chatbot_input = f"User: {user_input}\nChatbot: Please provide information and solutions related to AI-ENHANCED INTRUSION DETECTION SYSTEM."

        chatbot_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": chatbot_input}],
            web_search=False
        )

        gpt_response = chatbot_response.choices[0].message.content
        return jsonify({'response': gpt_response})

    return render_template('chat.html')

if __name__ == '__main__':
    app.run(debug=True)