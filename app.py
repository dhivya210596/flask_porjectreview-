from flask import Flask, request, jsonify
from joblib import load

app = Flask(__name__)

# Load saved model
model = load('product_sentiment_model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    review_text = [data['review']]
    prediction = model.predict(review_text)[0]
    sentiment = "Positive" if prediction == 1 else "Negative"
    return jsonify({'sentiment': sentiment})

if __name__ == '__main__':
    app.run(debug=True)


