from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load your trained model here (replace 'my_model_knn.pkl' with your actual model file)
knn_from_joblib = joblib.load('heart.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        json_data = request.json  # Get input data in JSON format
        query_df = pd.DataFrame(json_data)  # Convert JSON to DataFrame
        # Preprocess the input data (e.g., handle missing values, encode categorical features)
        # Example: query = preprocess_data(query_df)
        prediction = knn_from_joblib.predict(query_df)  # Make predictions
        return jsonify({'prediction': list(prediction)})  # Return predictions as JSON
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
