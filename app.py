from flask import Flask, request, jsonify, render_template
import numpy as np
try:
    from tensorflow import keras  # Update import for TensorFlow Keras
except ImportError:
    print("TensorFlow is not installed. Please install it using 'pip install tensorflow'.")
    exit(1)
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
import uuid

# Flask App Initialization
app = Flask(__name__)
CORS(app)

# SQLAlchemy Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///predictions.db'  # Example: SQLite database
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Define the Prediction model
class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    prediction_id = db.Column(db.String(36), unique=True, nullable=False)
    country = db.Column(db.String(100), nullable=False)
    disease = db.Column(db.String(100), nullable=False)
    prediction = db.Column(db.Float, nullable=False)
    range_min = db.Column(db.Float, nullable=False)
    range_max = db.Column(db.Float, nullable=False)
    histogram_bins = db.Column(db.PickleType, nullable=False)
    input_data = db.Column(db.PickleType, nullable=False)

# Initialize the database
with app.app_context():
    db.create_all()

@app.route('/web', methods=['GET'])
def web():
    return render_template('web.html')

# Load the trained model
try:
    model = keras.models.load_model("trained_model.keras")  # Use keras.models
    print("Model loaded successfully")
except Exception as e:
    print(f"Failed to load the model: {e}")
    exit(1)

# Prediction Endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Validate required fields
        required_fields = [
            "Population_Density", "Vaccination_Rate", "Mobility_Index",
            "Infection_Rate", "Mortality_Rate", "Disease", "Country",
            "Temperature", "Humidity"
        ]
        for field in required_fields:
            if field not in data or not isinstance(data[field], (int, float, str)):
                return jsonify({"error": f"Missing or invalid value for required field: {field}"}), 400

        # Prepare input data for the model
        try:
            input_features = np.array([[  # Ensure the input is in the correct format
                float(data["Population_Density"]),
                float(data["Vaccination_Rate"]),
                float(data["Mobility_Index"]),
                float(data["Infection_Rate"]),
                float(data["Mortality_Rate"]),
                float(data["Temperature"]),
                float(data["Humidity"])
            ]])
        except ValueError as ve:
            return jsonify({"error": f"Invalid input data: {ve}"}), 400

        # Generate prediction using the model
        try:
            prediction = model.predict(input_features).flatten()[0]  # Adjust indexing based on your model's output
        except Exception as e:
            return jsonify({"error": f"Model prediction failed: {e}"}), 500

        range_min = prediction * 0.8
        range_max = prediction * 1.2
        prediction_id = str(uuid.uuid4())

        # Generate histogram data
        histogram_bins = [prediction * 0.5, prediction * 0.75, prediction, prediction * 1.25, prediction * 1.5]

        # Store prediction in the database
        try:
            prediction_record = Prediction(
                prediction_id=prediction_id,
                country=data["Country"],
                disease=data["Disease"],
                prediction=prediction,
                range_min=range_min,
                range_max=range_max,
                histogram_bins=histogram_bins,
                input_data={
                    "population_density": data["Population_Density"],
                    "vaccination_rate": data["Vaccination_Rate"],
                    "mobility_index": data["Mobility_Index"],
                    "infection_rate": data["Infection_Rate"],
                    "mortality_rate": data["Mortality_Rate"],
                    "temperature": data["Temperature"],
                    "humidity": data["Humidity"]
                }
            )
            db.session.add(prediction_record)
            db.session.commit()
        except Exception as e:
            print(f"Failed to save prediction to the database: {e}")
            return jsonify({"error": "Failed to save prediction to the database"}), 500

        # Return the prediction response
        return jsonify({
            "prediction_id": prediction_id,
            "prediction": prediction,
            "range": {"min": range_min, "max": range_max},
            "disease": data["Disease"],
            "histogram_bins": histogram_bins
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint to Retrieve Prediction by ID
@app.route('/prediction/<prediction_id>', methods=['GET'])
def get_prediction(prediction_id):
    try:
        prediction = Prediction.query.filter_by(prediction_id=prediction_id).first()
        if not prediction:
            return jsonify({"error": "Prediction not found"}), 404

        # Convert the prediction object to a dictionary
        prediction_data = {
            "id": prediction.id,
            "prediction_id": prediction.prediction_id,
            "country": prediction.country,
            "disease": prediction.disease,
            "prediction": prediction.prediction,
            "range_min": prediction.range_min,
            "range_max": prediction.range_max,
            "histogram_bins": prediction.histogram_bins,
            "input_data": prediction.input_data
        }
        return jsonify(prediction_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/submit', methods=['POST'])
def submit():
    try:
        # Extract form data
        data = {
            "Population_Density": request.form.get("Population_Density"),
            "Vaccination_Rate": request.form.get("Vaccination_Rate"),
            "Mobility_Index": request.form.get("Mobility_Index"),
            "Infection_Rate": request.form.get("Infection_Rate"),
            "Mortality_Rate": request.form.get("Mortality_Rate"),
            "Disease": request.form.get("Disease"),
            "Country": request.form.get("Country"),
            "Temperature": request.form.get("Temperature"),
            "Humidity": request.form.get("Humidity")
        }

        # Call the prediction endpoint
        response = app.test_client().post('/predict', json=data)
        prediction_result = response.get_json()

        if response.status_code == 200:
            return render_template('result.html', result=prediction_result)
        else:
            return render_template('result.html', error=prediction_result.get("error", "Unknown error"))
    except Exception as e:
        return render_template('result.html', error=str(e))

# Run the Flask App
if __name__ == '__main__':
    app.run(debug=True)