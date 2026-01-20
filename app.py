from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load Model
MODEL_PATH = 'model/house_price_model.pkl'
try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}. Make sure to train the model first.")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return render_template('index.html', prediction_text="Error: Model not loaded.")

    try:
        # Extract features from form
        features = [
            float(request.form['OverallQual']),
            float(request.form['GrLivArea']),
            float(request.form['TotalBsmtSF']),
            float(request.form['GarageCars']),
            float(request.form['FullBath']),
            float(request.form['YearBuilt'])
        ]
        
        final_features = [np.array(features)]
        prediction = model.predict(final_features)
        output = round(prediction[0], 2)

        return render_template('index.html', prediction_text='Estimated House Price: ${:,.2f}'.format(output))
    except Exception as e:
        return render_template('index.html', prediction_text=f'Error: {str(e)}')

if __name__ == "__main__":
    app.run(debug=True)
