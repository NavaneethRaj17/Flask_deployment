import numpy as np
from flask import Flask, request, jsonify, render_template # Flask Creates the app, request: Retrieves data from the form submitted by the user
#render_template: Renders HTML files for the front-end.jsonify: Can be used to return JSON responses 
import pickle #Loads the pre-trained model from the serialized model.pkl file.

# Create flask app
flask_app = Flask(__name__) #creates an instance of a Flask application. The __name__ parameter tells Flask whether the app is running as the main program or imported as a module.

#Load the pickle model
model = pickle.load(open("model.pkl", "rb")) #Opens the file model.pkl in binary read mode.

@flask_app.route("/") #The @flask_app.route("/") decorator binds the Home() function to the root URL (http://127.0.0.1:5000/).
def Home():
    return render_template("index.html") # Home() function returns the index.html file, which serves as the front-end for the app.
#Ensure index.html exists in a folder named templates in the same directory as this script.

@flask_app.route("/predict", methods = ["POST"]) #Binds the predict() function to the /predict URL.Accepts HTTP POST requests (used for sending form data)
def predict(): 
    float_features = [float(x) for x in request.form.values()] #request.form.values(): Retrieves values submitted via the form
    #float(x): Converts the input values to floating-point numbers (expected format for ML models)
    features = [np.array(float_features)] #Converts the input into a NumPy array (a common format for ML models).
    prediction = model.predict(features)
    return render_template("index.html", prediction_text = "The flower species is {}".format(prediction))

if __name__ == "__main__": #Ensures the app runs only if the script is executed directly.
    flask_app.run(debug=True) #Starts the Flask development server.debug=True flag enables automatic server restarts on code changes and provides an interactive debugger for errors.