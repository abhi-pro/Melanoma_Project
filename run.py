# Import the required modules
from flask import Flask, request, render_template, redirect, session, make_response
# Import the ObjectId class from pymongo
from bson.objectid import ObjectId
import tensorflow as tf
import database
from database import store_image_and_prediction
from database import hair_remove
import numpy as np
import base64
import cv2
#import datetime
from datetime import datetime
import pdfkit
import subprocess


# Load the pre-trained model
model = tf.keras.models.load_model('model_demo.h5')

# Initialize a Flask application
app = Flask(__name__)
app.secret_key = 'root'
# # Define the endpoint for the home page
# @app.route('/')
# def index():
#     return render_template('index.html')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        # Retrieve the user data from the database
        user_data = database.get_user_by_username(username)
        
        if user_data and user_data['password'] == password:
            # Authentication successful, store the username in the session
            session['username'] = username
            # Redirect to the home page
            return redirect('/home')
        else:
            # Authentication failed, display an error message
            error_message = 'Invalid username or password. Please try again.'
            return render_template('index.html', error_message=error_message)

    return render_template('index.html')




# Define the endpoint for the home page
@app.route('/home')
def home():
    # Check if the user is logged in
    if 'username' not in session:
        return redirect('/')
    
    # Get the username from the session
    username = session['username']
    
    # Retrieve the user data from the database
    user_data = database.get_user_by_username(username)
    
    if user_data:
        # Retrieve all past results for the user
        past_results_data = database.get_past_results(username)
        
        # Convert date and time strings to datetime objects
        for result in past_results_data:
            datetime_str = f"{result['datetime']} {result['time']}"
            result['datetime'] = datetime.strptime(datetime_str, "%d %B %Y %H:%M")


        # Sort the past results by datetime in descending order
        sorted_past_results = sorted(past_results_data, key=lambda x: x['datetime'], reverse=True)


        # Convert the Base64-encoded images back to bytes
        # for result in past_results_data:
        #     result['image'] = base64.b64decode(result['image'])
        
        # Render the home template with the user data and past results
        return render_template('home.html', user=user_data, past_results=sorted_past_results)
    else:
        # Redirect to the login page if the user data is not found
        return redirect('/')


# Define the endpoint for updating the model
@app.route('/update', methods=['POST'])
def update():
    # Get the new data from the request
    new_data = request.json
    data = {
        'username': session['username'],
        'image': encoded_image,
        'prediction': prediction_label
    }
    # Store the new data in the database
    store_new_data(new_data)  # Note: You need to define this function to store the new data in a database.
    
    # Retrain the model on the new data
    model = retrain_model(model, new_data)  # Note: You need to define this function to retrain the model on the new data.
    
    # Save the updated model
    model.save('melanoma_detection.h5')
    
    # Return a success response
    return jsonify({'status': 'success'})

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        name = request.form['name']
        age = request.form['age']
        gender = request.form['gender']
        email = request.form['email']
        phone = request.form['phone']
        
        # Check if the username already exists in the database
        user_data = database.get_user_by_username(username)
        if user_data:
            # Username already exists, display a warning message
            return render_template('register.html', message='Username already exists', message_color='red')
        
        # Store the new user information in the database
        data = {
            'username': username,
            'password': password,
            'name': name,
            'age': age,
            'gender': gender,
            'email': email,
            'phone': phone
        }
        database.store_new_data(data)
        
        # Registration successful, redirect to the login page
        return redirect('/')
    
    return render_template('register.html')

@app.route('/generate_report', methods=['GET'])
def generate_report():
    # Check if the user is logged in
    if 'username' not in session:
        return redirect('/')

    # Get the username from the session
    username = session['username']

    # Retrieve the user data from the database
    user_data = database.get_user_by_username(username)

    if user_data:
        # Retrieve all past results for the user
        past_results_data = database.get_past_results(username)

        # Convert date and time strings to datetime objects
        for result in past_results_data:
            datetime_str = f"{result['datetime']} {result['time']}"
            result['datetime'] = datetime.strptime(datetime_str, "%d %B %Y %H:%M")

        # Sort the past results by datetime in descending order
        sorted_past_results = sorted(past_results_data, key=lambda x: x['datetime'], reverse=True)

    # # Render the data in a template
    rendered_template = render_template('report_template.html', user=user_data, past_results = sorted_past_results)
    # return render_template('report_template.html', user=user_data, past_results = sorted_past_results)

    # Specify the path to the wkhtmltopdf executable
    wkhtmltopdf_path = '/usr/bin/wkhtmltopdf'

    # Configure pdfkit with the path to wkhtmltopdf
    config = pdfkit.configuration(wkhtmltopdf=wkhtmltopdf_path)

    options = {'page-size': 'Letter', 'encoding': 'UTF-8','disable-local-file-access': None}

    try:
        # Generate PDF from string using pdfkit with the configured path and options
        pdf = pdfkit.from_string(rendered_template, False, configuration=config, options=options)
    
        # Create a response with PDF content
        response = make_response(pdf)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = 'attachment; filename={}.pdf'.format(username)
    
        return response
    
    except OSError as e:
        # Print the error message from wkhtmltopdf
        print('wkhtmltopdf error:', e)

        # You can also capture the standard error output from wkhtmltopdf for further analysis
        # stderr = e.stderr
        # print('wkhtmltopdf stderr:', stderr)

        # Handle the error gracefully and provide an appropriate response to the user
        return 'Error: Failed to generate PDF report'

    except Exception as e:
        # Handle any other exceptions that might occur during PDF generation
        print('PDF generation error:', e)
        return 'Error: Failed to generate PDF report'


# Define the endpoint for making predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'username' not in session:
        return redirect('/')


    # Get the image file from the request
    image = request.files['image']
    
    # Read the image data separately
    image_data = image.read()
    
    # Decode the image data
    decoded_image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_UNCHANGED)
    
    # Preprocess the image
    # ...
    
    # Resize the image to 240x240
    resized_image = cv2.resize(decoded_image, (240, 240))
    
    # Preprocess by removing hair from image
    image_after_hair_removed = hair_remove(resized_image)
    
    # Convert the image to a Numpy array
    image_array = np.array(image_after_hair_removed)

    image_array = (image_array - 127.5) / 127.5
    
    # Expand the dimensions of the image array to match the input shape of the model
    image = np.expand_dims(image_array, axis=0)
    
    # # Make a prediction with the model
    # prediction = model.predict(image)
    
    # # Extract the predicted value from the nested list
    # prediction_value = float(prediction[0][0])
    
    
    # Extract the probability of the positive class
    probabilities = model.predict(image)
    
    # Handle single probability value
    if len(probabilities.shape) > 1:
        positive_probability = probabilities[0][0]
    else:
        positive_probability = probabilities[0]

     # Determine the output label based on the prediction value
    if positive_probability >= 0.5:
        prediction_label = "Malignant"
    else:
        prediction_label = "Benign"

    positive_probability_percentage = positive_probability * 100
    positive_probability_percentage_rounded = round(positive_probability_percentage, 2)

    # Encode the image data in base64 format
    encoded_image = base64.b64encode(image_data).decode('utf-8')
    
    # Encode the image after hair removal in base64 format
    encoded_image_after_hair_removal = base64.b64encode(cv2.imencode('.jpg', image_after_hair_removed)[1]).decode('utf-8')
    
    # Create a data dictionary to store the image and prediction information
    data = {
        'username': session['username'],
        'image': encoded_image,
        'prediction': prediction_label,
        'probability': positive_probability_percentage_rounded.item(),  # Convert probability to a float
        'datetime': datetime.now().strftime('%d %B %Y'),  # Format the date as "12 March 2023"
        'time': datetime.now().strftime('%H:%M')  # Format the time as "15:30"
    }
    
    # Store the image and prediction in the database
    inserted_id = store_image_and_prediction(data)
    
    # Redirect to the home page
    return redirect('/home')




# Define the endpoint for logging out
@app.route('/logout')
def logout():
    # Clear the session data
    session.clear()
    # Redirect to the login page
    return redirect('/')




# Run the application if the script is being run directly
if __name__ == '__main__':
    app.run(debug=True)
