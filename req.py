import requests

url = 'http://localhost:8000/predict'

# Load the image file
with open('image.jpg', 'rb') as f:
    image = f.read()

# Send the request
response = requests.post(url, data=image, headers={'Content-Type': 'image/jpeg'})

# Get the prediction from the response
prediction = response.json()['prediction']
