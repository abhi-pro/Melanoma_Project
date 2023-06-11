import pymongo
import numpy as np
import cv2
import base64
import CNN
import numpy as np
import h5py
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Concatenate, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Connect to the MongoDB database
client = pymongo.MongoClient("mongodb://localhost:27017/")
db = client["melanoma_db"]
collection = db["images"]

# Function to store the new data in the database
def store_new_data(data):
    # Ensure the 'image' key is present in the dictionary
    if 'image' in data:
        # Convert the image to a Base64-encoded string
        image = base64.b64encode(data['image']).decode('utf-8')
        data['image'] = image
    
    result = collection.insert_one(data)
    return result.inserted_id


# Code of retraining will be called from this part
count = 0
def store_image_and_prediction(data):
    global count
    
    result = collection.insert_one(data)
    count += 1
    print("Dataset count")
    print(count)
    if count == 1000:
        retrain()
        count = 0
    
    return result.inserted_id



def hair_remove(image):
    
    # convert image to grayScale
    grayScale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # kernel for morphologyEx
    kernel = cv2.getStructuringElement(1,(17,17))
    
    # apply MORPH_BLACKHAT to grayScale image
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    
    # apply thresholding to blackhat
    _,threshold = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)
    
    # inpaint with original image and threshold image
    final_image = cv2.inpaint(image,threshold,1,cv2.INPAINT_TELEA)
    
    return final_image

# Function to retrieve all past image uploads by a user
def get_past_results(username):
    past_results = []
    for data in collection.find({'username': username}):
        # Ensure the 'image' key is present in the dictionary
        if 'image' in data:
            image = data['image']
            predicted_value = data['prediction']
            datetime = data['datetime']
            time = data['time']
            # Check if 'probability' key is present in the dictionary
            if 'probability' in data:
                probability = data['probability']
            else:
                probability = 0
            # Convert the Base64-encoded image back to bytes
            # image = base64.b64decode(image)        
            past_results.append({'image': image, 'prediction': predicted_value, 'datetime': datetime, 'time': time, 'probability': probability})

    return past_results



# ...
def get_user_by_username(username):
    return collection.find_one({'username': username})
# ...


# Function to retrain the model on the new data
# def retrain_model(model, new_data):
#     # Load the new data from the database
#     X = []
#     y = []
#     for data in collection.find():
#         image = data['image']
#         label = data['label']
        
#         X.append(image)
#         y.append(label)
    
#     X = np.array(X)
#     y = np.array(y)
    
#     # Train the model on the new data
#     model.fit(X, y)
    
#     return model

# Function to preprocess the image
def preprocess_image(image):
    # Decode the image from binary to a numpy array
    image = np.frombuffer(image, dtype=np.uint8)
    
    # Resize the image
    image = cv2.resize(image, (224, 224))
    
    # Normalize the image
    image = image / 255
    
    # Expand the dimensions of the image
    image = np.expand_dims(image, axis=0)
    
    return image


def load_dataset_from_hdf5(file_path):
    with h5py.File(file_path, 'r') as f:
        x_train = np.array(f['x_train'])
        y_train = np.array(f['y_train']).astype(int)
    return x_train, y_train

def store_dataset_in_mongodb(x_train, y_train):
    # Create a list of documents to insert into the MongoDB collection
    documents = []
    for i in range(len(x_train)):
        document = {
            'image': x_train[i],
            'label': y_train[i]
        }
        documents.append(document)

    # Insert the documents into the collection
    collection.insert_many(documents)

import base64
import cv2
import numpy as np

def load_dataset_from_mongodb():
    # Retrieve the documents from the MongoDB collection
    documents = collection.find()

    # Extract the images and labels from the documents
    images = []
    labels = []
    for document in documents:
        if 'username' in document and 'image' in document:
            image_base64 = document['image']
            diagnosis = document['prediction']

            # Convert the base64-encoded image to bytes
            image_bytes = base64.b64decode(image_base64)
            
            # Decode the image bytes into an image array
            image_array = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            
            # Resize the image to the desired shape (240, 240)
            resized_image = cv2.resize(image_array, (240, 240))
            resized_image = (resized_image - 127.5) / 127.5

            images.append(resized_image)
            print(resized_image)
            # Assign label based on diagnosis
            label = 0
            if diagnosis == 'Malignant':
                label = 1
            labels.append(label)

    # Convert the images and labels to Numpy arrays
    x_train = np.array(images)
    y_train = np.array(labels)

    return x_train, y_train




def preprocess_data(x_train, y_train, x_validation, y_validation):
    # Shuffle the data
    x_train, y_train = shuffle(x_train, y_train, random_state=42)

    # Convert labels to one-hot encoded format
    y_train_encoded = tf.keras.utils.to_categorical(y_train, num_classes=2)
    y_validation_encoded = tf.keras.utils.to_categorical(y_validation, num_classes=2)

    return x_train, y_train_encoded, x_validation, y_validation_encoded

def create_model(dropout_rate):
    def fire_module(x, squeeze_filters, expand_filters):
        squeeze = Conv2D(squeeze_filters, (1, 1), activation='relu')(x)
        expand_1x1 = Conv2D(expand_filters, (1, 1), activation='relu')(squeeze)
        expand_3x3 = Conv2D(expand_filters, (3, 3), padding='same', activation='relu')(squeeze)
        output = Concatenate()([expand_1x1, expand_3x3])
        return output

    # Input shape
    input_shape = (240, 240, 3)

    # Create the SqueezeNet model
    input_layer = tf.keras.Input(shape=input_shape)

    x = Conv2D(64, (3, 3), strides=(2, 2), activation='relu')(input_layer)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = fire_module(x, 16, 64)
    x = fire_module(x, 16, 64)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = fire_module(x, 32, 128)
    x = fire_module(x, 32, 128)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = fire_module(x, 48, 192)
    x = fire_module(x, 48, 192)
    x = fire_module(x, 64, 256)
    x = fire_module(x, 64, 256)

    # Regularization with Dropout
    x = Dropout(dropout_rate)(x)

    x = Conv2D(2, (1, 1), activation='relu')(x)

    x = GlobalAveragePooling2D()(x)
    output_layer = Dense(2, activation='softmax')(x)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model

def train_model(model, x_train, y_train_encoded, x_validation, y_validation_encoded, batch_size, epochs):
    # Early Stopping
    early_stopping = EarlyStopping(patience=5, restore_best_weights=True)

    # Set the batch size in model.fit
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Learning rate scheduler for adjusting learning rate during training
    def learning_rate_scheduler(epoch):
        return 1e-3 * 0.95 ** (epoch + epochs)

    # Create a learning rate scheduler callback
    learning_rate = tf.keras.callbacks.LearningRateScheduler(learning_rate_scheduler, verbose=1)

    class_weights = {0: 0.631, 1: 2.398}

    history = model.fit(x_train, y_train_encoded, batch_size=batch_size, epochs=epochs, class_weight=class_weights, validation_data=(x_validation, y_validation_encoded), verbose=1, callbacks=[learning_rate, early_stopping])

    # Save the model
    model.save("model00.h5")

    return history

def retrain():
    # Load your dataset from .hdf5 file
    # x_train, y_train = load_dataset_from_hdf5('dataset_cycleDAG_240x240_HairRemoval.hdf5')
    # x_temp, y_temp = load_dataset_from_hdf5('dataset_cycleDAG_2_240x240_hairremoval_benign.hdf5')

    # Concatenate with the previous data
    # x_train = np.concatenate((x_train, x_temp), axis=0)
    # y_train = np.concatenate((y_train, y_temp), axis=0)

    # Store the dataset in MongoDB
    # store_dataset_in_mongodb(x_train, y_train)

    # Load the dataset from MongoDB
    # x_temp1, y_temp1 = load_dataset_from_mongodb()

    # Load the dataset from MongoDB
    x_train, y_train = load_dataset_from_mongodb()

    # Concatenate with the previous data
    # x_train = np.concatenate((x_train, x_temp1), axis=0)
    # y_train = np.concatenate((y_train, y_temp1), axis=0)

    # Shuffle the data
    x_train, y_train = shuffle(x_train, y_train, random_state=42)

    # Split the data into train, validation, and test sets
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

    # Preprocess the data
    x_train, y_train_encoded, x_validation, y_validation_encoded = preprocess_data(x_train, y_train, x_validation, y_validation)

    # Configure the model with the best hyperparameters
    best_position = []
    best_position.append(64)
    best_position.append(0.6815258565558058)
    best_position.append(2)
    batch_size = int(best_position[0])
    dropout_rate = best_position[1]
    dropout_period = int(best_position[2])

    # Create the model
    model = create_model(dropout_rate)

    # Train the model with the best hyperparameters
    epochs = 5
    history = train_model(model, x_train, y_train_encoded, x_validation, y_validation_encoded, batch_size, epochs)
    del x_train, y_train, y_train_encoded, x_validation, y_validation_encoded


