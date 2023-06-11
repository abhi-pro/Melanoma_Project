# import numpy as np
# import h5py
# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Concatenate, GlobalAveragePooling2D
# from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
# from pymongo import MongoClient
# from sklearn.model_selection import train_test_split
# from sklearn.utils import shuffle
# import database
# import pymongo documents = collection.find()
# NameError: name 'collection' is not defined

# Connect to MongoDB
# client = MongoClient('mongodb://localhost:27017/')
# db = client['your_database_name']
# collection = db['your_collection_name']

# def load_dataset_from_hdf5(file_path):
#     with h5py.File(file_path, 'r') as f:
#         x_train = np.array(f['x_train'])
#         y_train = np.array(f['y_train']).astype(int)
#     return x_train, y_train

# def store_dataset_in_mongodb(x_train, y_train):
#     # Create a list of documents to insert into the MongoDB collection
#     documents = []
#     for i in range(len(x_train)):
#         document = {
#             'image': x_train[i],
#             'label': y_train[i]
#         }
#         documents.append(document)

#     # Insert the documents into the collection
#     collection.insert_many(documents)

# def load_dataset_from_mongodb():
#     # Retrieve the documents from the MongoDB collection
#     documents = collection.find()

#     # Extract the images and labels from the documents
#     images = []
#     labels = []
#     for document in documents:
#         images.append(document['image'])
#         labels.append(document['label'])

#     # Convert the images and labels to Numpy arrays
#     x_train = np.array(images)
#     y_train = np.array(labels)

#     return x_train, y_train

# def preprocess_data(x_train, y_train, x_validation, y_validation):
#     # Shuffle the data
#     x_train, y_train = shuffle(x_train, y_train, random_state=42)

#     # Convert labels to one-hot encoded format
#     y_train_encoded = tf.keras.utils.to_categorical(y_train, num_classes=2)
#     y_validation_encoded = tf.keras.utils.to_categorical(y_validation, num_classes=2)

#     return x_train, y_train_encoded, x_validation, y_validation_encoded

# def create_model(dropout_rate):
#     def fire_module(x, squeeze_filters, expand_filters):
#         squeeze = Conv2D(squeeze_filters, (1, 1), activation='relu')(x)
#         expand_1x1 = Conv2D(expand_filters, (1, 1), activation='relu')(squeeze)
#         expand_3x3 = Conv2D(expand_filters, (3, 3), padding='same', activation='relu')(squeeze)
#         output = Concatenate()([expand_1x1, expand_3x3])
#         return output

#     # Input shape
#     input_shape = (240, 240, 3)

#     # Create the SqueezeNet model
#     input_layer = tf.keras.Input(shape=input_shape)

#     x = Conv2D(64, (3, 3), strides=(2, 2), activation='relu')(input_layer)
#     x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

#     x = fire_module(x, 16, 64)
#     x = fire_module(x, 16, 64)
#     x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

#     x = fire_module(x, 32, 128)
#     x = fire_module(x, 32, 128)
#     x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

#     x = fire_module(x, 48, 192)
#     x = fire_module(x, 48, 192)
#     x = fire_module(x, 64, 256)
#     x = fire_module(x, 64, 256)

#     # Regularization with Dropout
#     x = Dropout(dropout_rate)(x)

#     x = Conv2D(2, (1, 1), activation='relu')(x)

#     x = GlobalAveragePooling2D()(x)
#     output_layer = Dense(2, activation='softmax')(x)

#     model = Model(inputs=input_layer, outputs=output_layer)

#     return model

# def train_model(model, x_train, y_train_encoded, x_validation, y_validation_encoded, batch_size, epochs):
#     # Early Stopping
#     early_stopping = EarlyStopping(patience=5, restore_best_weights=True)

#     # Set the batch size in model.fit
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#     # Learning rate scheduler for adjusting learning rate during training
#     def learning_rate_scheduler(epoch):
#         return 1e-3 * 0.95 ** (epoch + epochs)

#     # Create a learning rate scheduler callback
#     learning_rate = tf.keras.callbacks.LearningRateScheduler(learning_rate_scheduler, verbose=1)

#     class_weights = {0: 0.631, 1: 2.398}

#     history = model.fit(x_train, y_train_encoded, batch_size=batch_size, epochs=epochs, class_weight=class_weights, validation_data=(x_validation, y_validation_encoded), verbose=1, callbacks=[learning_rate, early_stopping])

#     # Save the model
#     model.save("model.h5")

#     return history

# def retrain():
#     # Load your dataset from .hdf5 file
#     x_train, y_train = load_dataset_from_hdf5('dataset_cycleDAG_240x240_HairRemoval.hdf5')
#     x_temp, y_temp = load_dataset_from_hdf5('dataset_cycleDAG_2_240x240_hairremoval_benign.hdf5')

#     # Concatenate with the previous data
#     x_train = np.concatenate((x_train, x_temp), axis=0)
#     y_train = np.concatenate((y_train, y_temp), axis=0)

#     # Store the dataset in MongoDB
#     # store_dataset_in_mongodb(x_train, y_train)

#     # Load the dataset from MongoDB
#     x_temp1, y_temp1 = load_dataset_from_mongodb()

#     # Concatenate with the previous data
#     x_train = np.concatenate((x_train, x_temp1), axis=0)
#     y_train = np.concatenate((y_train, y_temp1), axis=0)

#     # Shuffle the data
#     x_train, y_train = shuffle(x_train, y_train, random_state=42)

#     # Split the data into train, validation, and test sets
#     x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

#     # Preprocess the data
#     x_train, y_train_encoded, x_validation, y_validation_encoded = preprocess_data(x_train, y_train, x_validation, y_validation)

#     # Configure the model with the best hyperparameters
#     best_position = []
#     best_position.append(64)
#     best_position.append(0.6815258565558058)
#     best_position.append(2)
#     batch_size = int(best_position[0])
#     dropout_rate = best_position[1]
#     dropout_period = int(best_position[2])

#     # Create the model
#     model = create_model(dropout_rate)

#     # Train the model with the best hyperparameters
#     epochs = 1
#     history = train_model(model, x_train, y_train_encoded, x_validation, y_validation_encoded, batch_size, epochs)


# retrain()


# def retrain():
#     # Load your dataset
#     x_train, y_train = load_dataset('/kaggle/input/dataset-after-hairremove-240x240/dataset_cycleDAG_240x240_HairRemoval.hdf5')
#     x_temp, y_temp = load_dataset('/kaggle/input/dataset-after-hairremove-240x240/dataset_cycleDAG_2_240x240_hairremoval_benign.hdf5')

#     # Concatenate datasets
#     x_train, y_train = concatenate_datasets(x_train, y_train, x_temp, y_temp)

#     # Split the data into train, validation, and test sets
#     x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

#     # Preprocess the data
#     x_train, y_train_encoded, x_validation, y_validation_encoded = preprocess_data(x_train, y_train, x_validation, y_validation)

#     # Configure the model with the best hyperparameters
#     best_position = []
#     best_position.append(64)
#     best_position.append(0.6815258565558058)
#     best_position.append(2)
#     batch_size = int(best_position[0])
#     dropout_rate = best_position[1]
#     dropout_period = int(best_position[2])

#     # Create the model
#     model = create_model(dropout_rate)

#     # Train the model with the best hyperparameters
#     epochs = 1
#     history = train_model(model, x_train, y_train_encoded, x_validation, y_validation_encoded, batch_size, epochs)


