#import object_detection_api
import os
from PIL import Image
from flask import Flask, request, Response, jsonify
import cv2
import numpy as np
import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

app = Flask(__name__)

# for CORS
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST') # Put any other methods you need here
    return response


@app.route('/')
def index():
    return Response('Web-RTC server for computer vision')


@app.route('/local')
def local():
    return Response(open('./static/local.html').read(), mimetype="text/html")

def load_model():
    model = Sequential()
    model.add(Conv2D(32, (5,5), padding='same', activation='relu',input_shape=(200, 200, 3)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (5,5), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(128, (5,5), padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(2, activation='softmax'))
    
    model.load_weights('trained_model_checkpoint.pkl') #file not included in this repo. Put your trained checkpoint file here
    
    model.compile(Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model


@app.route('/image', methods=['GET','POST'])
def image():
    try:
        image_file = request.files['image']  # get the image
        image_object = Image.open(image_file) # fetch image using PIL Image
        file_path = 'frames/'+str(datetime.datetime.now())+'.jpg' 
        image_object.save(file_path) # save frame by frame 
        image_arr = np.array(image_object) # convert frame into array 
        image_arr = cv2.resize(image_arr, (200, 200))
        input_image = np.expand_dims(image_arr,axis=0)
        model = load_model() # load ML model - above model is just a sample. load your own model and checkpoint in def model()
        prediction = model.predict(input_image) # model prediction
        print(prediction) # print prediction on console
        output = {'object':'return something'} #currently we dont have anything to return
        data = jsonify(output)
        return data


    except Exception as e:
        print('POST /image error: %e' % e)
        return e


if __name__ == '__main__':
	# without SSL
    app.run(debug=True, host='127.0.0.1', port=5050)

	# with SSL
    #app.run(debug=True, host='127.0.0.1', port=5050, ssl_context=('ssl/server.crt', 'ssl/server.key'))
