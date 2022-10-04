from numpy import loadtxt
from keras.models import model_from_json
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from keras.preprocessing import image
import numpy as np
from keras.models import load_model
import tensorflow as tf
import os
import seaborn as sns
import matplotlib.pyplot as plt

preds = []

def get_model_arr():
    label_arr = ['Mild', 'Moderate', 'No_DR', 'PDR', 'Severe']
    model_arr = os.listdir("Models")
    #Removing out VGG16 19 old new mobilenet v1 v2 old from the array
    # model_arr.pop(model_arr.index('VGG19.h5'))
    # model_arr.pop(model_arr.index('VGG16.h5'))
    # model_arr.pop(model_arr.index('VGG19_old.h5'))
    # model_arr.pop(model_arr.index('VGG16_old.h5'))
    # model_arr.pop(model_arr.index('MobileNet_v2_old.h5'))
    # model_arr.pop(model_arr.index('MobileNet_v1_old.h5'))
    return label_arr,model_arr

label_arr,model_arr = get_model_arr()

def img_pred(file_path,model_arr):
    prediction_arr = []
    for model in model_arr:
        temp = return_prediction(model, file_path)
        prediction_arr.append(temp)
    label = max_label(prediction_arr)
    return prediction_arr, label


def max_label(pred_arr):
    from collections import Counter
    arr = []
    for i in range(len(pred_arr)):
        arr.append(pred_arr[i][2])
    count = Counter(arr)
    return max((zip(count.values(), count.keys())))


def return_prediction(model, file_path):
    img = image.load_img(file_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    model_name = model
    if "mobilenet" in model:
        img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_batch)
    elif "vgg16" in model:
        img_preprocessed = tf.keras.applications.vgg16.preprocess_input(img_batch)
    elif "vgg19" in model:
        img_preprocessed = tf.keras.applications.vgg19.preprocess_input(img_batch)
    elif "resnet" in model:
        img_preprocessed = tf.keras.applications.resnet.preprocess_input(img_batch)
    elif "densenet" in model:
        img_preprocessed = tf.keras.applications.densenet.preprocess_input(img_batch)
    else:
        img_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(img_batch)

    model = load_model("Models\/" + model)
    predicted_arr = model.predict(img_preprocessed)

    predicted_label = label_arr[np.where(predicted_arr[0] == max(predicted_arr[0]))[0][0]]
    return model_name, predicted_arr, predicted_label
# print(label_arr[img_pred("severe.png")[0][0]])

def del_all(file_name):
    os.remove(file_name)