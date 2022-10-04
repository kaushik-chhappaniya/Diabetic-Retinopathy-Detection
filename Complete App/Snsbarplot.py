from keras.preprocessing import image
import numpy as np
from keras.models import load_model
import tensorflow as tf
import matplotlib.pyplot as plt
import model_testing as mt
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import os
import model_testing as mt
from random import randint as ri
import PIL

label_arr,model_arr = mt.get_model_arr()

def plot_graph(predicted_arr,label,model_arr):
    f = plt.figure(figsize=(20,20))
    #fig, axes = plt.subplots(figsize=(15, 15))
    for i in range(len(predicted_arr)):
        Pred_arr = predicted_arr[i][1][0]
        f.add_subplot(4, 2, i + 1)
        plt.subplots_adjust(left=0.125,bottom=0.1,right=0.9,top=0.9,wspace=0.4,hspace=0.9)
        sns.barplot(x=label_arr, y=Pred_arr)
        plt.title(model_arr[i])
    f.suptitle('Predicted Label is {}'.format(label))
    #plt.title(("Predicted Label {0}".format(label)), fontsize=15)  # print name on the grahp expected name
    output_img_name = "Output_"+str(ri(1,1000))+".png"
    plt.savefig(output_img_name)
    Image = PIL.Image.open(r"{}".format(output_img_name))
    Image.show()
    return output_img_name

def display_img(file_name):
    Image = PIL.Image.open(r"{}".format(file_name))
    return Image.show()


