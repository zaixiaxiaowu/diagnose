import tensorflow as tf
import numpy as np
from django.http import HttpResponse
from tensorflow.keras.models import load_model
from .form import UploadImageForm
from .models import Image1
from django.shortcuts import render
from PIL import Image
from tensorflow.keras import backend as K
import cv2
import matplotlib.pyplot as plt

# coding:utf-8
cpu = tf.config.list_physical_devices("CPU")
tf.config.set_visible_devices(cpu)


def dice_coef(y_true, y_pred, smooth=100):
    y_true_flatten = K.flatten(y_true)
    y_pred_flatten = K.flatten(y_pred)

    intersection = K.sum(y_true_flatten * y_pred_flatten)
    union = K.sum(y_true_flatten) + K.sum(y_pred_flatten)
    return (2 * intersection + smooth) / (union + smooth)


# function to create dice loss
def dice_loss(y_true, y_pred, smooth=100):
    return -dice_coef(y_true, y_pred, smooth)


# function to create iou coefficient
def iou_coef(y_true, y_pred, smooth=100):
    intersection = K.sum(y_true * y_pred)
    sum = K.sum(y_true + y_pred)
    iou = (intersection + smooth) / (sum - intersection + smooth)
    return iou


def index(request):
    """图片的上传"""
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            picture = Image1(photo=request.FILES['image'])
            picture.save()

            m, n = imgdetect(picture)
            plt.figure(figsize=(12, 12))
            plt.subplot(1, 2, 1)
            plt.imshow(m)
            plt.axis('off')
            plt.title('Original Image')

            plt.subplot(1, 2, 2)
            plt.imshow(n)
            plt.title('prediction')
            plt.axis('off')
            plt.savefig('../media/brain_test1.jpg')
            imgpath = "../media/brain_test1.jpg"
            img_data = open(imgpath, 'rb').read()
            return HttpResponse(img_data, content_type="image/png")
    else:
        form = UploadImageForm()

    return render(request, 'brain.html', {'form': form})


def imgdetect(picture):
    model = load_model("../models/brain.hdf5",
                       custom_objects={'dice_loss': dice_loss, "dice_coef": dice_coef, "iou_coef": iou_coef})
    img = Image.open(picture.photo.path)
    img = img.resize((256, 256))
    img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = img / 255
    img = img[np.newaxis, :, :, :]
    predicted_img = model.predict(img)

    return np.squeeze(img), np.squeeze(predicted_img > 0.5)
