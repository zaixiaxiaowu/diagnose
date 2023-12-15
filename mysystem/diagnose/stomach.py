import tensorflow as tf
import numpy as np
from django.http import HttpResponse
from tensorflow.keras.models import load_model
from .form import UploadImageForm
from .models import Image1
from django.shortcuts import render
from skimage.io import imread
from skimage.transform import resize
import matplotlib.pyplot as plt
from PIL import Image

cpu = tf.config.list_physical_devices("CPU")
tf.config.set_visible_devices(cpu)


def index(request):
    """图片的上传"""
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            picture = Image1(photo=request.FILES['image'])
            picture.save()

            n = imgdetect(picture)
            image = np.array(Image.open(picture.photo.path))
            plt.figure(figsize=(12, 12))
            plt.subplot(1, 2, 1)
            plt.imshow(image)
            plt.axis('off')
            plt.title('Original Image')

            plt.subplot(1, 2, 2)
            plt.imshow(n)
            plt.title('prediction')
            plt.axis('off')
            plt.savefig('../media/stomach_test1.jpg')
            imgpath = "../media/stomach_test1.jpg"
            img_data = open(imgpath, 'rb').read()
            return HttpResponse(img_data, content_type="image/png")

    else:
        form = UploadImageForm()

        return render(request, 'stomach.html', {'form': form})


def imgdetect(picture):
    model = load_model("../models/stomach.h5")
    img = imread(picture.photo.path)[:, :, :3]
    img = resize(img, (256, 256), mode='constant', preserve_range=True)
    predMask = model.predict(np.expand_dims(img, axis=0), verbose=0)

    return np.squeeze(predMask)
