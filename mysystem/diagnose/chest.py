import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from .form import UploadImageForm
from .models import Image1
from django.shortcuts import render

cpu = tf.config.list_physical_devices("CPU")
tf.config.set_visible_devices(cpu)


def index(request):
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            picture = Image1(photo=request.FILES['image'])
            picture.save()

            lab, confidence, result = imgdetect(picture)
            return render(request, 'chest_result.html',
                          {'picture': picture, 'label': lab, 'confidence': confidence, 'result': result})

    else:
        form = UploadImageForm()
        return render(request, 'chest.html', {'form': form})


def imgdetect(picture):
    model = load_model("../models/chest.h5", )
    img = tf.keras.utils.load_img(picture.photo.path, target_size=(256, 256))
    img = tf.keras.utils.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    Result = model.predict(img)
    label = ["Nomal(正常)", "Pneumonia(肺炎)"]
    result = ["初步诊断结果为正常，无明显异常。", "初步诊断结果为肺炎，肺部颜色变白，可能患有急性肺炎；肺纹理变粗、呈网状或条索状、斑点状阴影，可能有老慢支、肺气肿的风险。"]
    if (Result[0][0] > Result[0][1]):
        return label[0], Result[0][0], result[0]
    else:
        return label[1], Result[0][1], result[1]
