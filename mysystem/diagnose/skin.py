import uuid

import tensorflow as tf
import numpy as np
from django.http import HttpResponse
from tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing
from .form import UploadImageForm
from .models import Image1
from django.shortcuts import render
from PIL import Image
import json
import os


def index(request):
    """图片的上传"""
    if request.method == 'POST':
        form = UploadImageForm(request.POST, request.FILES)
        if form.is_valid():
            picture = Image1(photo=request.FILES['image'])
            picture.save()

            lab, confidence, result = imgdetect(picture)
            return render(request, 'skin_result.html',
                          {'picture': picture, 'label': lab, 'confidence': confidence, 'result': result})

    else:
        form = UploadImageForm()

        return render(request, 'skin.html', {'form': form})


def imgdetect(picture):
    model = load_model("../models/skin_cancer.h5")
    img = Image.open(picture.photo.path)
    img = img.resize((224, 224))
    img = preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    class_names = {0: 'AKIEC(光化性角化病)', 1: 'BCC(基底细胞癌)', 2: 'BKL(脂溢性角化病/太阳慢病毒)', 3: 'DF(瘤病变)', 4: 'MEL(黑色素)',
                   5: 'NV(血管病变)', 6: 'VASC(皮肤纤维化)'}
    d = {
        0: '智能分析：初步诊断结果为光线性角化病（日光性角化病），这是由长期暴晒所引起的癌前增生。随着年龄的增长，人们会更容易患这种皮肤病。它可能进一步发展为鳞状细胞癌，这是一种皮肤癌。光化性角化病的皮肤通常为粉红色、红色，少数情况下为灰色或棕色，它们会让皮肤感觉粗糙和有鳞屑。',
        1: '智能分析：初步诊断结果基底细胞癌，它位于表皮的最底层。虽然基底细胞癌可能并非起源于基底细胞，但这种命名是因为癌细胞在显微镜下看起来像基底细胞。基底细胞癌是最常见的皮肤癌。在有日光暴晒史的肤色白皙的人中更加常见，而在肤色较深的人中较为少见。基底细胞癌一般发生于易曝光部位的皮肤表面，如头颈部。基底细胞癌的治疗几乎始终都是成功的，这种癌症很少会致命。但是大约有 25% 的已治愈患者会在5年内复发。因此，但凡有基底细胞癌既往史的患者均应每年进行一次皮肤检查。',
        2: '智能分析：初步诊断结果脂溢性角化病，是皮肤上的黄褐色、米色、棕色或黑色增生物，看起来像大疣。脂溢性角化病不需要治疗，如有刺激、发痒，或者你不喜欢它们的样子，医生可以通过液氮冷冻与电针的方式将它们去除。',
        3: '智能分析：初步诊断结果瘤病变。',
        4: '智能分析：初步诊断结果黑色素。',
        5: '智能分析：初步诊断结果血管病变。',
        6: '智能分析：初步诊断结果皮肤纤维瘤，它是由胶原纤维沉积而引起的红色至棕色的肿物（结节），胶原纤维是由成纤维细胞合成的蛋白，构成了皮下柔软的组织。一般情况下，皮肤纤维瘤无需治疗，除非体积变化或带来困扰。如果需要，医生可手术切除。'}
    predictions = model.predict(img)
    scores = tf.nn.softmax(predictions[0])
    scores = scores.numpy()

    context = {
        'result': d[np.argmax(scores)],
        'prediction': class_names[np.argmax(scores)],
        'confidence': (100 * np.max(scores)).round(2)
    }

    return class_names[np.argmax(scores)], (100 * np.max(scores)).round(2), d[np.argmax(scores)]
