from django import forms


class UploadImageForm(forms.Form):
    """图像上传表单"""
    #image = forms.ImageField(label='请上传一张图片:', )
    image = forms.ImageField(label='', )