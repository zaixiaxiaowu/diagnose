from django.urls import path
from . import skin, chest, stomach, brain


urlpatterns = [
    path('skin/', skin.index),
    path('chest/', chest.index),
    path('stomach/', stomach.index),
    path('brain/', brain.index),
    path('brain_result/', brain.HttpResponse),
    path('stomach_result/', stomach.HttpResponse),
    path('skin_result/',skin.HttpResponse)

]
