# Create your views here.
from django.shortcuts import render

def IndexView(request):
    request.encoding = 'utf-8'
    return render(request, "index.html")