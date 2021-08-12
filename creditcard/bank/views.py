from django.shortcuts import render
from .graphs import get_graph,gender,prior,prediction1 
# Create your views here.
def home(request):
    return render(request,'home.html', {})

def data(request):
    # applicants()
    # uri1 = get_graph()
    gender()
    uri20 = get_graph()
    prior()
    uri30 = get_graph()
    # education()
    # uri4 = get_graph()
    return render(request,'data_analysis.html', {'img20':uri20, 'img30':uri30}) 

def predict(request):
    prediction1()
    uri3 = get_graph()
    # ranking()
    # uri4 = get_graph()
    return render(request, 'prediction_analysis.html', {'img3': uri3})