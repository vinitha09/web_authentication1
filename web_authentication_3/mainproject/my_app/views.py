from django.shortcuts import render

# Create your views here.
from django.shortcuts import render,redirect
from django.http import HttpResponse
from django.contrib.auth.models import User,auth
from django.contrib import messages
from django.contrib.auth import authenticate, login as auth_login
import numpy as np
from keras.models import load_model
from django.http import JsonResponse
from cnn_model import *
from dataset import *
#from .models import Student



# def display_records(request):
#     all_records = Student.objects.all()
#     return render(request, 'table.html', {'all_records': all_records})


# from .models import Feature
# # Create your views here.
def index(request):
    #features=Feature.objects.all()
    return render(request,'index.html')
# # def counter(request):
# #     text=request.POST['text']
# #     no_of_words=len(text.split())
# #     return render(request,'counter.html',{'amount':no_of_words})



            
def login(request):
    if request.method == 'POST':
        username = request.POST['username']
        gender=request.POST['gender']
        password = request.POST['password']
        # Check if the request contains image data
       
        obj=Login_fun(username,gender)
        result=obj.predicting_live_image()
        # If the request does not contain image data, proceed with login form submission
        
        
        user = authenticate(username=username, password=password)
        
        if user is not None:
            if not result:
                auth_login(request, user)
                return redirect('/home')
            else:
                messages.info(request, 'Invalid facial detection')
                return redirect('')

        else:
            messages.info(request, 'Invalid Username or Password')
            return redirect('login.html')
    else:
        return render(request, 'login.html')

    
def register(request):

    if request.method=='POST':
        username=request.POST['username']
        gender=request.POST['gender']
     
        obj_cnn=CNN_model(username,gender)
        obj_cnn.dataset_initializer()
        obj_cnn.model_initializer()
        email=request.POST['email']
        password=request.POST['password']
        password2=request.POST['password2']
        if password==password2:
            if User.objects.filter(email=email).exists():
                messages.info(request,'Email Already Used')
                return redirect('register')
            elif User.objects.filter(username=username).exists():
                messages.info(request,'Username Already Used')
                return redirect('register')
            else:
                user=User.objects.create_user(username=username,email=email,password=password)
                user.save()
                return redirect('login.html')
        else:
            messages.info(request,'Password do not match')
            return redirect('register')
    else:
        return render(request,'register.html')
    

    
def logout(request):
    auth.logout(request)
    return redirect('/')

def post(request,pk):
    return render(request,'post.html',{'pk':pk})
