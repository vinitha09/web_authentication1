from django.urls import path
from . import views
urlpatterns=[

    path('home/',views.index,name='index'),
    #path('counter',views.counter,name='counter'),
    path('register/',views.register,name='register'),
    path('',views.login,name='login.html'),
    path('logout/',views.logout,name='logout.html'),
    path('post/<str:pk>',views.post,name='post.html')
      #path('',views.display_records,name='table.html'),
]
