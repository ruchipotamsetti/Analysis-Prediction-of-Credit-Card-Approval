from django.conf.urls import url
from bank import views

urlpatterns=[
    url('home',views.home,name='home'),
    url('data', views.data, name='data'),
    url('predict', views.predict, name='predict'),
]