from django.urls import path
from .views import search_papers

urlpatterns = [
    path('search/', search_papers, name='search_papers'),
]