from django.urls import path
from .views import predict_network_traffic, preview_arff_dataset, split_arff_dataset

urlpatterns = [
    path('split-dataset/', split_arff_dataset, name='split_dataset'),
    path('preview-dataset/', preview_arff_dataset, name='preview_dataset'),
    path('predict-network/', predict_network_traffic, name='predict_network'),
]
