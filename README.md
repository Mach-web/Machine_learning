# About

This repository covers implementation of various machine learning techniques. 

Implementation is mostly done by tensorflow and its dependencies.

The repository implements concepts in Tensorflow's playlist on youtube and Udacity's course on Introduction to Tensorflow for deep learning.



# NLP folder

This folder implements basic natural language processing using tensorflow.
 
#### recognize_sentiment_sarcasm.ipynb

Trained a model to predict if a news headline is sarcastic or not.

#### Standardization in sklearn.ipynb

Used StandardScaler, MinMaxScaler to standardize values in sklearn.

#### create_poetry.ipynb

Given an irish poem, created a model using rnn to generate a new poem.



# Image_recognition folder

This folder implements image classification using tensorflow.

#### Basic_computer_vision.ipynb

Implements clothing image classification without image augmentation.

#### code_to_graph.ipynb

Tensorflow 2.0

#### Copy of l06c02_exercise_flowers_with_transfer_learning

Lab exercise on classifying flower images using transfer learning.

#### Copy_of_l05c03_exercise_flowers_with_data_augmentation.ipynb

Lab exercise on flower image classification after image augmentation but without transfer learning.

#### degrees_to_fahrenheit.ipynb

Simple neural network with one layer that trains to convert degrees to fahrenheit.

#### dogs_vs_cats_with_augmentation.ipynb

Practice on image classification with image augmentation without transfer learning.

#### dogs_vs_cats_without_augmentation.ipynb

Practice on image classification without image augmentation.
val_accuracy = 0.7560

#### Image_classification_with_CNN.ipynb

Clothing image classification with CNN and MaxPooling. No image augmentation.
val_accuracy = 0.91

#### Transfer_learning_with_tensorflow.ipynb

Used transfer learning to identify dogs and cats.
val_accuracy = 0.98


# Time_series_forecasting

#### common_patterns.ipynb

Created graphs which visualize trends, seasonality and noise.

#### naive_forecasting.ipynb

Used naive forecasting to predict future values. In naive forecasting, the immediate previous value is taken as the prediction.

#### moving_average.ipynb

Use the average of the previous 30 days to make current prediction. Remove trends and seasonality then use the moving_average to make predictions.



# Datasets folder

Contains datasets that are used in this repository. 

The flower dataset was too large and is not included in the repository. However, you can install the dataset for ([here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz)). Some datasets are also downloaded through tensorflow_datasets package.

