# About

This repository covers implementation of various machine learning techniques. 

Implementation is mostly done by tensorflow and its dependencies.

The repository implements concepts in ([Tensorflow's playlist on youtube](https://www.youtube.com/@TensorFlow)) and ([https://learn.udacity.com/courses/ud187](Udacity's course on Introduction to Tensorflow for deep learning.))



# NLP folder

This folder implements basic natural language processing using tensorflow.
 
#### recognize_sentiment_sarcasm.ipynb

Trained a model to predict if a news headline is sarcastic or not.

#### Standardization in sklearn.ipynb

Used StandardScaler, MinMaxScaler to standardize values in sklearn.

#### create_poetry.ipynb

Given an irish poem, created a model using rnn to generate a new poem.

#### Tokenization_udacity.ipynb

Implemented tokenization in Udacity's course.

#### NLP Intro- preparing text to use with Tensorflow models.ipynb

Tokenized sentences. Converted the tokenized sentences to sequences of the same length. Added padding to the end of the sequences.

#### NLP intro- tokenizing and sequencing a real dataset

Tokenized, sequences and padded data from a large dataset of reviews.

#### NLP - Word Embeddings and Sentiment.ipynb

Trained a Basic Sentiment Model with Embeddings.

#### NLP - Tweaking the model.ipynb

Improved the previous sentiment model by tweaking some parameters such as vocab_size, embedding_dimension, and input length.

#### What's in a (sub)word?.ipynb

Created subwords from then trained the model. The model was overfitting.

#### Use LSTMs with the reviews subwords dataset.ipynb

Compared predictions between Embeddings and GlobalAveragePooling layer, Embeddings and a Bidirectional LSTM layer, Embeddings and a multiple Bidirectional LSTM layer.


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

#### time_windows.ipynb

Created a dataset of 10-step windows for training. The last function takes a time series and converts it to a dataset with windows. It takes series, window_size, batch_size and buffer_size as the arguments.

#### forecasting_with_machine_learning.ipynb

Train a regression and dense model to make predictions. Use keras to find the best learning rate. Used early stopping to stop training when the loss does not change for a while.

#### forecasting_with_rnn.ipynb

Trained a RNN model to make predictions. Also used Sequence to sequence forecasting using an RNN model. Data preparation for sequence to sequence forecating is a little different. 

Expand dimensions of a series using tf.expand_dims(series, axis = -1). 
Tried the Adam and SGD optimizers, no major differences was noticed.

#### forecasting_with_stateful_rnn.ipynb

Trained a stateful RNN model. During data preparation, the data is not shuffled. The output from the initial rnn is passed as the state input to the next rnn.

#### forecasting_with_cnn.ipynb

In the first step, created a model with a cnn layer as first layer then had the rest of the layers as LSTM and Dense layers.

I then created a fully connected CNN using wavenet architecture. It performed better than the RNNs. Got the lowest mae.


# Datasets folder

Contains datasets that are used in this repository. 

The flower dataset was too large and is not included in the repository. However, you can install the dataset for ([here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz)). Some datasets are also downloaded through tensorflow_datasets package.

The reviews folder text reviews from various companies. The data can be used to perform sentiment analysis.

The songdata dataset for NLP text generation can be downloaded ([here](https://drive.google.com/uc?id=1LiJFZd41ofrWoBtW-pMYsfz1w8Ny0Bj8))