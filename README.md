Reference: 
https://youtu.be/a99p_fAr6e4?si=fqL4_LTpaLFLoImt 

Platform - Pycharm
Requirements:
mediapipe 0.8.1  
OpenCV 3.4.2 or Later 
Tensorflow 2.3.0 or Later<br>tf-nightly 2.5.0.dev or later (Only when creating a TFLite for an LSTM model) 
scikit-learn 0.23.2 or Later (Only if you want to display the confusion matrix)
matplotlib 3.3.2 or Later (Only if you want to display the confusion matrix) 

Functionalities:
The project offers two functionalities:
Hand Sign Recognition: This feature identifies static hand gestures captured in an image.
Finger Gesture Recognition: This feature recognizes dynamic finger gestures based on historical fingertip coordinates.

File Structure:
The project is organized into the following directories:
1.app.py: This is the main program for inference. It can perform both hand sign and finger gesture recognition based on trained models.
2.keypoint_classification.ipynb: This Jupyter Notebook script trains a model for hand sign recognition using keypoint data.
3.point_history_classification.ipynb: This Jupyter Notebook script trains a model for finger gesture recognition using fingertip coordinate history.
4.model/: This directory stores files related to both hand sign and finger gesture recognition models.
5.keypoint_classifier/: This subdirectory stores files specific to hand sign recognition:
6.training_data (keypoint.csv): This file stores keypoint data for training the hand sign recognition model.
7.trained_model (keypoint_classifier.tflite): This file stores the trained TensorFlow Lite model for hand sign recognition.
8.label_data (keypoint_classifier_label.csv): This file defines the labels associated with the hand sign key points.
9.inference_module (keypoint_classifier.py): This Python script provides functions for hand sign recognition using the trained model.
10.point_history_classifier/: This subdirectory stores files specific to finger gesture recognition:
11.training_data (point_history.csv): This file stores fingertip coordinate history data for training the finger gesture recognition model.
12.trained_model (point_history_classifier.tflite): This file stores the trained TensorFlow Lite model for finger gesture recognition.
13.label_data (point_history_classifier_label.csv): This file defines the labels associated with the finger gestures.
14.inference_module (point_history_classifier.py): This Python script provides functions for finger gesture recognition using the trained model.
15.utils/cvfpscalc.py: This Python module helps calculate the Frames Per Second (FPS) during inference.


Training:
The project allows training models for both hand sign and finger gesture recognition. You can add or modify training data (.csv files) and retrain the models using the provided Jupyter Notebooks (keypoint_classification.ipynb and point_history_classification.ipynb). These notebooks likely contain machine learning code for training the models.


This documentation provides a basic understanding of the project structure and functionalities. 

Error : Libraries might have dependencies on other libraries with specific version requirements. Using incompatible versions of these dependencies can cause conflicts.


