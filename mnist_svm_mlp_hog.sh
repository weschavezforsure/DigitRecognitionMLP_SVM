#!/bin/bash

# mnist_svm_mlp_hog.sh
# - Wesley Chavez 4/26/17
#
# Runs classification on a sweep of models, features, 
# and preprocessing techniques and appends 'Accuracy.txt'
# with the results.
#
# Usage:
# bash mnist_svm_mlp_hog.sh

python DigitRecognitionMLP_SVM.py linearSVM pixels nonorm nodeskew >> Accuracy.txt
python DigitRecognitionMLP_SVM.py linearSVM pixels nonorm deskew >> Accuracy.txt
python DigitRecognitionMLP_SVM.py linearSVM pixels norm nodeskew >> Accuracy.txt
python DigitRecognitionMLP_SVM.py linearSVM pixels norm deskew >> Accuracy.txt
python DigitRecognitionMLP_SVM.py linearSVM HoG nonorm nodeskew >> Accuracy.txt
python DigitRecognitionMLP_SVM.py linearSVM HoG nonorm deskew >> Accuracy.txt
python DigitRecognitionMLP_SVM.py linearSVM HoG norm nodeskew >> Accuracy.txt
python DigitRecognitionMLP_SVM.py linearSVM HoG norm deskew >> Accuracy.txt
python DigitRecognitionMLP_SVM.py linearSVM sumofpixels nonorm nodeskew >> Accuracy.txt
python DigitRecognitionMLP_SVM.py linearSVM sumofpixels nonorm deskew >> Accuracy.txt
python DigitRecognitionMLP_SVM.py linearSVM sumofpixels norm nodeskew >> Accuracy.txt
python DigitRecognitionMLP_SVM.py linearSVM sumofpixels norm deskew >> Accuracy.txt
python DigitRecognitionMLP_SVM.py nonlinearSVM pixels nonorm nodeskew >> Accuracy.txt
python DigitRecognitionMLP_SVM.py nonlinearSVM pixels nonorm deskew >> Accuracy.txt
python DigitRecognitionMLP_SVM.py nonlinearSVM pixels norm nodeskew >> Accuracy.txt
python DigitRecognitionMLP_SVM.py nonlinearSVM pixels norm deskew >> Accuracy.txt
python DigitRecognitionMLP_SVM.py nonlinearSVM HoG nonorm nodeskew >> Accuracy.txt
python DigitRecognitionMLP_SVM.py nonlinearSVM HoG nonorm deskew >> Accuracy.txt
python DigitRecognitionMLP_SVM.py nonlinearSVM HoG norm nodeskew >> Accuracy.txt
python DigitRecognitionMLP_SVM.py nonlinearSVM HoG norm deskew >> Accuracy.txt
python DigitRecognitionMLP_SVM.py nonlinearSVM sumofpixels nonorm nodeskew >> Accuracy.txt
python DigitRecognitionMLP_SVM.py nonlinearSVM sumofpixels nonorm deskew >> Accuracy.txt
python DigitRecognitionMLP_SVM.py nonlinearSVM sumofpixels norm nodeskew >> Accuracy.txt
python DigitRecognitionMLP_SVM.py nonlinearSVM sumofpixels norm deskew >> Accuracy.txt
python DigitRecognitionMLP_SVM.py MLP pixels nonorm nodeskew >> Accuracy.txt
python DigitRecognitionMLP_SVM.py MLP pixels nonorm deskew >> Accuracy.txt
python DigitRecognitionMLP_SVM.py MLP pixels norm nodeskew >> Accuracy.txt
python DigitRecognitionMLP_SVM.py MLP pixels norm deskew >> Accuracy.txt
python DigitRecognitionMLP_SVM.py MLP HoG nonorm nodeskew >> Accuracy.txt
python DigitRecognitionMLP_SVM.py MLP HoG nonorm deskew >> Accuracy.txt
python DigitRecognitionMLP_SVM.py MLP HoG norm nodeskew >> Accuracy.txt
python DigitRecognitionMLP_SVM.py MLP HoG norm deskew >> Accuracy.txt
python DigitRecognitionMLP_SVM.py MLP sumofpixels nonorm nodeskew >> Accuracy.txt
python DigitRecognitionMLP_SVM.py MLP sumofpixels nonorm deskew >> Accuracy.txt
python DigitRecognitionMLP_SVM.py MLP sumofpixels norm nodeskew >> Accuracy.txt
python DigitRecognitionMLP_SVM.py MLP sumofpixels norm deskew >> Accuracy.txt
