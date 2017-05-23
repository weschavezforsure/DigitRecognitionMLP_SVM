#!/usr/bin/env python

'''
SVM (linear and nonlinear) and MLP handwritten digit recognition from a subset of MNIST ('./digits.png').
Input features are either the pixels themselves or HoG features.
User may choose to 'deskew' the inputs using moments.
User may also normalize the inputs (by subtracting the mean and dividing by the standard deviation).

For HoG, we transform histograms to space with Hellinger metric (see [1] (RootSIFT))

[1] R. Arandjelovic, A. Zisserman
    "Three things everyone should know to improve object retrieval"
    http://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf

Example Usage:
    python digits.py linearSVM pixels nonorm nodeskew
    python digits.py nonlinearSVM HoG norm deskew
    python digits.py MLP sumofpixels norm deskew

Developed from opencv-master/samples/python/digits.py and 
               opencv-master/samples/python/letter_recog.py

-Wesley Chavez
4/24/17

'''
# Python 2/3 compatibility
from __future__ import print_function

import cv2
from sys import argv

# built-in modules
from multiprocessing.pool import ThreadPool

import numpy as np
from numpy.linalg import norm

# local modules
from common import clock, mosaic

SZ = 20 # size of each digit is SZ x SZ
CLASS_N = 10
DIGITS_FN = 'digits.png'

# Splits 'digits.png' into 5000 separate images
def split2d(img, cell_size, flatten=True):
    h, w = img.shape[:2]
    sx, sy = cell_size
    cells = [np.hsplit(row, w//sx) for row in np.vsplit(img, h//sy)]
    cells = np.array(cells)
    if flatten:
        cells = cells.reshape(-1, sy, sx)
    return cells

# Loads 'digits.png' and corresponding labels
def load_digits(fn):
    print('loading "%s" ...' % fn)
    digits_img = cv2.imread(fn, 0)
    digits = split2d(digits_img, (SZ, SZ))
    labels = np.repeat(np.arange(CLASS_N), len(digits)/CLASS_N)
    return digits, labels

# Deskews an image using second order moments 
def deskew(img):
    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11']/m['mu02']
    M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img

# Reshapes labels to 4500 x 10 one-hot vectors
class StatModel(object):
    def unroll_responses(self, responses):
        sample_n = len(responses)
        new_responses = np.zeros(sample_n*CLASS_N, np.int32)
        resp_idx = np.int32( responses + np.arange(sample_n)*CLASS_N)
        new_responses[resp_idx] = 1
        return new_responses

# SVM classifier with train and predict functions, linear or nonlinear depending on input args
class SVM(StatModel):
    def __init__(self, C = 1, gamma = 0.5, linearity = 'linear'):
        self.model = cv2.ml.SVM_create()
        self.model.setGamma(gamma)
        self.model.setC(C)
        if (linearity == 'linear'):
            self.model.setKernel(cv2.ml.SVM_LINEAR)
        elif (linearity == 'nonlinear'):
            self.model.setKernel(cv2.ml.SVM_RBF)
        self.model.setType(cv2.ml.SVM_C_SVC)

    def train(self, samples, responses):
        self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)

    def predict(self, samples):
        return self.model.predict(samples)[1].ravel()

# MLP classifier, with different train and predict functions
class MLP(StatModel):
    def __init__(self):
        self.model = cv2.ml.ANN_MLP_create()

    def train(self, samples, responses):
        # In the case of sumofpixels
        if (samples.shape == (4500,)):
            sample_n = len(samples)
            new_responses = self.unroll_responses(responses).reshape(-1, CLASS_N)
            layer_sizes = np.int32([1, 100, 100, CLASS_N])
        # In the case of HoG and pixels
        else:
            sample_n, var_n = samples.shape
            new_responses = self.unroll_responses(responses).reshape(-1, CLASS_N)
            layer_sizes = np.int32([var_n, 100, 100, CLASS_N])
        self.model.setLayerSizes(layer_sizes)
        self.model.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP)
        self.model.setBackpropMomentumScale(0.0)
        self.model.setBackpropWeightScale(0.001)
        self.model.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 20, 0.01))
        self.model.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM, 2, 1)

        self.model.train(samples, cv2.ml.ROW_SAMPLE, np.float32(new_responses))

    def predict(self, samples):
        ret, resp = self.model.predict(samples)
        return resp.argmax(-1)


# Evaluates a trained model.  Predicts, calculates accuracy and a confusion matrix, and returns mosaic
# of test images, red images for the incorrectly predicted test images.
def evaluate_model(model, digits, samples, labels):
    resp = model.predict(samples)
    resp = resp.astype(np.int32)

    err = (labels != resp).mean()
    print('Accuracy: %.2f %%' % ((1-err)*100))

    confusion = np.zeros((10, 10), np.int32)
    for i, j in zip(labels, resp):
        confusion[i, j] += 1
    print('confusion matrix:')
    print(confusion)
    print()

    vis = []
    for img, flag in zip(digits, resp == labels):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if not flag:
            img[...,:2] = 0
        vis.append(img)
    return mosaic(25, vis)

# Preprocesses pixels with or without normalization (subtract mean and divide by standard deviation
# of the training set)
def preprocess_pixels(digits,normbool):
    if (normbool == 'nonorm'):
        return np.float32(digits).reshape(-1, SZ*SZ) / 255.0
    elif (normbool == 'norm'):
        ret = np.float32(digits).reshape(-1, SZ*SZ)
        ret = ret - np.mean(ret)
        ret = ret / np.std(ret)
        return ret
    else:
        print ('Please choose \'norm\' or \'nonorm\' for normbool')
        exit()
        
# Calculates HoG for each image
def preprocess_hog(digits):
    samples = []
    for img in digits:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
        mag, ang = cv2.cartToPolar(gx, gy)
        bin_n = 16
        bin = np.int32(bin_n*ang/(2*np.pi))
        bin_cells = bin[:10,:10], bin[10:,:10], bin[:10,10:], bin[10:,10:]
        mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
        hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)
        # transform to Hellinger kernel
        eps = 1e-7
        hist /= hist.sum() + eps
        hist = np.sqrt(hist)
        hist /= norm(hist) + eps

        samples.append(hist)
    return np.float32(samples)


if __name__ == '__main__':

    # Arguments from command line
    script, modeltype, feats, normbool, deskewbool = argv

    # Load digits and labels
    digits, labels = load_digits(DIGITS_FN)
    print('preprocessing...')

    # shuffle digits
    rand = np.random.RandomState(321)
    shuffle = rand.permutation(len(digits))
    digits, labels = digits[shuffle], labels[shuffle]

    # Preprocessing based on deskewbool and feats
    if (deskewbool == 'deskew'):
        digits = list(map(deskew, digits))
    if (feats == 'HoG'):
        samples = preprocess_hog(digits)
    elif (feats == 'pixels'):
        samples = preprocess_pixels(digits,normbool)
    elif (feats == 'sumofpixels'):
        digits1 = np.asarray(digits)
        digits1 = digits1.reshape((digits1.shape[0], -1), order='F')
        samples = np.float32(np.mean(digits1,axis=1))
    else:
        print ('Please choose \'HoG\' or \'pixels\' or \'sumofpixels\' for feats')
        exit()

    # Train with 90% of dataset
    train_n = int(0.9*len(samples))

    # Display the test set
    cv2.imshow('test set', mosaic(25, digits[train_n:]))

    # Split dataset into training and testing
    digits_train, digits_test = np.split(digits, [train_n])
    samples_train, samples_test = np.split(samples, [train_n])
    labels_train, labels_test = np.split(labels, [train_n])

    # Chooses model based on modeltype arg
    if (modeltype == 'linearSVM'):
        print('training linear SVM...')
        model = SVM(C=2.67, gamma=5.383, linearity='linear')
    elif (modeltype == 'nonlinearSVM'):
        print('training nonlinear SVM...')
        model = SVM(C=2.67, gamma=5.383, linearity='nonlinear')
    elif (modeltype == 'MLP'):
        print('training MLP...')
        model = MLP()
    else:
        print ('Please choose \'linearSVM\' or \'nonlinearSVM\' or \'MLP\' for modeltype')
        exit()

    # Train and evaluate model
    model.train(samples_train, labels_train)
    vis = evaluate_model(model, digits_test, samples_test, labels_test)

    # Display correct and incorrect test images (Correct in white, incorrect in red)
    cv2.imshow('Test', vis)

    cv2.waitKey(0)
