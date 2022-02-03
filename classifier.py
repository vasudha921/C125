from ensurepip import version
from tkinter import N
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps

X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X_train,  X_test, y_test, y_train = train_test_split(X, y, random_state = 9, train_size = 7500, test_size = 2500)

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

clf= LogisticRegression(solver = 'saga', multi_class = 'monomial').fit(X_train_scaled, y_train)

def getPrediction(image):

    im_PIL = Image.open(image)
    image_bw = im_PIL.convert('L')
    image_bw_resized = image_bw.resize((28,28), Image.ANTIALIAS)
    pixelfilter = 20
    minpixel = np.percentile(image_bw_resized), pixelfilter
    image_bw_resized_inverted_scale = np.clip(image_bw_resized - minpixel,0 ,255)
    maxpixel = np.max(image_bw_resized)
    image_bw_resized_inverted_scale = np.asarray(image_bw_resized_inverted_scale)/maxpixel
    testsample = np.array(image_bw_resized_inverted_scale).reshape(1,784)
    test_pred = clf.predict(testsample)
    return(test_pred[0])