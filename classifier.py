#Importing all the important models and install them if not installed on your device
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from PIL import Image
import PIL.ImageOps
import os, ssl, time



#Fetching the data
X = np.load('image.npz')['arr_0']
y = pd.read_csv("labels.csv")["labels"]
print(pd.Series(y).value_counts())
classes = ['A', 'B', 'C', 'D', 'E','F', 'G', 'H', 'I', 'J', "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
nclasses = len(classes)


#Splitting the data and scaling it
X_train, X_test, Y_train, Y_test = train_test_split(X, y, random_state=9, train_size=3500, test_size=500)
#scaling the features
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

#Fitting the training data into the model
LR = LogisticRegression(solver='saga', multi_class='multinomial').fit(X_train_scaled, Y_train)
print(len(X_train_scaled[0]),len(Y_train))
#Calculating the accuracy of the model
Y_pred = LR.predict(X_test_scaled)
accuracy = accuracy_score(Y_test, Y_pred)
print("The accuracy is : ",round(accuracy*100,2),"%")


def predictImage(image):
    opened_image = Image.open(image)
    #greying it
    greyedout_image=opened_image.convert('L')
    #resizing it
    resized_image = greyedout_image.resize((22,30),Image.ANTIALIAS)
    #pixel filter
    pixel_filter = 20
    #min/max pixels
    min_pixels,max_pixels = np.percentile(resized_image,pixel_filter),np.max(resized_image)
    #scaling it
    rescaled_image = np.clip(resized_image-min_pixels,0,255)
    #inverting it
    inverted_image = np.asarray(rescaled_image)/max_pixels
    #sampling
    test_sample = np.array(inverted_image).reshape(1,660)

    test_predict =LR.predict(test_sample)
    print("Predicted alphabet is "+test_predict[0])
    return test_predict[0]