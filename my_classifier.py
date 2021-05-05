import ssl,time,os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
from PIL import Image
import PIL.ImageOps

init_start_time = time.time()
# making secure connection
if (not os.environ.get('PYTHONHTTPSVERIFY','') and getattr(ssl,'_create_unverified_context',None)):
    ssl._create_delault_https_context = ssl._create_unverified_context
    print("Creating SSL Context")
'''
fetching the dataset of images of handwritten digits from the OpenML datasets.
The X here would be the data of images represented in Binary, while y would be the label of that image, i.e - 0, 1, 2, ..., 9.
'''
print("Fetching OpemMl Data...")
fstart = time.time()
X,Y= fetch_openml('mnist_784',version=1,return_X_y = True,data_home=os.getcwd()+"\\data")
print("--- Took %s seconds ---" % (time.time() - fstart))
#tells us the count of samples for each of the labels.
print('Counting Samples...')
sstart = time.time()
print(pd.Series(Y).value_counts())
classes=['0','1','2','3','4','5','6','7','8','9']
nclasses=len(classes)
print("--- Took %s seconds ---" % (time.time() - sstart))

'''
Testing and spliting the data 
'''
print("Spliting the Data...")
tsstart = time.time()
Xtr,Xts,Ytr,Yts = train_test_split(X,Y,random_state=23,train_size=7500,test_size=2500)
#scale the features
XtrScale = Xtr/255.00
XtsScale = Xts/255.00
print("--- Took %s seconds ---" % (time.time() - tsstart))

print("Traning the Model...")
trstart = time.time()
#Logistics regression
LR = LogisticRegression(solver= 'saga',multi_class = 'multinomial').fit(XtrScale,Ytr)
print("--- Took %s seconds ---" % (time.time() - trstart))

'''
Checking the accuracy
'''
print("Checking the accuracy...")
Yprediction =LR.predict(XtsScale)
accuracy = accuracy_score(Yts,Yprediction)
print(f"Accuracy : {round(accuracy*100,2)}%")


def predictImage(image):
    opened_image = Image.open(image)
    #greying it
    greyedout_image=opened_image.convert('L')
    #resizing it
    resized_image = greyedout_image.resize((28,28),Image.ANTIALIAS)
    #pixel filter
    pixel_filter = 20
    #min/max pixels
    min_pixels,max_pixels = np.percentile(resized_image,pixel_filter),np.max(resized_image)
    #scaling it
    rescaled_image = np.clip(resized_image-min_pixels,0,255)
    #inverting it
    inverted_image = np.asarray(rescaled_image)/max_pixels
    #sampling
    test_sample = np.array(inverted_image).reshape(1,784)

    test_predict =LR.predict(test_sample)
    return test_predict[0]
