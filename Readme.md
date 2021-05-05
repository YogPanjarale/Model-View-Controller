# Class 125 - Model View Controller

- digit recognition image classifier over flask api
upload image to api and get predictions

## Accuracy ~91%

### How to run

- `pip install -r requirements.txt`
- `python api.py`

### How to Test

- Send `POST` request to `http://localhost:5000/predict-image` with body as multi-part form with key `digit` as a image file of the number you want to predict
