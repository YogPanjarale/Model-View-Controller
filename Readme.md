# Project 125 - Model View Controller

- Alphabet-recognition image classifier over flask api
upload image to api and get predictions

## Accuracy ~94.4%

### How to run

- `pip install -r requirements.txt`
- `python api.py`

### How to Test

- Send `POST` request to `http://localhost:5000/predict-image` with body as multi-part form with key `alphabet` as a image file of the number you want to predict
