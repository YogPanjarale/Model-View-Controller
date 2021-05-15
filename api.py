from flask import Flask,jsonify,request
from my_classifier2 import predictImage
import socket
app = Flask(__name__)

@app.route("/predict-image",methods = ["POST"])
def predict():
    image_data = request.files.get("digit")
    print(request)
    result = predictImage(image_data)
    print(result)
    return jsonify({
        "result":result
    })
@app.route("/hello")
def hello():
    return "Hello"
if __name__== "__main__":
    hostname = socket.gethostname()
    IPAddr = socket.gethostbyname(hostname)
    app.run(host=IPAddr,port=5000)