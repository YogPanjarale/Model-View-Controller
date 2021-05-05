from flask import Flask,jsonify,request
from classifier import predictImage

app = Flask(__name__)

@app.route("/predict-image",methods = ["POST"])
def predict():
    image_data = request.files.get("digit")
    result = predictImage(image_data)
    print(result)
    return jsonify({
        "result":result
    }),200

if __name__== "__main__":
    app.run(debug=True)