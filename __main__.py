from flask import Flask, render_template, request
import script
import os

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict')
def predict():
    stock = request.args.get('stock')
    predict = script.predictStock(stock)
    return render_template('result.html', predict=predict)
    

@app.route('/latest')
def getLatest():
    return render_template('result.html')



if __name__ == "__main__":
    app.run(host='0.0.0.0')
