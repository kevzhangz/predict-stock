from flask import Flask, render_template, request
import app.script as script
import os
# import sys
# import logging

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

# app.logger.addHandler(logging.StreamHandler(sys.stdout))
# app.logger.setLevel(logging.ERROR)