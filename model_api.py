# app.py
from flask import Flask,jsonify
from flask_restful import Api, Resource, reqparse
import numpy as np
from LangPred import Predictor
from CheckFileUnicode import checkFileUnicode

from sklearn import metrics
import os
import warnings
from pathlib import Path
from os import path
import shutil
import json

# start flask
app = Flask(__name__)

def predict(file):
    # lang in str
    myfile = open(file, encoding='utf-8', mode='r').read()
    lang = predictor.language(myfile)
    return lang

# get the data for the requested query
@app.route('/predictonfile/')
def predictonfile():
    # ==== Prediction ====
    predDictionary = {}
    for root, dirs, files in os.walk(new_data_dir):
        i = 0
        for file in files:
            i += 1
            print ("Entry:", file)
            # if i > 10:
                # break
            pred = predict(os.path.join(root, file))
            predDictionary[str(file)] = pred         
            print ("Path:", str(file), "pred:", pred)
    return jsonify(predDictionary)

# get the data for the requested query
@app.route('/train/<mode>')
def train(mode):
    if mode == 'clean' : 
        [f.unlink() for f in Path(os.path.join(os.getcwd(),config['model_directory'])).glob("*") if f.is_file()] 
        accuracy = predictor.learn(train_data_dir)
    else : 
        accuracy = predictor.learnPartial(new_train_data_dir)
    return jsonify({"accuracy" : str(accuracy)})

# get the data for the requested query - train and predict
@app.route('/checkunicode/<mode>')
def checkunicode(mode):
    if mode == 'predict' : 
        checkFileUnicode(new_data_dir)
    elif mode == 'train':
        checkFileUnicode(train_data_dir)
    return jsonify({"Success":"1"})

if __name__ == '__main__':
    #: supported languages with associated extensions
    with open('config/config.json') as f:
        config = json.load(f)
    train_data_dir = os.path.join(os.getcwd(),config['train_data_directory'])
    new_train_data_dir = os.path.join(os.getcwd(),config['new_train_data_directory'])
    model_directory = os.path.join(os.getcwd(),config['model_directory'],config['checkpoint_filename']) 
    new_data_dir = os.path.join(os.getcwd(),config['new_data_directory'])
    predictor = Predictor(model_dir=model_directory)
    app.run(debug=True, port='1080')