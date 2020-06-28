import flask
from flask import Flask
import os
from flask import Flask, jsonify, render_template, request, url_for
import _pickle as pickle
from wtforms import Form, TextAreaField, validators
from nltk.corpus import stopwords
import json
import string, re
import nltk

with open("model/model_predict_airbnb_review.pkl", "rb") as file:
    sen_model = pickle.load(file)

class ReviewForm(Form):
	message = TextAreaField('',
			[validators.DataRequired(), validators.length(min=4)])

def preprocess(document):

    punctuations = string.punctuation
    stops = list(stopwords.words('english'))

    #Remove all characters except A-Z
    alphas_only = re.sub("[^a-zA-Z]", r" ", str(document))
    #Normalize all charachters to lowercase and split them
    words = alphas_only.lower().split() 
    #Remove all character where is not in our stopwords
    myword = [w for w in words if w not in stops and w not in punctuations]

    return myword

def create_app():
    app_name = 'flask'
    print('app_name = {}'.format(app_name))

    # create app
    app = Flask(__name__, instance_relative_config=True)

    @app.route('/')
    def index():
        return render_template('sentiment.html')

    @app.route('/sentiment', methods = ['POST'])
    def sentiment():
        form = ReviewForm(request.form)
        if request.method == 'POST' and form.validate():
            review = request.form['message']
            a = preprocess(review)
            pred = sen_model.predict(a)[0]
            proba = sen_model.predict_proba(a)[0, pred]
            return render_template('sentiment.html', prediction=pred, proba = round(proba*100, 2), content=review)
        check = "Please fill either the title or the review text box."
        return render_template('sentiment.html', form = form, check=check)

    @app.route('/dataset')
    def dataset():
        SITE_ROOT = os.path.realpath(os.path.dirname(__file__))
        json_url = os.path.join(SITE_ROOT, "data", "nyc_sample.json")
        dataset = json.load(open(json_url))
        return render_template('dataset.html', dataset=dataset)
    
    # return app
    return app



