from flask import Flask,render_template, url_for , redirect
#from forms import RegistrationForm, LoginForm
#from sklearn.externals import joblib
from flask import request
import numpy as np
from PIL import Image
from flask import flash
#from flask_sqlalchemy import SQLAlchemy
#from model_class import DiabetesCheck, CancerCheck


import os
from tensorflow import keras
import tensorflow
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

from flask import send_from_directory
from tensorflow.keras.preprocessing import image
import tensorflow as tf

#from this import SQLAlchemy
app=Flask(__name__,template_folder='template')


app.config['SECRET_KEY'] = '5791628bb0b13ce0c676dfde280ba245'
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///site.db"

dir_path = os.path.dirname(os.path.realpath(__file__))
# UPLOAD_FOLDER = dir_path + '/uploads'
# STATIC_FOLDER = dir_path + '/static'
UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
#from keras.models import load_model

# global graph
# graph = tf.get_default_graph()
model = load_model('Covid_model.h5')


def api(full_path):
    data = tensorflow.keras.preprocessing.image.load_img(full_path, target_size=(224, 224, 3))
    #print(data.shape)
    data = np.expand_dims(data, axis=0)
    data = data * 1.0 / 255

    #with graph.as_default():
    predicted = model.predict(data)
    return predicted


# procesing uploaded file and predict it
	
@app.route('/upload', methods=['POST','GET'])
def upload_file():

    if request.method == 'GET':
        return render_template('index.html')
    else:
        try:
            file = request.files['image']
            full_name = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(full_name)

            indices = {1: 'Healthy', 0: 'Corona-Infected'}
            result = api(full_name)

            predicted_class = np.asscalar(np.argmax(result, axis=1))
            accuracy = round(result[0][predicted_class] * 100, 2)
            label = indices[predicted_class]

            return render_template('predict.html', image_file_name = file.filename, label = label, accuracy = accuracy)
        except :
            flash("Please select the image first !!", "danger")      
            return redirect(url_for("corona"))
	

	

@app.route('/uploads/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/")

@app.route("/home")
def home():
	return render_template("home.html")

@app.route("/about")
def about():
	return render_template("about.html")

@app.route("/corona")
def corona():
    return render_template("index.html")


if __name__ == "__main__":
	app.run(debug=True)
