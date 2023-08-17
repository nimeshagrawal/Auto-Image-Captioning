from pickle5 import pickle
import numpy as np
import os
from tqdm import tqdm
import json
import joblib


#from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
#from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


from PIL import Image
import matplotlib.pyplot as plt

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
#######################################################################


# Define a flask app
app = Flask(__name__)

# load features form pickle
# with open(os.path.join(os.getcwd(), 'features2.pickle'), "rb") as f:
#   features = pickle.load(f)

# load features form pickle
# with open(os.path.join(os.getcwd(), 'tokenizer2.pkl'), "rb") as f:
#   tokenizer = pickle.load(f)

features = joblib.load(os.path.join(os.getcwd(), 'features2.pkl'))
all_captions = joblib.load(os.path.join(os.getcwd(), 'all_captions.pkl'))
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_captions)
vocab_size = len(tokenizer.word_index) + 1


# Opening JSON file
with open('meta_data_mobilenet.json', 'r') as openfile:
    # Reading from json file
    json_object = json.load(openfile)
 
max_length, vocab_size = json_object['max_length'], json_object['vocab_size']

def get_model():
    inputs1 = Input(shape=(1000,))
    fe1 = Dropout(0.4)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    # sequence feature layers
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    #se1 = Word2Vec(all_captions, 256, min_count=1)(inputs2)
    se2 = Dropout(0.4)(se1)
    se3 = LSTM(256, return_sequences=True)(se2)
    se4 = LSTM(256)(se3)

    # decoder model
    decoder1 = add([fe2, se4])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    opt = Adam()
    model.compile(loss='categorical_crossentropy', optimizer=opt)

    return model

model = get_model()


# Model saved with Keras model.save()
MODEL_PATH = os.path.join(os.getcwd(), "model_mobilenet.h5")

# Load your trained model
model.load_weights(MODEL_PATH)
#model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded......... Check http://127.0.0.1:5000/')


def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length):
    # add start tag for generation process
    in_text = 'startseq'
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            in_text += ' ' + 'endseq'
            break
      
    return in_text

def generate_caption(image_name, model):
    # load the image
    # image_name = "1001773457_577c3a7d70.jpg"
    image_id = image_name.split('/')[-1][:-4]
    img_path = image_name
    image = Image.open(img_path)
    # predict the caption
    y_pred = predict_caption(model, features[image_id], tokenizer, max_length)

    y_pred = y_pred.replace("startseq", "")
    y_pred = y_pred.replace("endseq", "")
    y_pred = y_pred.lstrip()
    y_pred = y_pred.rstrip()

    return y_pred


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        print("\nthe current working directory is ..................\n");
        print(os.getcwd())
        print(file_path)
        print("\n\n\n\n\n")

       
        y_pred = generate_caption(file_path, model)

        return y_pred 
    return None


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=7860)


