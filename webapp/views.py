from flask import render_template, request, flash, redirect, url_for
from webapp import app
from .forms import InputTextForm
import utils
import gensim
import evaluate
from text_manipulation import split_sentences

if utils.config['test']:
    word2vec = None
else:
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(utils.config['word2vecfile'], binary=True)

model = evaluate.load_model()


def treat_text(raw_text):
    sentences = split_sentences(raw_text)
    print(sentences)

    cutoffs = evaluate.predict_cutoffs(sentences, model, word2vec)
    total = []
    segment = []
    for i, (sentence, cutoff) in enumerate(zip(sentences, cutoffs)):
        segment.append(sentence)
        if cutoff:
            total.append(segment)
            segment = []

    return total 




@app.route('/')
@app.route('/index', methods=['GET', 'POST'])
def index():
    return render_template('index.html')


@app.route('/result', methods=['POST'])
def result():
    segmentation = treat_text(request.form['Text'])
    print(segmentation)
    return render_template('result.html', segmentation=segmentation)
