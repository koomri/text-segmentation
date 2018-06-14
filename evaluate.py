import torch
import numpy as np
from torch.autograd import Variable
from choiloader import word_model
import utils
import text_manipulation

def load_model(model_path=None, is_cuda=None):
    if model_path is None:
        model_path = utils.config['model']

    with open(model_path, 'r') as f:
        model = torch.load(f)

    model.eval()
    if is_cuda is None:
        is_cuda = utils.config['cuda']

    return utils.maybe_cuda(model, is_cuda)


def prepare_tensor(sentences):
    tensored_data = []
    for sentence in sentences:
        if len(sentence) > 0:
            tensored_data.append(utils.maybe_cuda(torch.FloatTensor(np.concatenate(sentence))))

    return tensored_data



def text_to_word2vec(sentences, word2vec):
    new_text = []
    for sentence in sentences:
        words = text_manipulation.extract_sentence_words(sentence)
        new_text.append([word_model(w, word2vec) for w in words])

    return new_text


def predict_cutoffs(sentences, model, word2vec):
    word2vec_sentences = text_to_word2vec(sentences, word2vec)
    tensored_data = prepare_tensor(word2vec_sentences)
    batched_tensored_data = []
    batched_tensored_data.append(tensored_data)
    output = model(batched_tensored_data)
    values, argmax = output.max(1)
    argmax = argmax.data.cpu().numpy()
    return argmax
