from ui import ChatUI
from curses import wrapper

import json
import pickle
import random
from tensorflow.keras.models import load_model
import numpy as np
import sys

import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()


nltk.download('omw-1.4')

try:
    model = load_model('trained_model.h5')
except:
    print('can\'t find: trained_model.h5')
    sys.exit(0)
try:
    intents = json.loads(open('intents.json').read())
except:
    print('can\'t find: intents.json')
    sys.exit(0)
try:
    words = pickle.load(open('words.pkl', 'rb'))
except:
    print('can\'t find: words.pkl')
    sys.exit(0)
try:
    classes = pickle.load(open('classes.pkl', 'rb'))
except:
    print('can\'t find: classes.pkl')
    sys.exit(0)


# 4 -> HERE | 5
def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(
        word.lower()) for word in sentence_words]
    return sentence_words
# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence


# 2 -> HERE | 4
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return(np.array(bag))


# 1 -> HERE | 2
def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


# 1 -> HERE | 3
def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    return result


# FIRST CALL | 1
def chatbot_response(text):
    ints = predict_class(text, model)
    res = getResponse(ints, intents)
    return res


def main(stdscr):
    stdscr.clear()
    ui = ChatUI(stdscr)
    ui.userlist.append(' ')
    ui.redraw_userlist()
    inp = ""
    while inp != "/quit":
        inp = ui.wait_input("Type here: ")
        ui.chatbuffer_add('You typed: ')
        ui.chatbuffer_add(inp)
        res = chatbot_response(inp)
        ui.chatbuffer_add('Bot says: ')
        ui.chatbuffer_add(res)
        ui.chatbuffer_add('---------------- x ----------------')


wrapper(main)