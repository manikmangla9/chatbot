#import files
from flask import Flask, render_template, request
import nltk
nltk.download('punkt') 
import numpy as np
import tflearn
import tensorflow as tf
#from tensorflow.python.compiler.tensorrt import trt_convert as trt
import random
import json
import pickle
from nltk.stem.lancaster import LancasterStemmer
stemmer=LancasterStemmer()

app = Flask(__name__)

with open("data.pickle","rb") as f:
    words,labels,training,output=pickle.load(f)


f=open('intents.json',)

intents = json.load(f)


tf.reset_default_graph()
net=tflearn.input_data(shape=[None,len(training[0])])
net=tflearn.fully_connected(net,8)
net=tflearn.fully_connected(net,8)
net=tflearn.fully_connected(net,len(output[0]),activation="softmax")
net=tflearn.regression(net)

model=tflearn.DNN(net)

model.load("model.tflearn")
def bag_of_words(s,words):
    bag=[0 for i in range(len(words))]
    s_words=nltk.word_tokenize(s)
    s_words=[stemmer.stem(w.lower()) for w in s_words]
    
    for se in s_words:
        for i,w in enumerate(words):
            if se==w:
                bag[i]=1
    return np.array(bag)
            
def chat(userText):
    '''print("start chating...")
    while(True):
        inp=input("you: ")
        if inp.lower()=="quit":
            break;
        else:'''
            #results=[]
    results=model.predict([bag_of_words(userText,words)])
    #a=max(results)
    results=np.array(results)
    results_index=np.argmax(results)
    #print(results[0][results_index])
    if results[0][results_index]> 0.7:
        tag=labels[results_index]
        for tg in intents['intents']:
            if tg['tag']==tag:
                response=tg['responses']
        return random.choice(response)
    else:
        return("sorry i didn't get that, Try again..")


@app.route("/")
def home():    
    return render_template("home.html") 
@app.route("/get",methods=["GET","POST"])
def get_bot_response():    
    userText = request.args.get('msg')    
    return str(chat(userText)) 
if __name__ == "__main__":    
    app.run(port=7000)
