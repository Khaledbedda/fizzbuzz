# We now need the json library so we can load and export json data

from flask import Flask

from transformers import T5Tokenizer, T5ForConditionalGeneration
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")


# Creation of the Flask app
server = Flask(__name__)

# API 1
# Flask route so that we can serve HTTP traffic on that route
@server.route('/')
# Get data from json and return the requested row defined by the variable Line
def Welcome():
     return 'The model is up and running. Send a POST request'
    

# API 2
# Flask route so that we can serve HTTP traffic on that route
@server.route('/translate/German/<Line>',methods=['POST', 'GET'])
# Return prediction for both Neural Network and LDA inference model with the requested row as input
def translate_to_German(Line):
    input_ids = tokenizer("translate English to German: " + Line, return_tensors="pt").input_ids

    outputs = model.generate(input_ids)
    return {'Translation': tokenizer.decode(outputs[0], skip_special_tokens=True)}

@server.route('/translate/French/<Line>',methods=['POST', 'GET'])
# Return prediction for both Neural Network and LDA inference model with the requested row as input
def translate_to_french(Line):
    input_ids = tokenizer("translate English to French: " + Line, return_tensors="pt").input_ids

    outputs = model.generate(input_ids)
    return {'Translation': tokenizer.decode(outputs[0], skip_special_tokens=True)}

@server.route('/translate/Romanian/<Line>',methods=['POST', 'GET'])
# Return prediction for both Neural Network and LDA inference model with the requested row as input
def translate_to_romanian(Line):
    input_ids = tokenizer("translate English to Romanian: " + Line, return_tensors="pt").input_ids

    outputs = model.generate(input_ids)
    return {'Translation': tokenizer.decode(outputs[0], skip_special_tokens=True)}




