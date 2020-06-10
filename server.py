from flask import Flask, request, render_template, jsonify
import json
from methods.preprocess import pre_process_driver
import pandas as pd
from flask_api import status
import keras.backend.tensorflow_backend as tb

server = Flask(__name__)

@server.route("/")
def index():
    return "Under Construction Index Pages"

def check_if_json(data):
    if("body" and "url" in data ):
        if (data.get('body') and data.get('url') is not None):
            return True
        else:
            return False
    else:
        return False    

@server.route("/process", methods=["POST"])
def extensionInput():
    tb._SYMBOLIC_SCOPE.value = True     #IDK what is this but it worked for keras and TF. Some threading value issue. 
    extension_data = request.get_json()
    if( check_if_json(extension_data) ):
        response = pre_process_driver(extension_data['body'])
        web_url = extension_data['url']
        # return ''' The Content is: {}''' .format(content)
        # return jsonify(
        #     URL = web_url,
        #     policy = response
        # ) 
        return (json.dumps(response), status.HTTP_200_OK)
    else:
        response = "Bad Request"
        return response, status.HTTP_400_BAD_REQUEST


   
    
@server.route("/bad")
def view():
    return render_template("badRequest.html")

if __name__ == "__main__":
    server.run(debug=True, host='0.0.0.0',threaded=False)

