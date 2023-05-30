
from flask import Flask, jsonify, make_response, request
from agents.conversation_agent import conversation_agent

app = Flask(__name__)

@app.route("/", methods = ['POST'])
def chat():
    if not request.is_json:
        return make_response(
            jsonify(
                {"success": False,
                 "error": "Unexpected error, request is not in JSON format"}),
            400)
    
    try:
        data = request.json
        message = data["message"]
        result = conversation_agent(message)
        return jsonify({"success": True, "data": {result}})
    except:
        return make_response(
            jsonify(
                {"success": False, 
                 "error": "Unexpected error: failed to send the message"}),
            400)

@app.route("/", methods =["GET"])
def test():
    try:
        return make_response(
            jsonify(
            {
                "success": True,
                "data": "It is working"
            }
            )
        )
    except:
        return make_response(
            jsonify(
            {
                "success": False,
                "error": "Unexpected error"
            }
            )
        )