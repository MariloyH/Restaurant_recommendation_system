from flask import Flask, url_for, render_template, request, jsonify
import json
import ML_recommendator


app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/test', methods=['POST'])
def test():
    output = request.get_json()
    result = json.loads(output)
    latitud = result['latitude']
    longitud = result['longitude']
    words = result['words']
    return ML_recommendator.findRecommendations(latitud,longitud,words)


if __name__ =="__main__":
    app.run(debug=True)

#For reference https://dataanalyticsireland.ie/2021/12/13/how-to-pass-a-javascript-variable-to-python-using-json/