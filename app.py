import urllib
from PIL import Image
import neural_network
import numpy as np
from io import BytesIO

from flask import Flask, abort, render_template, request, jsonify

# We need the following endpoints
# 1) predict: Takes image data and makes a single feed-forward pass, returning the predicted category
# 2) (optional): Train&Eval Loop with given meta-parameters, cancellation, progress...

app = Flask(__name__)
app.secret_key = b'\x9c,\x9fHp\x045\xe9\xb9_\xd3s\xed\x03\xdb\x8d'

@app.route("/")
def index():
    return render_template('nn.html')

@app.route('/predict/mnist/', methods=['POST'])
def predict():
    if request.content_length > 2000:
        abort(413, f"Too much data: {request.content_length}") # Payload too large

    request_json = request.get_json()
    assert request_json['label'] == 'mnist_digits_v1'
    image_url = request_json['data']
    assert image_url.startswith('data:image/')
    image_response = urllib.request.urlopen(image_url)
    image_bytes = image_response.read()
    image = Image.open(BytesIO(image_bytes))
    app.logger.warning(image.getbands())
    image = image.convert('L')
    app.logger.warning(image.getbands())
    size = image.size
    app.logger.warning(f"Size: {size}")
    target_size = (28,28)
    if size != target_size:
        app.logger.warning(f"Resizing image from {size} to {target_size}")
        image = image.resize(target_size)
    
    pixel_data = image.getdata()
    assert len(pixel_data) == target_size[0] * target_size[1]
    pixels = np.array(pixel_data) / 255

    nn = neural_network.NN.LoadFromFile('mnist_best.npz')
    eval =  neural_network.ForwardEvaluator()
    prediction = eval.Evaluate(nn, pixels)
    digit = int(np.argmax(prediction))
    full = np.round(prediction, 2)
    return jsonify({
        'label': 'mnist_digits_v1',
        'prediction': digit,
        'full_prediction': full.tolist(),
    })


