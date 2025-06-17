import urllib
from PIL import Image
import neural_network
from tensorflow.keras import models as keras_models
import numpy as np
import glob, os
from io import BytesIO
from flask import Flask, abort, render_template, request, jsonify

app = Flask(__name__)
app.secret_key = b'\x9c,\x9fHp\x045\xe9\xb9_\xd3s\xed\x03\xdb\x8d'

@app.route("/")
def index():
    """Serves the default path and renders the basic HTML (+ CSS / JS)."""
    return render_template('nn.html')

@app.route('/list/mnist')
def models():
    models = glob.glob('*.npz') + glob.glob('*.keras')
    names = [os.path.splitext(os.path.basename(m))[0] for m in models]
    return jsonify(names)


def read_pixels(image_url):
    assert image_url.startswith('data:image/')

    # 1: Open the image 
    image_response = urllib.request.urlopen(image_url)
    image_bytes = image_response.read()
    with Image.open(BytesIO(image_bytes)) as image:

        # 2: Convert to 8bit grayscale
        # https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.convert
        # https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
        image = image.convert('L')

        # 3: Ensure we have the expected size.
        size = image.size
        target_size = (28,28)
        if size != target_size:
            app.logger.warning(f"Resizing image from {size} to {target_size}")
            image = image.resize(target_size)
        
        # 4: Get the pixel values and normalize from [0,255] to [0,1)
        pixel_data = image.getdata()
        assert len(pixel_data) == target_size[0] * target_size[1]
        pixels = np.array(pixel_data) / 255
        return pixels

@app.route('/predict/mnist', methods=['POST'])
def predict():
    """Predicts the written digit using a forward pass through a simple feed-forward neural network.
       Expects a request with a JSON payload of the following format:
       {
         label: 'mnist_digits_v1',
         data: 'data:image/png;base64,iVBORw0KGgoAAAA ... =' # A data URL for a 28x28 pixel image
       }
       and returns the following JSON response:
       {
         label: 'mnist_digits_v1',
         prediction: 5,  # digit in [0,9]
         full_prediction: [ 0.2, 0.01, 0.03, 0.5 ... ]  # full prediction probabilities for each digit in [0,9]
       }
    """
    # Ensure the request is reasonable before parsing into JSON.
    if request.content_length > 2000:
        abort(413, f"Too much data: {request.content_length}") # Payload too large

    # Ensure we agree on the version and that the image URL uses the data scheme.
    request_json = request.get_json()
    model = request_json['label']
    model_file = f"{model}.npz"
    if np.lib.npyio.DataSource().exists(model_file):
        # 5: Feed-forward through network
        nn = neural_network.NN.LoadFromFile(model_file)
        eval =  neural_network.ForwardEvaluator()
        prediction = eval.Evaluate(nn, read_pixels(request_json['data']))
    else:
        model_file = f"{model}.keras"
        tf_model = keras_models.load_model(model_file)
        pixels = read_pixels(request_json['data'])
        pixels = pixels.reshape(28, 28)
        pixels = np.expand_dims(pixels, -1)
        pixels = np.expand_dims(pixels, 0)
        prediction = tf_model.predict(pixels, verbose=0)[0]
    # 6: Formulate response.
    digit = int(np.argmax(prediction))
    full = np.round(np.asarray(prediction, dtype='float64'), 2)
    return jsonify({
        'label': model,
        'prediction': digit,
        'full_prediction': full.tolist(),
    })
    
if __name__ == '__main__':
    app.run()