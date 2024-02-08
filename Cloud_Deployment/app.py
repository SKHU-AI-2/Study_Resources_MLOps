from flask import Flask, render_template, request, send_file
import cv2
import numpy as np
from io import BytesIO
import base64

app = Flask(__name__)

def convert_image_to_grayscale(image):
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return grayscale_image

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            image_stream = BytesIO(uploaded_file.read())
            image_stream.seek(0)
            file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            option = request.form.get('option')

            if option == '흑백변환':
                grayscale_image = convert_image_to_grayscale(image)
                _, img_encoded = cv2.imencode('.png', grayscale_image)
                grayscale_img_base64 = base64.b64encode(img_encoded).decode('utf-8')
                return render_template('results.html', processed_image=grayscale_img_base64)

    return render_template('index.html')

if __name__ == '__main__':
    app.run('0.0.0.0', port=5000, debug=True)