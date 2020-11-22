import os
import cv2
import uuid
import numpy as np
import face_recognition as fc
from PIL import Image
from flask import Flask
from flask import request, jsonify

app = Flask(__name__)


@app.route("/api/compare_images", methods=['POST'])
def compare_images():
    im1 = request.files['image1']
    filename1, file_extension1 = os.path.splitext(im1.filename)
    genimagename1 = str(uuid.uuid4())
    im1.save(os.path.join('UPLOADEDIMAGES', (genimagename1+file_extension1)))
    image1 = fc.load_image_file(os.path.join('UPLOADEDIMAGES', (genimagename1+file_extension1)))
    image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    image1encode = fc.face_encodings(image1)[0]

    im2 = request.files['image2']
    filename2, file_extension2 = os.path.splitext(im2.filename)
    genimagename2 = str(uuid.uuid4())
    im2.save(os.path.join('UPLOADEDIMAGES', (genimagename2+file_extension2)))
    image2 = fc.load_image_file(os.path.join('UPLOADEDIMAGES', (genimagename2+file_extension2)))
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    image2encode = fc.face_encodings(image2)[0]



    # image2 = fc.load_image_file("images/2.jpg")

    result = fc.compare_faces([image1encode], image2encode)
    r = fc.face_distance([image1encode], image2encode)
    print(result[0])
    print(r)

    return jsonify(bool(result[0]))


if __name__ == "__main__":
    app.run(port=9000)
