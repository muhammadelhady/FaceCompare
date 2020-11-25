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
    image1Faces = fc.face_encodings(image1, num_jitters=28)


    im2 = request.files['image2']
    filename2, file_extension2 = os.path.splitext(im2.filename)
    genimagename2 = str(uuid.uuid4())
    im2.save(os.path.join('UPLOADEDIMAGES', (genimagename2+file_extension2)))
    image2 = fc.load_image_file(os.path.join('UPLOADEDIMAGES', (genimagename2+file_extension2)))
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    image2Faces = fc.face_encodings(image2, num_jitters=60)

    if image1Faces.__len__() == 0 or image2Faces.__len__() == 0:
        return jsonify("there is No Faces Detected in the image ... try again!!!")

    if image1Faces.__len__() > 1 or image2Faces.__len__() > 1:
        return jsonify("there is too many Faces Detected in the image ... try again!!!")


    image1encode = image1Faces[0]
    os.remove(os.path.join('UPLOADEDIMAGES', (genimagename1 + file_extension1)))

    image2encode = image2Faces[0]
    os.remove(os.path.join('UPLOADEDIMAGES', (genimagename2+file_extension2)))



    result = fc.compare_faces([image1encode], image2encode, 0.5)

    distance = fc.face_distance([image1encode], image2encode)
    sure_percentage = (1-distance)*100

    print(result[0])
    print(distance)
    print(sure_percentage)


    return jsonify({
       "match": bool(result[0]),
       "sure": int(sure_percentage)
    }

    )


if __name__ == "__main__":
    app.run(port=9000)
