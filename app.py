import os
import sys
import io
from flask import Flask, redirect, render_template, request, send_from_directory, url_for, jsonify
import requests
import shutil
import numpy as np
import cv2
import warnings
from PIL import Image
import tensorflow as tf
import keras
from keras import backend as K
import base64
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images
import mrcnn.model as modellib
from mrcnn.model import log
from trash import trash
import json


K.clear_session()

from tensorflow.python.keras.backend import set_session

sess = tf.compat.v1.Session()
graph = tf.compat.v1.get_default_graph()

warnings.filterwarnings("ignore", category=DeprecationWarning)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
# Directory to save logs and trained model
# MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# # Path to Trash trained weights
# TRASH_WEIGHTS_PATH = "weights/mask_rcnn_trash_0200_030519_large.h5"

# config = trash.TrashConfig()
# class InferenceConfig(config.__class__):
#     # Run detection on one image at a time
#     GPU_COUNT = 1
#     IMAGES_PER_GPU = 1

# config = InferenceConfig()
# # config.display()

# DEVICE = "/cpu:0"
# TEST_MODE = "inference"

# #loading model
# set_session(sess)
# model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# print("model loaded!")

# #loading weights
# weights_path = os.path.join(ROOT_DIR, TRASH_WEIGHTS_PATH)
# model.load_weights(weights_path, by_name=True)
# model.keras_model._make_predict_function()
# print("weights loaded!")

# @app.route('/predict', methods = ['GET', 'POST'])
# def get_trash():
#     # image_url = "https://firebasestorage.googleapis.com/v0/b/litterally-asean.appspot.com/o/down.png?alt=media&token=d723cb68-ac15-4e70-8bc1-190352311131"
#     image_url = request.data.decode('utf-8')
#     r = requests.get(image_url)

#     if r.status_code == 200:
#         image = Image.open(io.BytesIO(r.content))
#         # image.save('uploads/temp.png')
#         image = np.array(image)
#         image = image[:,:,:3]
#         print(image.shape)
#         global sess
#         global graph
#         with graph.as_default():
#             set_session(sess)
#             results = model.detect([image], verbose=1)
#             print(results)
        
#         data = dict()
#         # data['rois'] = np.array2string(results[0]['rois'], precision=2, separator=',')
#         data['rois'] = json.dumps(results[0]['rois'].tolist())
#         # data['class_ids'] = np.array2string(results[0]['class_ids'], precision=2, separator=',')
#         data['class_ids'] = json.dumps(results[0]['class_ids'].tolist())
#         # data['scores'] = np.array2string(results[0]['scores'], precision=2, separator=',')
#         data['scores'] = json.dumps(results[0]['scores'].tolist())
#         return jsonify(data)
#     else:
#         data = dict()
#         data['error'] = "error downloading image!"

def reconstruct(pb_path):
    if not os.path.isfile(pb_path):
        print("Error: %s not found" % pb_path)

    print("Reconstructing Tensorflow model")
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(pb_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    print("Success!")
    return detection_graph

def image2tensor(image):
    # npim = image2np(image)
    return np.expand_dims(image, axis=0)

def detect(detection_graph, test_image):
    with detection_graph.as_default():
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.01)
        with tf.compat.v1.Session(graph=detection_graph,config=tf.compat.v1.ConfigProto(gpu_options=gpu_options)) as sess:
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # image = Image.open(test_image_path)
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image2tensor(test_image)}
            )
            
        return boxes, scores, classes, num

@app.route('/classify', methods = ['GET', 'POST'])
def trash_classification():
    image_url = request.data.decode("utf-8")
    r = requests.get(image_url)

    if r.status_code == 200:
        image = Image.open(io.BytesIO(r.content))
        image = np.array(image)
        image = image[:,:,:3]

        ANNOTATIONS_FILE = './taco_ssd_weights/annotations.json'

        with open(ANNOTATIONS_FILE) as json_file:
            data = json.load(json_file)
            
        categories = data['categories']
        classes = []
        for i in range(len(categories)):
            classes.append(categories[i]['name'])

        detection_graph = reconstruct("./taco_ssd_weights/ssd_mobilenet_v2_taco_2018_03_29.pb")
        b, s, c, n = detect(detection_graph, image)

        class_names = []
        scores = []

        for i in range(int(n[0])):
            class_names.append(classes[int(c[0][i]) - 1])
            scores.append(str(s[0][i]))
        print(class_names)
        print(scores)

        op = dict()
        op['trash_classes'] = json.dumps(class_names)
        op['confidence'] = json.dumps(scores)
        return jsonify(op)
    else:
        op = dict()
        op['error'] = "error loading image file!"
        return jsonify(op)


if __name__ == '__main__':
    app.run(debug=False, threaded=True)