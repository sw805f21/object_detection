import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

import os
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

import cv2 
import numpy as np
import time

import re

DETECT_HISTORY_THRESHOLD = 0.6

# Natural sorting. sorting for natural keys to sort checkpoints, since they are in the format "ckpt- + some_number"
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

CUSTOM_MODEL_NAME = 'hand_sign_recognition_model'
WORKSPACE_PATH = 'Tensorflow\\workspace'
SCRIPTS_PATH = 'Tensorflow\\scripts'
APIMODEL_PATH = 'Tensorflow\\models'
ANNOTATION_PATH = WORKSPACE_PATH+'\\annotations'
IMAGE_PATH = WORKSPACE_PATH+'\\images'
MODEL_PATH = WORKSPACE_PATH+'\\models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH+'\\pre-trained-models'
CONFIG_PATH = MODEL_PATH+'/' + CUSTOM_MODEL_NAME + '/pipeline.config'
CHECKPOINT_PATH = MODEL_PATH+'/' + CUSTOM_MODEL_NAME + '/checkpoints'


CONFIG_PATH = MODEL_PATH+'\\'+CUSTOM_MODEL_NAME+'\\pipeline.config'
print(CONFIG_PATH)


# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)


# Pick last checkpoint for inference
checkpoints = []
for subdir, dirs, files in os.walk(CHECKPOINT_PATH):
    for file_name in files:
        if(file_name.endswith("index")):
            checkpoints.append(file_name[:-6])
checkpoints.sort(key=natural_keys)

last_checkpoint = checkpoints[-1]


# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, last_checkpoint)).expect_partial()


@tf.function
def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    #print(prediction_dict[0][0])
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections



category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH+'/label_map.pbtxt')

#cap.release()

# Setup capture
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('test_video.mp4')

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Write video setup
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.avi', fourcc, 20.0, (width, height))

history = []
old_word = ""



while(cap.isOpened()): 
    ret, frame = cap.read()
    if(ret == True):
        image_np = np.array(frame)
        
        # Writes content to disk
        #out.write(frame)
        
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)
        
        #print(detections.items())

        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        #print(detections)
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        #print(category_index) # five categories

        #print(detections['detection_classes']+label_id_offset)

        viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    detections['detection_boxes'],
                    detections['detection_classes']+label_id_offset,
                    detections['detection_scores'],
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=5,
                    min_score_thresh=.5,
                    agnostic_mode=False)

        cv2.imshow('object detection',  cv2.resize(image_np_with_detections, (800, 600)))
        
        #print(detections['detection_classes'][np.argmax(detections['detection_scores'])]) # number between 0 and 4
        #print("Detection_classes with index max / 0")
        #print(detections['detection_classes'][np.argmax(detections['detection_scores'])])
        #print("Detection_classes")
        #print(detections['detection_classes'])
        #print("arg_max detection_scores : 0 0 0 0 0 ")

        # per default detections_scores is sorted after highest probability, meaning the first member of the array will always be the maximum.
        
        detection_score = detections['detection_scores'][0]

        # If it is above 80% certain it have detected a object add it to history. Only add new entries when words change. 
        if(detection_score >= DETECT_HISTORY_THRESHOLD):
            word = category_index[detections['detection_classes'][np.argmax(detections['detection_scores'])] + 1]['name']
            if(word != old_word):
                history.append(word)
                print(history)
                old_word = word
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            break
    else:
        break

cap.release()
# Write video release
#out.release()
