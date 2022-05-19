import argparse
import numpy as np
import cv2
import os
import sys
import tensorflow as tf
from utils import label_map_util
from utils import visualization_utils as vis_util
import threading

MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017'
# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = 'models/' + MODEL_NAME + '/frozen_inference_graph.pb'
# List of the strings that is used to add correct label for each box.
CWD_PATH = os.getcwd()
PATH_TO_LABELS = os.path.join(CWD_PATH,'object_detection', 'data', 'mscoco_label_map.pbtxt')
NUM_CLASSES = 90
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080
targetX, targetY = 500, 500
targetDim = 250

class OutputFrame:
    def __init__(self):
        self.frame = np.zeros((IMAGE_HEIGHT,IMAGE_WIDTH,3))
        self.boxes = ()

def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

class WebcamThread(threading.Thread):
   def __init__(self, name):
      threading.Thread.__init__(self)
      self.name = name
   def run(self):
      print("Starting " + self.name)
      get_frame(self.name)
      print("Exiting " + self.name)

def get_frame(threadName):
    while not done:
        _, frame = cap.read()
        output_frame.frame = frame

class PredictorThread(threading.Thread):
   def __init__(self, name):
      threading.Thread.__init__(self)
      self.name = name
   def run(self):
      print("Starting " + self.name)
      predict(self.name)
      print("Exiting " + self.name)

def predict(threadName):
    while not done:
        _, image_np = cap.read()
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        output_frame.boxes = sess.run(
          [boxes, scores, classes, num_detections],
          feed_dict={image_tensor: image_np_expanded})


def click_callback(event, x, y, flags, param):
    global targetX, targetY
    if event == 1:
        targetX, targetY = x, y

def change_zoom(x):
    global targetDim
    zoom = cv2.getTrackbarPos('Zoom','Original')
    zoom *= 50
    targetDim = zoom + 200

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=int, default=0, help='The video source')
    parser.add_argument('--crop', action='store_true', help='Crop video to cats')
    parser.add_argument('--manual', action='store_true', help='Move the crop manually by clicking')
    opt = parser.parse_args()

    source, crop, manual = opt.source, opt.crop, opt.manual
    currentCrop = [0, 1000, 0, 1000]
    centerX, centerY = 500, 500
    largestDim = 250
    lastTargetId = 0
    lostTargetCount = 0
    catBox = None
    firstTargetX, firstTargetY, firstTargetDim = 0, 0, 0
    secondTargetX, secondTargetY, secondTargetDim = 0, 0, 0

    done = False
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    cap = cv2.VideoCapture(source)
    cap.set(3, IMAGE_WIDTH)
    cap.set(4, IMAGE_HEIGHT)
    sess = tf.compat.v1.Session(graph=detection_graph)
    output_frame = OutputFrame()

    webcam_thread = WebcamThread("Webcam Thread")
    webcam_thread.start()
    if not manual:
        predictor_thread = PredictorThread("Predictor Thread")
        predictor_thread.start()

    print("---- Starting CatTracker ----")

    cv2.namedWindow('Original')
    if manual:
        cv2.setMouseCallback('Original', click_callback)
        cv2.createTrackbar('Zoom','Original',1,10,change_zoom)

    while True:
        if output_frame.boxes == ():
            to_show = output_frame.frame
        else:
            to_show = output_frame.frame
        if not manual and not output_frame.boxes == ():
                catBoxes = vis_util.visualize_boxes_and_labels_on_image_array(
                to_show,
                np.squeeze(output_frame.boxes[0]),
                np.squeeze(output_frame.boxes[2]).astype(np.int32),
                np.squeeze(output_frame.boxes[1]),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=8,
                width=IMAGE_WIDTH,
                height=IMAGE_HEIGHT,
                crop=crop)

                firstTargetX, firstTargetY, firstTargetDim, secondTargetX, secondTargetY, secondTargetDim = 0,0,0,0,0,0
                if catBoxes and len(catBoxes) != 0:
                    firstTargetX = int((catBoxes[0][1] + catBoxes[0][3]) / 2)
                    firstTargetY = int((catBoxes[0][0] + catBoxes[0][2]) / 2)
                    firstTargetDim = int(max([int(catBoxes[0][3]) - int(catBoxes[0][1]), int(catBoxes[0][2]) - int(catBoxes[0][0])]) * 1.3)
                    if len(catBoxes) > 1:
                        secondTargetX = int((catBoxes[1][1] + catBoxes[1][3]) / 2)
                        secondTargetY = int((catBoxes[1][0] + catBoxes[1][2]) / 2)
                        secondTargetDim = int(max([int(catBoxes[1][3]) - int(catBoxes[1][1]), int(catBoxes[1][2]) - int(catBoxes[1][0])]) * 1.3)

                if (secondTargetX == 0 and lostTargetCount == 0 and firstTargetX != 0):
                    targetX = firstTargetX
                    targetY = firstTargetY
                    targetDim = firstTargetDim

                if (secondTargetX != 0):
                    if (abs(secondTargetX - firstTargetX) < (.3 * IMAGE_WIDTH) and abs(secondTargetY - firstTargetY) < (.3 * IMAGE_HEIGHT)):
                        lostTargetCount = 2
                        targetX = int((firstTargetX + secondTargetX) / 2)
                        targetY = int((firstTargetY + secondTargetY) / 2)
                        targetDim = int(max(firstTargetDim, secondTargetDim) * 1.85)
                    else:
                        targetX = firstTargetX
                        targetY = firstTargetY
                        targetDim = firstTargetDim
                        lostTargetCount = lostTargetCount - 1 if lostTargetCount != 0 else 0
                
                if (secondTargetX == 0 and lostTargetCount > 0):
                    lostTargetCount -= 1

        newCenterX = int(centerX * .98 + targetX * .02)
        if newCenterX - centerX > 10:
            newCenterX = centerX + 10
        elif centerX - newCenterX > 10:
            newCenterX = centerX - 10
        newCenterY = int(centerY * .98 + targetY * .02)
        if newCenterY - centerY > 10:
            newCenterY = centerY + 10
        elif centerY - newCenterY > 10:
            newCenterY = centerY - 10
        newLargestDim = int(largestDim * .98 + targetDim * .02)
        if newLargestDim - largestDim > 10:
            newLargestDim = largestDim + 10
        elif largestDim - newLargestDim > 10:
            newLargestDim = largestDim - 10
        centerX, centerY, largestDim = newCenterX, newCenterY, newLargestDim

        startY = int(centerY - largestDim / 2)
        endY = int(centerY + largestDim / 2)
        startX = int(centerX - largestDim / 2)
        endX = int(centerX + largestDim / 2)

        # Keep camera within frame boundries
        if startY <= 0 or endY >= IMAGE_HEIGHT:
            if startY <= 0:
                endY += (0 - startY)
                startY = 0
            else:
                startY -= (endY - IMAGE_HEIGHT)
                endY = IMAGE_HEIGHT
        if startX <= 0 or endX >= IMAGE_WIDTH:
            if startX <= 0:
                endX += (0 - startX)
                startX = 0
            else:
                startX -= (endX - IMAGE_WIDTH)
                endX = IMAGE_WIDTH
        currentCrop = [int(max(0, startY)), int(min(IMAGE_HEIGHT, endY)), int(max(0, startX)), int(min(IMAGE_WIDTH, endX))]
        imgTest = to_show[currentCrop[0]: currentCrop[1], currentCrop[2]: currentCrop[3]]
        resizedImgTest = cv2.resize(imgTest, (500, 500), interpolation=cv2.INTER_AREA)
        if manual:
            vis_util.draw_bounding_box_on_image_array(
            to_show,
            currentCrop[0],
            currentCrop[2],
            currentCrop[1],
            currentCrop[3],
            use_normalized_coordinates=False,
            color='purple'
            )

        if crop:
            cv2.imshow('Cat', resizedImgTest)
            cv2.imshow('Original', to_show)
        else:
            cv2.imshow('Original', to_show)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            done = True
            break

    cap.release()
    cv2.destroyAllWindows()
