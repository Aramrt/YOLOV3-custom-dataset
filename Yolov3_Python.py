import numpy as np
import cv2 as cv

# Initialize the parameters
confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4  # Non-maximum suppression threshold

inpWidth = 416  # 608     # Width of network's input image
inpHeight = 416  # 608     # Height of network's input image
# I don not need classe.txt that contains labels

# barcode Decencies (YOLO OR Tiny-YOLO)
# Tiny YOLO has been choice because of the speed it has
#
# modelConfiguration_barcode = "./uti/yolov3-tiny.cfg"
# modelWeights_barcode = "./uti/yolov3-tiny_final.weights"

modelConfiguration_barcode = "./uti/Both_cfg.cfg"
modelWeights_barcode = "./uti/yolov3_final_barcode.weights"

net_barcode = cv.dnn.readNetFromDarknet(modelConfiguration_barcode, modelWeights_barcode)
net_barcode.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
net_barcode.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)


def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Remove the bounding boxes with low confidence using non-maxima suppression
def post_process(img, outs):
    frameHeight = img.shape[0]
    frameWidth = img.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    # False for vars
    left, top, width, height = False, False, False, False
    for out in outs:
        # print("out.shape : ", out.shape)
        for detection in out:
            # if detection[4]>0.001:
            # print(detection[0])
            scores = detection[5:]
            classId = np.argmax(scores)
            # if scores[classId]>confThreshold:
            confidence = scores[classId]
            if detection[4] > confThreshold:
                # print(detection[4], " - ", scores[classId],
                #     " - th : ", confThreshold)
                # print(detection)
                pass
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])
                # print(boxes)

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    # print(indices)
    counter = 0
    # for having right boxes we use croper = []
    croper = []
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]

        croper.append([left, top, width, height])
    return croper


# TWO parameter :
# 1. Converted image to array
# 2. the path that cropped barcode has to be saved
# returning True for 5 times will give us the permision to OCR the last frame...
def Predict_barcode(image_array_barcode):
    img_barcode = cv.resize(image_array_barcode, None, fx=0.5, fy=0.5)
    blob_barcode = cv.dnn.blobFromImage(img_barcode, 0.00392, (inpWidth, inpHeight), (0, 0, 0), 1, crop=False)

    # Sets the input to the network
    net_barcode.setInput(blob_barcode)

    outs_barcode = net_barcode.forward(getOutputsNames(net_barcode))
    # making a predict var to check the existence and boxes --> once <-- (faster...)
    predict = post_process(img_barcode, outs_barcode)
    # not giving frames without boxes into network...
    if len(predict) != 0:
        for x in predict:
            crop_barcode = img_barcode[x[1]: x[1] + x[3], x[0]: x[0] + x[2]]
            cv.imshow('image', crop_barcode)
            cv.waitKey(0)
    else:
        print("Not Found!")


image_barcode_predict = cv.imread("1.jpg")
Predict_barcode(image_barcode_predict)
