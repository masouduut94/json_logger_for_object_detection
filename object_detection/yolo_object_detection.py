# USAGE
# python yolo_object_detection.py --input ../example_videos/janie.mp4
# --output ../output_videos/yolo_janie.avi --yolo yolo-coco --display 0

# python yolo_object_detection.py --input ../example_videos/janie.mp4
# --output ../output_videos/yolo_janie.avi --yolo yolo-coco --display 0 --use-gpu 1
from os.path import splitext, basename, join

from imutils.video import FPS
from tqdm import tqdm
from json_parser.json_parser import JsonParser
from argparser import parser
import numpy as np
import cv2
import os


args = parser()
labelsPath = os.path.sep.join([args.yolo, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")
weightsPath = os.path.sep.join([args.yolo, "yolov3.weights"])
configPath = os.path.sep.join([args.yolo, "yolov3.cfg"])

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

if args.use_gpu:
    # set CUDA as the preferable backend and target
    print("[INFO] setting preferable backend and target to CUDA...")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the width and height of the frames in the video file
W = None
H = None
writer = None
json_logger = JsonParser(top_k_labels=1)


print("[INFO] accessing video stream...")
vs = cv2.VideoCapture()
codec = cv2.VideoWriter_fourcc(*'XVID')
fps = FPS().start()

if vs.open(args.input if args.input else 0):
    property_id = int(cv2.CAP_PROP_FRAME_COUNT)
    total_frames = int(cv2.VideoCapture.get(vs, property_id))

    width, height = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH)), \
                    int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_out = vs.get(cv2.CAP_PROP_FPS)
    filename, extension = splitext(basename(args.input))
    output_path = join(args.output_dir, filename + '.avi')
    writer = cv2.VideoWriter(output_path, codec, fps_out, (width, height))
    frame_counter = 0
    pbar = tqdm(total=total_frames + 1)

    json_logger.set_start()
    json_logger.set_top_k(1)
    video_details = dict(frame_width=width, frame_height=height, frame_rate=fps_out, video_name=basename(args.input))
    json_logger.add_video_details(**video_details)

    while vs.isOpened():
        frame_counter += 1
        ret, frame = vs.read()
        if not ret:
            break

        if W is None or H is None:
            (H, W) = frame.shape[:2]

        step = 1 if args.step == 0 else args.step
        if frame_counter % step == 0:

            # add frame only when you are sure it can be read
            json_logger.add_frame(frame_counter)
            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                         swapRB=True, crop=False)
            net.setInput(blob)
            layerOutputs = net.forward(ln)

            boxes = []
            confidences = []
            classIDs = []

            for output in layerOutputs:
                for detection in output:
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]

                    if confidence > args.confidence:
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")

                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))

                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)

            idxs = cv2.dnn.NMSBoxes(boxes, confidences, args.confidence,
                                    args.threshold)

            if len(idxs) > 0:
                for i in idxs.flatten():
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])

                    color = [int(c) for c in COLORS[classIDs[i]]]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    text = "{}: {:.4f}".format(LABELS[classIDs[i]],
                                               confidences[i])
                    cv2.putText(frame, text, (x, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    json_logger.add_bbox_to_frame(frame_counter, int(i), x, y, w, h)
                    json_logger.add_label_to_bbox(frame_counter, int(i), category=LABELS[classIDs[i]],
                                                  confidence=confidences[i])
            json_logger.schedule_output(output_path=args.output_dir, seconds=5)
            writer.write(frame)
            fps.update()
        pbar.update(1)
    # json_logger.json_output(output_name='output.json')
    fps.stop()
    pbar.close()

    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
