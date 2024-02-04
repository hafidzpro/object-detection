import cv2
import numpy as np
import csv

Threshold = 0.5
image_size = 320
report_size = (800, 600)
classes_names = []

csv_file_path = 'object_detection_results.csv'
with open(csv_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Frame', 'Class', 'Confidence', 'X', 'Y', 'Width', 'Height'])

def write_to_csv(frame_number, class_name, confidence, x, y, width, height):
    with open(csv_file_path, 'a', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow([frame_number, class_name, confidence, x, y, width, height])

def predictions(img, frame, final_box, cordinates, confidence_score, ids, width_ratio, height_ratio, frame_number):
    global classes_names

    total_persons_detected = []
    total_cars_detected = []
    total_motorcycles_detected = []

    c = []
    k = []
    m = []
    count1 = 0
    count2 = 0
    count3 = 0

    font = cv2.FONT_HERSHEY_COMPLEX_SMALL

    if len(final_box) > 0:
        for i in final_box.flatten():
            if classes_names[ids[i]] == 'car':
                count1 += 1
                k.append(count1)

                x, y, w, h = cordinates[i]
                x = int(x * width_ratio)
                y = int(y * height_ratio)
                w = int(w * width_ratio)
                h = int(h * height_ratio)
                cnf = str(round(confidence_score[i], 2))
                text = str(classes_names[ids[i]]) + '-' + cnf + '%'
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, text, (x, y - 2), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

                write_to_csv(frame_number, 'car', confidence_score[i], x, y, w, h)

            elif classes_names[ids[i]] == 'person':
                count2 += 1
                c.append(count2)
                x, y, w, h = cordinates[i]
                x = int(x * width_ratio)
                y = int(y * height_ratio)
                w = int(w * width_ratio)
                h = int(h * height_ratio)
                cnf = str(round(confidence_score[i], 2))
                text = str(classes_names[ids[i]]) + '-' + cnf
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, text, (x, y - 2), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

                write_to_csv(frame_number, 'person', confidence_score[i], x, y, w, h)

            elif classes_names[ids[i]] == 'motorbike':
                count3 += 1
                m.append(count3)
                x, y, w, h = cordinates[i]
                x = int(x * width_ratio)
                y = int(y * height_ratio)
                w = int(w * width_ratio)
                h = int(h * height_ratio)
                cnf = str(round(confidence_score[i], 2))
                text = str(classes_names[ids[i]]) + '-' + cnf
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame, text, (x, y - 2), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

                write_to_csv(frame_number, 'motorbike', confidence_score[i], x, y, w, h)

        if len(k) > 0:
            text1 = 'Total cars : {}'.format(k[-1])
            cv2.putText(img, text1, (10, 25), font, 1, (255, 255, 0), 1, cv2.LINE_4)
        else:
            k.append(0)
            text1 = 'Total cars : {}'.format(k[-1])
            cv2.putText(img, text1, (10, 25), font, 1, (255, 255, 0), 1, cv2.LINE_4)

        if len(c) > 0:
            text1 = 'Persons : {}'.format(c[-1])
            cv2.putText(img, text1, (200, 25), font, 1, (0, 0, 255), 1, cv2.LINE_4)
        else:
            c.append(0)
            text1 = 'Persons : {}'.format(c[-1])
            cv2.putText(img, text1, (300, 25), font, 1, (0, 0, 255), 1, cv2.LINE_4)

        if len(m) > 0:
            text1 = 'Total Motorcycles : {}'.format(m[-1])
            cv2.putText(img, text1, (400, 25), font, 1, (0, 255, 255), 1, cv2.LINE_4)
        else:
            m.append(0)
            text1 = 'Total Motorcycles : {}'.format(m[-1])
            cv2.putText(img, text1, (400, 25), font, 1, (0, 255, 255), 1, cv2.LINE_4)

        print('cars = ', len(k))
        print('persons = ', len(c))
        print('motorcycles = ', len(m))

def bounding_box(detections):
    confidence_score = []
    ids = []
    cordinates = []

    for i in detections:
        for j in i:
            probs_values = j[5:]
            class_ = np.argmax(probs_values)
            confidence_ = probs_values[class_]

            if confidence_ > Threshold:
                w, h = int(j[2] * image_size), int(j[3] * image_size)
                x, y = int(j[0] * image_size - w / 2), int(j[1] * image_size - h / 2)
                cordinates.append([x, y, w, h])
                ids.append(class_)
                confidence_score.append(confidence_)
    final_box = cv2.dnn.NMSBoxes(cordinates, confidence_score, Threshold, .6)
    return final_box, cordinates, confidence_score, ids

cap = cv2.VideoCapture(0)

frame_number = 0
while cap.isOpened():
    res, frame = cap.read()
    if res:
        frame_number += 1

        img = np.zeros([report_size[1], report_size[0], 3], dtype='uint8')
        original_width, original_height = frame.shape[1], frame.shape[0]

        Neural_Network = cv2.dnn.readNetFromDarknet('yolov4.cfg', 'yolov4.weights')
        with open('coco.names', 'r') as k:
            for i in k.readlines():
                classes_names.append(i.strip())

        blob = cv2.dnn.blobFromImage(frame, 1 / 255, (320, 320), True, crop=False)
        Neural_Network.setInput(blob)
        cfg_data = Neural_Network.getLayerNames()
        layer_names = Neural_Network.getUnconnectedOutLayers()
        outputs = [cfg_data[i - 1] for i in layer_names]
        output_data = Neural_Network.forward(outputs)
        final_box, cordinates, confidence_score, ids = bounding_box(output_data)
        predictions(img, frame, final_box, cordinates, confidence_score, ids, original_width / 320, original_height / 320, frame_number)

        cv2.imshow('frames', frame)
        cv2.imshow('report', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
