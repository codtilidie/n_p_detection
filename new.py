import cv2
import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
import easyocr
import os
from ultralytics import YOLO
reader = easyocr.Reader(['en'], gpu = False)
detection = []



# img = cv2.imread('C:/Users/ashis/NP_detection_arv/ARV/cropped_img_02.jpg')

# # scale_percent = 500  # percent of original size
# # width = int(img.shape[1] * scale_percent / 100)
# # height = int(img.shape[0] * scale_percent / 100)
# # dim = (width, height)
# # img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # thres = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]
# thres = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 9)

# contours ,img_cont = cv2.findContours(thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# centroid_contours = []
# for contour in contours:
#     # Calculate the centroid using the moments
#     M = cv2.moments(contour)
#     if M["m00"] != 0:
#         cX = int(M["m10"] / M["m00"])
#         cY = int(M["m01"] / M["m00"])
#         centroid_contours.append((cX, cY, contour))
        
# sorted_contours = sorted(centroid_contours, key=lambda x: (x[1], x[0]))
# sorted_contours = [contour[2] for contour in sorted_contours]


# Sort contours based on centroid coordinates (top-left to bottom-right)
# print(img.shape)
# for cnt in sorted_contours:
#     cv2.drawContours(img, cnt, -1, (0, 255, 0), 1)
#     plate = img.shape
#     (w, h) = cv2.boundingRect(cnt)[2:]
#     plate_char_percentage = ((w * h)/(plate[0] * plate[1]))*100
#     # print('percentage',plate_char_percentage)
#     aspect_ratio = w / float(h)
#     # print(aspect_ratio)
#     # print('area',cv2.contourArea(cnt))
#     # if cv2.contourArea(cnt) >20:
#     if (aspect_ratio > 0.40 and aspect_ratio < 0.8) and (plate_char_percentage > 2 and plate_char_percentage < 6):
#         cv2.drawContours(img, cnt, -1, (0, 255, 0), 1)
#         x1, y1, w, h = cv2.boundingRect(cnt)
#         cv2.rectangle(img, (x1, y1), (x1+w, y1+h), (0, 255, 0),  3)
#         # cv2.imshow('',img)
#         # cv2.waitKey(0)
#         img = cv2.resize(img, (640, 480))
#         cropped_char = img[int(y1-10):int(y1+h+10), int(x1-10):int(x1+w+10)]
#         img_gray = cv2.cvtColor(cropped_char, cv2.COLOR_BGR2GRAY)
#         _, thres = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY_INV)
#         # cv2.imshow('thres', thres)
#         # cv2.waitKey(0)
#         detection.append(reader.readtext(thres))

# points =[]

# for detect in detection:
#     print(detect)
    


# exp = plt.imread('C:/Users/ashis/NP_detection_arv/ARV/cropped_img_2116.jpg')
# plt.imshow(exp)
# plt.show()





# light[light == 0] =1
# light[light == 255] =0

# horizontal_projection = np.sum(light, axis=1)


# height, width = light.shape
# blank_img = np.zeros((height, width, 3), np.uint8)


# for row in range(height):
#     cv2.line(blank_img,(0,row), (int(horizontal_projection[row]*width/height),row), (255,255,255),1)

# cv2.imshow(' ', blank_img)
# cv2.waitKey(0)








# rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
# squareKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))


# img = cv2.imread('C:/Users/ashis/NP_detection_arv/ARV/cropped_img_02.jpg')
# cv2.imshow(' ', img)
# cv2.waitKey(0)

# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
# cv2.imshow('', gray)
# cv2.waitKey(0)

# light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKernel)
# light = cv2.threshold(gray, 145, 255, cv2.THRESH_BINARY)[1]
# light = cv2.adaptiveThreshold(light, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 1)

# cv2.imshow(' ', light)
# cv2.waitKey(0)

# gradX = cv2.Sobel(light, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
# gradX = np.absolute(gradX)
# (minVal, maxVal) = (np.min(gradX), np.max(gradX))
# gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")

# gradX = cv2.GaussianBlur(gradX, (5, 5), 0)
# cv2.imshow(' ', gradX)
# cv2.waitKey(0)

# gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
# thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
# cv2.imshow('', thresh)
# cv2.waitKey(0)








pre_trained_model = YOLO("C:/Users/ashis/NP_DETECTION_ARV/ARV/best.pt")

img = cv2.imread(os.path.join("C:/Users/ashis/NP_DETECTION_ARV/ARV", "IMG_2116.JPG"))

scale_percent = 50  # percent of original size
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)
img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
img = cv2.resize(img, (640, 640))

detections = pre_trained_model(img)[0]
for detection in detections.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = detection
    print(score)
    if(score>0.6):
        # drawn_frame = cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 4)
        # cv2.imshow('frame',drawn_frame)
        # cv2.waitKey(0)

        cropped_img = img[int(y1):int(y2), int(x1):int(x2)]

        cv2.imshow('cropped',cropped_img)
        cv2.waitKey(0)
        cv2.imwrite('C:/Users/ashis/NP_detection_arv/ARV/cropped_img_01.jpg',cropped_img)


    