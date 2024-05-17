from ultralytics import YOLO
import cv2
import os
import easyocr

reader = easyocr.Reader(['en'], gpu = False)
detection = []



def recognize_char(region, org_img):
    cropped_char = org_img[int(region[0]-3):int(region[1]+3), int(region[2]-3):int(region[3]+3)]
    img_gray = cv2.cvtColor(cropped_char, cv2.COLOR_BGR2GRAY)
    _, thres = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('v', thres)          
    cv2.waitKey(0)
    detection.append(reader.readtext(thres))

    for detect in detection:
        print(detect)
    

def segmentcharacter(img, org_img, class_id):
    contors ,_ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if class_id == 1:
        centroid_contours = []
        for contour in contors:            
        # Calculate the centroid using the moments
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                centroid_contours.append((cX, cY, contour))
            
        sorted_contours = sorted(centroid_contours, key=lambda x: (x[1], x[0]))
        contors = [contour[2] for contour in sorted_contours]
                
    for cnt in contors:
        plate = img.shape
        (w, h) = cv2.boundingRect(cnt)[2:]
        plate_char_percentage = ((w * h)/(plate[0] * plate[1]))*100
        aspect_ratio = w / float(h)
        
        if (aspect_ratio > 0.40 and aspect_ratio < 0.8) and (plate_char_percentage > 2 and plate_char_percentage < 6):
            x1, y1, w, h = cv2.boundingRect(cnt)
            # cv2.rectangle(org_img, (x1, y1), (x1+w, y1+h), (0, 255, 0),  3)
            cv2.imshow('seg',org_img)
            cv2.waitKey(0)
            region = [y1, y1+h, x1, x1+w]
            recognize_char(region,org_img)

def filter_img(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thres = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 8)
    cv2.imshow('', thres)
    cv2.waitKey(0)
    
    segmentcharacter(thres, img, region[4])


def crop_img(region, img):
    cropped_img = img[int(region[0]):int(region[1]), int(region[2]):int(region[3])]
    cv2.imshow('', cropped_img)
    cv2.waitKey(0)
    filter_img(cropped_img)


#loading the costum_model
pre_trained_model = YOLO("C:/Users/ashis/NP_DETECTION_ARV/ARV/best.pt")

#providing the path of the image or video source
img = cv2.imread(os.path.join("C:/Users/ashis/NP_DETECTION_ARV/ARV", "IMG_2377.JPG"))
detections = pre_trained_model(img)[0]

for detection in detections.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = detection
    if(score>0.6):
        region = [y1, y2, x1, x2, int(class_id)]
        crop_img(region, img)

    

    