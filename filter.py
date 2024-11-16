import cv2


img = cv2.imread('C:/Users/ashis/NP_detection_arv/ARV/cropped_img_2116.jpg')

# scale_percent = 500  # percent of original size
# width = int(img.shape[1] * scale_percent / 100)
# height = int(img.shape[0] * scale_percent / 100)
# dim = (width, height)
# img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# thres = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)[1]
thres = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 13, 9)

contours ,img_cont = cv2.findContours(thres, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
centroid_contours = []
for contour in contours:
    # Calculate the centroid using the moments
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        centroid_contours.append((cX, cY, contour))
        
sorted_contours = sorted(centroid_contours, key=lambda x: (x[1], x[0]))
sorted_contours = [contour[2] for contour in sorted_contours]


#Sort contours based on centroid coordinates (top-left to bottom-right)
for cnt in sorted_contours:
    cv2.drawContours(img, cnt, -1, (0, 255, 0), 1)
    plate = img.shape
    (w, h) = cv2.boundingRect(cnt)[2:]
    plate_char_percentage = ((w * h)/(plate[0] * plate[1]))*100
    # print('percentage',plate_char_percentage)
    aspect_ratio = w / float(h)
    # print(aspect_ratio)
    # print('area',cv2.contourArea(cnt))
    # if cv2.contourArea(cnt) >20:
    if (aspect_ratio > 0.40 and aspect_ratio < 0.8) and (plate_char_percentage > 2 and plate_char_percentage < 6):
        cv2.drawContours(img, cnt, -1, (0, 255, 0), 1)
        x1, y1, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(img, (x1, y1), (x1+w, y1+h), (0, 255, 0),  3)
        cv2.imshow('reeya',img)
        cv2.waitKey(0)
        img = cv2.resize(img, (640, 480))
        cropped_char = img[int(y1-10):int(y1+h+10), int(x1-10):int(x1+w+10)]
        img_gray = cv2.cvtColor(cropped_char, cv2.COLOR_BGR2GRAY)
        _, thres = cv2.threshold(img_gray, 80, 255, cv2.THRESH_BINARY_INV)
        cv2.imshow('thres', thres)
        cv2.waitKey(0)
