import os
import numpy as np
import cv2

def smaller_plz(img, d=8):
    return cv2.resize(img, (img.shape[1]//d, img.shape[0]//d), cv2.INTER_AREA) # 因為顯示關係

def rotatedDice(image, cnt):
    # 取得最小擬合橢圓並對圖像做翻轉
    ellipse = cv2.fitEllipse(cnt)
    (center, axes, angle) = ellipse
    angle = angle - 90
    rotation_matrix = cv2.getRotationMatrix2D(tuple(center), angle, 1)
    image = cv2.warpAffine(image, rotation_matrix,(image.shape[1], image.shape[0]))
    # 計算裁切位置
    mark = np.zeros_like(image)
    cv2.drawContours(mark, [cnt], 0, (255, 255, 255), -1)
    mark = cv2.warpAffine(mark, rotation_matrix,(mark.shape[1], mark.shape[0]))
    mark = cv2.cvtColor(mark, cv2.COLOR_RGB2GRAY)
    cnts, hier = cv2.findContours(mark, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(cnts[0])
    matting_result = image[y:y+h,x:x+w,:]
    return matting_result

names = os.listdir("tilt_plate_my")
for name in names:
    img = cv2.imread("tilt_plate_my\\"+name)
    img = cv2.resize(img,(640,360))

    h,w,_ = img.shape
    print("h:w=",h,w)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.equalizeHist(gray)
    
    filt1 = np.array([  [0, 0, 0],
                        [1, 0, -1],
                        [0, 0, 0] ])
    filt2 = np.array([  [0, 0, 0],
                        [-1, 0, 1],
                        [0, 0, 0] ])
    conv1 = cv2.filter2D(hist, -1, filt1)
    conv2 = cv2.filter2D(hist, -1, filt2)
    conv = conv1+conv2
    
    smth = cv2.medianBlur(conv, 3)
    ret, thres = cv2.threshold(smth, 100, 255, cv2.THRESH_BINARY)
    blur_thres = cv2.medianBlur(thres,5)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    kernel1 = np.ones((3,3),np.uint8)
    dilats1 = cv2.dilate(blur_thres, kernel1, iterations=3) # 膨脹
    # blur = cv2.medianBlur(dilats1,5)
    # kernel_r = np.array([
    #     [0, 0, 0],
    #     [1, 1, 1],
    #     [0, 0, 0]], np.uint8)
    # opened_r = cv2.morphologyEx(dilats1, cv2.MORPH_OPEN, kernel_r, iterations=3)
    # kernel_c = np.array([
    #     [0, 1, 0],
    #     [0, 1, 0],
    #     [0, 1, 0]], np.uint8)
    # opened_c = cv2.morphologyEx(dilats1, cv2.MORPH_OPEN, kernel_c, iterations=3)
    # opened = opened_r + opened_c
    (cnts ,_) = cv2.findContours(dilats1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = img.copy()
    if len(cnts) != 0:
        # find the biggest countour (c) by the area
        c = max(cnts, key = cv2.contourArea)
        # cv2.drawContours(output, [c], -1, (0,0,255), 2)
        # draw the biggest contour (c) in green
        
        x,y,ww,hh = cv2.boundingRect(c)
        cv2.rectangle(output,(x,y),(x+ww,y+hh),(0,255,0),2)
        
        # ellipse = cv2.fitEllipse(c)
        # (center, axes, angle) = ellipse
        # cv2.ellipse(output, ellipse, (0, 255, 255), 2)
        
        
        
        
        # [[vx],[vy],[x],[y]] = cv2.fitLine(c, cv2.DIST_L2,0,0.01,0.01)
        # lefty = int((-x*vy/vx) + y)
        # righty = int(((w-x)*vy/vx)+y)
        # cv2.line(output,(w-1,righty),(0,lefty),(0,255,0),4)

        
        
        hull = cv2.convexHull(c)
        cv2.drawContours(output, [hull], -1, (255,0,0), 2)
        
        h_mask = np.zeros(thres.shape, np.uint8)
        cv2.drawContours(h_mask, [hull], -1, (255,255,255), -1)
        thres_masked = cv2.bitwise_and(thres, h_mask)
        
        # rotated = rotatedDice(cv2.cvtColor(c_mask.copy(),cv2.COLOR_GRAY2BGR),c)
        
        c_mask = h_mask
        dst = cv2.cornerHarris(c_mask,2,3,0.04)
        #result is dilated for marking the corners, not important
        dst = cv2.dilate(dst,None)
        # Threshold for an optimal value, it may vary depending on the image.
        output[dst>0.01*dst.max()]=[0,0,255]
        # cv2.imshow('o',output)
        
        
        
        # perimeter = cv2.arcLength(hull, True)
        # approx = cv2.approxPolyDP(hull, 0.05 * perimeter, True)
        # # drawing points
        # for point in approx:
        #     x, y = point[0]
        #     cv2.circle(output, (x, y), 3, (255, 0, 0), -1)
        # # drawing skewed rectangle
        # cv2.drawContours(output, [approx], -1, (255, 0, 0))
        
        
        # rect = cv2.minAreaRect(c)
        # print("中心坐标:", rect[0])
        # print("宽度:", rect[1][0])
        # print("长度:", rect[1][1])
        # print("旋转角度:", rect[2])
        # box = cv2.boxPoints(rect)
        # box = np.int0(box)
        # print("四个顶点坐标为;", box)
        # cv2.drawContours(output, [box], -1, (255,0,0), 2)   
    
    cv2.imshow("gray",smaller_plz(gray, 1))
    cv2.imshow("hist",smaller_plz(hist, 1))
    # cv2.imshow("conv2",smaller_plz(conv2))
    cv2.imshow("conv",smaller_plz(conv, 1))
    cv2.imshow("smth",smaller_plz(smth, 1))
    cv2.imshow("thres",smaller_plz(thres, 1))
    cv2.imshow("b_thres",smaller_plz(blur_thres, 1))
    cv2.imshow("dilate",smaller_plz(dilats1, 1))
    cv2.imshow("output",smaller_plz(output, 1))
    cv2.imshow("c_mask",smaller_plz(c_mask, 1))
    cv2.imshow("h_mask",smaller_plz(h_mask, 1))
    cv2.imshow("thres_masked",smaller_plz(thres_masked, 1))
    # cv2.imshow("rotated",smaller_plz(rotated, 1))
    # cv2.imshow("lines",smaller_plz(linesimg, 1))

    # cv2.imshow("open",smaller_plz(opened, 1))
    # cv2.imshow("masked2",smaller_plz(masked2))
    
    cv2.waitKey(0)
    