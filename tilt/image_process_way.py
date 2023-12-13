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

def yt_way(gray):
    # bfilt = cv2.bilateralFilter(gray,11, 11, 17) # noise reduce
    # cv2.imshow("bfilt", bfilt)
    # edged = cv2.Canny(bfilt, 50, 200)
    # cv2.imshow("edged", edged)
    
    # (points,_) = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = sorted(points, key=cv2.contourArea, reverse=True) [:100]
    # location = None
    # for cnt in cnts:
    #     ap = cv2.approxPolyDP(cnt, 10, True)
    #     if len(ap) == 4:
    #         location = ap
    #         break
    # q = img.copy()
    # cv2.drawContours(q, [location], -1, (0,0,255), 2)
    # cv2.imshow("qq", q)
    pass

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
        
        # x,y,ww,hh = cv2.boundingRect(c)
        # cv2.rectangle(output,(x,y),(x+ww,y+hh),(0,255,0),2)
        
        
        hull = cv2.convexHull(c)
        h_mask = np.zeros(thres.shape, np.uint8)
        cv2.drawContours(h_mask, [hull], -1, (255,255,255), -1)
        
        (X, Y), (LAxis, SAxis), Angle = ellipse = cv2.fitEllipse(hull) # angle順時針 0是垂直， 90是水平，180是垂直
        # cv2.ellipse(output,ellipse, (255,0,0), 2)
        if LAxis < SAxis: LAxis, SAxis = SAxis, LAxis
        
        Angle = Angle - 90
        rotation_matrix = cv2.getRotationMatrix2D((X,Y), Angle, 1)
        iii = cv2.warpAffine(h_mask.copy(), rotation_matrix,(w, h))
        iij = cv2.warpAffine(img.copy(), rotation_matrix,(w, h))
        # iii = iii[int(Y-SAxis/2):int(Y+SAxis/2), int(X-LAxis/2):int(X+LAxis/2)]
        cv2.rectangle(iii, (int(X-LAxis/2),int(Y-SAxis/2)), (int(X+LAxis/2),int(Y+SAxis/2)), (255, 255, 255), 2)
        
        print((X, Y), (LAxis, SAxis), Angle)
        
        # rect = cv2.RotatedRect((X, Y), (LAxis, SAxis), Angle)
        # box = cv2.boxPoints(rect) # cv2.cv.BoxPoints(rect) for OpenCV <3.x
        # pt0, pt1, pt2, pt3 = box.tolist()
        # print(pt0, pt1, pt2, pt3)
        # lpt, rpt, dpt = [], [], [0,0] # 左點 右典 下中點
        # for i in range(2):
        #     lpt.append(int((pt0[i] + pt3[i])/2))
        #     rpt.append(int((pt1[i] + pt2[i])/2))
        # cv2.circle(output, lpt, 0, (0, 0, 255), 5)
        # cv2.circle(output, rpt, 0, (0, 0, 255), 5)
        
        # if Angle > 90:
        #     dpt[0] = rpt[0]
        #     dpt[1] = lpt[1]
        # else:
        #     dpt[0] = lpt[0]
        #     dpt[1] = rpt[1]
        # dpt = [(lpt[0]+rpt[0])//2, min(lpt[1], rpt[1])]
            
        # cv2.circle(output, dpt, 0, (0, 255, 0), 5)
        
        # vw = w//8
        # vh = h//8
        # cx = w//2
        # cy = h//2
        # dx = cx # +vw if Angle > 90 else cx-vw
        # dy = cy-vh//2 # if Angle > 90 else 
        # p1 = np.float32([lpt, rpt, dpt])
        # p2 = np.float32([[cx-vw, cy], [cx+vw, cy], [dx, dy]])
        # M = cv2.getAffineTransform(p1, p2)
        # output2 = cv2.warpAffine(output, M, (w, h))
        # cv2.ellipse(output, ellipse, (255,0,0), 2)
        
        
        
        
        
        
        # [[vx],[vy],[x],[y]] = cv2.fitLine(c, cv2.DIST_L2,0,0.01,0.01)
        # lefty = int((-x*vy/vx) + y)
        # righty = int(((w-x)*vy/vx)+y)
        # cv2.line(output,(w-1,righty),(0,lefty),(0,255,0),4)

        
        
        # hull = cv2.convexHull(c)
        # cv2.drawContours(output, [hull], -1, (255,0,0), 2)
        
        
        # h_mask = np.zeros(thres.shape, np.uint8)
        # cv2.drawContours(h_mask, [hull], -1, (255,255,255), -1)
        # thres_masked = cv2.bitwise_and(hist, h_mask)
        # thres_masked += ~h_mask
        # ret, bin_masked = cv2.threshold(thres_masked, 150, 255, cv2.THRESH_BINARY_INV)
        # (cnts ,_) = cv2.findContours(bin_masked, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # if len(cnts) > 10:
        #     cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
        
        # def greater(a):
        #     x, y, w, h = cv2.boundingRect(c)
        #     return x
            
        # cnts = sorted(cnts, key=greater)
        # leftmost = cnts[0]
        # rightmost = cnts[-1]
        # nnnn_mask = np.zeros(thres.shape, np.uint8)
        # cv2.drawContours(nnnn_mask, [leftmost], -1, (255,255,0), -1)
        # cv2.drawContours(nnnn_mask, [rightmost], -1, (255,0,0), -1)
        
        
        
        
        
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
    # cv2.imshow("output2",smaller_plz(output2, 1))
    # cv2.imshow("c_mask",smaller_plz(c_mask, 1))
    cv2.imshow("h_mask",smaller_plz(h_mask, 1))
    # cv2.imshow("th_masked",smaller_plz(thres_masked, 1))
    # cv2.imshow("bin_masked",smaller_plz(bin_masked, 1))
    # cv2.imshow("nnn_masked",smaller_plz(nnnn_mask, 1))
    cv2.imshow("iii",smaller_plz(iii, 1))
    cv2.imshow("iij",smaller_plz(iij, 1))
    
    
    yt_way(gray)
    # cv2.imshow("qq",qq)
    # cv2.imshow("rotated",smaller_plz(rotated, 1))
    # cv2.imshow("lines",smaller_plz(linesimg, 1))

    # cv2.imshow("open",smaller_plz(opened, 1))
    # cv2.imshow("masked2",smaller_plz(masked2))
    
    cv2.waitKey(0)
    