import cv2
import os
from matplotlib import pyplot as plt
import numpy as np
import math


def findIntersection(line1,line2):
    x1,y1 = line1[0], line1[1]
    x2,y2 = line1[2], line1[3]
    x3,y3 = line2[0], line2[1]
    x4,y4 = line2[2], line2[3]
    if (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) == 0: # 平行狀況
        return None
    px= ( (x1*y2-y1*x2)*(x3-x4)-(x1-x2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) ) 
    py= ( (x1*y2-y1*x2)*(y3-y4)-(y1-y2)*(x3*y4-y3*x4) ) / ( (x1-x2)*(y3-y4)-(y1-y2)*(x3-x4) )
    return [px, py]

names = os.listdir("images")
for name in names:
    img = cv2.imread(name)
    assert img is not None, "file could not be read, check with os.path.exists()"
    # img = cv2.medianBlur(img,5)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.equalizeHist(gray)
    height = img.shape[0]
    width = img.shape[1]
    pixels = height * width

    # 前處理
    ret,th1 = cv2.threshold(hist,127,255,cv2.THRESH_BINARY)
    th1 = cv2.medianBlur(th1, 5)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    # th1 = cv2.morphologyEx(th1, cv2.MORPH_DILATE, kernel)
    bin2 =cv2.adaptiveThreshold(th1, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    can = cv2.Canny(th1,75,150)
    
    # 輸出
    cv2.imshow("Original Img",img)
    cv2.imshow("hist",hist)
    cv2.imshow("th1",th1)
    cv2.imshow("can",can)
    cv2.imshow("bin2",bin2)
    
    # 後面輸出用
    test_hough1 = img.copy()
    test_hough2 = img.copy()
    test_hough3 = img.copy()
    test_hough4 = img.copy()
    
    # lines = cv2.HoughLines(can, 1, np.pi / 180, 150, None, 0, 0)
    # print(len(lines))
    # if lines is not None:
    #     for i in range(0, len(lines)):
    #         rho = lines[i][0][0]
    #         theta = lines[i][0][1]
    #         a = math.cos(theta)
    #         b = math.sin(theta)
    #         x0 = a * rho
    #         y0 = b * rho
    #         pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    #         pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
    #         cv2.line(test_hough1, pt1, pt2, (0,0,255), 3, cv2.LINE_AA)
        
    
    linesP = cv2.HoughLinesP(bin2, 1, np.pi / 180, 5, None, 25, 5)
    
    angles = [0, np.pi/2] # 預先加入0,90度 以便分類
    warped = img.copy()
    cv2.putText(warped,"Not Modified",(10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    def lineValid(lines): # 線條有找到、且線條數>5
        if lines is not None:
            if len(lines) >= 5:
                return True
        return False

    if lineValid(linesP):
        for l in linesP: # 計算角度
            cv2.line(test_hough1, (l[0][0], l[0][1]), (l[0][2], l[0][3]), (0, 255, 0), 3, cv2.LINE_AA)
            a = abs(math.atan2(l[0][1]-l[0][3],l[0][2]-l[0][0]))
            # if a < 0: a += 180
            angles.append(a)
        # print(angles)
    
        angles = np.array(angles,dtype=np.float32)
        criteria = ( cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        ret, labels, centers = cv2.kmeans(angles, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        # print(ret, labels, centers)

        idxs = labels[2:] # 去除一開始的0, 90度
        linesH = linesP[idxs.ravel()==0]
        linesV = linesP[idxs.ravel()==1]
        if labels[0][0] == 1:
            linesH, linesV = linesV, linesH
        
        # 依線段長度排序
        linesH = sorted(linesH,key=lambda l: (l[0][0]-l[0][2])**2+(l[0][1]-l[0][3])**2,reverse=True)
        linesV = sorted(linesV,key=lambda l: (l[0][0]-l[0][2])**2+(l[0][1]-l[0][3])**2,reverse=True)
        if len(linesH) > 20:
            linesH = linesH[:len(linesH)//3]
        # if len(linesV) > 5:
        #     linesV = linesV[:int(len(linesV)*0.7)]
        
        # 平行線段們
        all_length = 0
        avg_theta = 0
        cv2.putText(test_hough2,"Horizontal",(10, 20),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
        for i in range(0, len(linesH)):
            l = linesH[i][0]
            cv2.line(test_hough2, (l[0], l[1]), (l[2], l[3]), (0, 255, 0), 3, cv2.LINE_AA)
            length = (l[0]-l[2])**2+(l[1]-l[3])**2
            avg_theta += math.atan2(l[1]-l[3],l[2]-l[0]) * length
            all_length += length
        avg_theta = avg_theta / all_length
        a = math.cos(avg_theta)
        b = math.sin(avg_theta)
        #print(a,b)
        c = width//2
        line_top = [int(c+1000*(a)),int(0+1000*(-b)),int(c-1000*(a)),int(0-1000*(-b))]
        line_dwn = [int(c+1000*(a)),int(height+1000*(-b)),int(c-1000*(a)),int(height-1000*(-b))]
        cv2.line(test_hough3, line_top[:2], line_top[2:], (255, 0, 0), 3, cv2.LINE_AA)
        cv2.line(test_hough3, line_dwn[:2], line_dwn[2:], (255, 0, 0), 3, cv2.LINE_AA)
        
        # 垂直線段們
        all_length = 0
        avg_theta = 0
        cv2.putText(test_hough2,"Vertical",(10, 40),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 1)
        for i in range(0, len(linesV)):
            l = linesV[i][0]
            cv2.line(test_hough2, (l[0], l[1]), (l[2], l[3]), (0, 0, 255), 3, cv2.LINE_AA)
            length = math.sqrt((l[0]-l[2])**2+(l[1]-l[3])**2)
            theta = math.atan2(l[1]-l[3],l[2]-l[0])
            if theta < 0: theta += np.pi # 將-89變成91度
            avg_theta += theta * length
            all_length += length
        avg_theta = avg_theta / all_length
        a = math.cos(avg_theta)
        b = math.sin(avg_theta)
        #print(a,b)
        h = height//2
        line_l = [int(0+1000*(a)),int(h+1000*(-b)),int(0-1000*(a)),int(h-1000*(-b))]
        line_r = [int(width+1000*(a)),int(h+1000*(-b)),int(width-1000*(a)),int(h-1000*(-b))]
        cv2.line(test_hough3, line_l[:2], line_l[2:], (255, 0, 0), 3, cv2.LINE_AA)
        cv2.line(test_hough3, line_r[:2], line_r[2:], (255, 0, 0), 3, cv2.LINE_AA)
        
        # 找四角
        pt1 = findIntersection(line_l,line_top)
        pt2 = findIntersection(line_top,line_r)
        pt3 = findIntersection(line_r,line_dwn)
        pt4 = findIntersection(line_dwn,line_l)
        
        if pt1 is not None and pt2 is not None and pt3 is not None and pt4 is not None:
            M = cv2.getPerspectiveTransform(np.float32([pt1,pt2,pt3,pt4]),np.float32([[0,0],[width,0],[width,height],[0,height]]))
            warped = cv2.warpPerspective(img.copy(),M,(width,height),flags=cv2.INTER_LINEAR)
    
        
    cv2.imshow("Detected Lines - Probabilistic Hough Line", test_hough1)
    cv2.imshow("Line Segmentation", test_hough2)
    cv2.imshow("Draw Lines",test_hough3)

    cv2.imshow("Perspective Transform",warped)
    
    cv2.waitKey(0)
    continue

    # ===== 以下無用 =====


    
    cnts, _ = cv2.findContours(can.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) #approx simple可以找到簡略的邊緣，none則是所有邊緣
    # 去除面積過小的contours (最小的10%)
    cnts = sorted(cnts,key=cv2.contourArea,reverse=True)
    if(len(cnts) > 30):
        cnts = cnts[:int(len(cnts)*0.9)]
    catcnts = np.concatenate(cnts)
    cnt_points = [pt[0] for pt in catcnts]
    
    output = img.copy()
    # cv2.drawContours(output, [c], -1, (0, 255, 0), 3)
    for pt in cnt_points:
        cv2.circle(output, pt, 0, (0, 255, 0), 2)
    
    # (x, y, w, h) = cv2.boundingRect(c)
    # text = "original, num_pts={}".format(len(c))
    # cv2.putText(output, text, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX,
    #     0.9, (0, 255, 0), 2)
    # show the original contour image
    # print("[INFO] {}".format(text))
    cv2.imshow("Original Contour Points", output)
    # cv2.waitKey(0)
    
    # conv = img.copy()
    # hull = cv2.convexHull(np.concatenate(cnts))
    # cv2.drawContours(conv, [hull], -1, (0,255,0), 3)
    # cv2.imshow("convexHull", conv)
    
    # cv2.waitKey(0)
    # continue
    
    # ===== 開始 =====
    # temp_th1 = cv2.transpose(th1)
    # w_pixels = np.argwhere(temp_th1 == 255) #計算所有白色點
    # print(w_pixels)
    # print("w_pixels count:",w_pixels.shape[0])
    Ransac = RANSAC(data=cnt_points, threshold=1., P=.99, S=.9, N=5,img=img,converted=th1,my_thres=20)
    (X, Y), (LAxis, SAxis), Angle = ellipse = Ransac.execute_ransac()
    
    # 將橢圓中心校正為圖片中心
    # (X, Y), (LAxis, SAxis), Angle = ellipse = ((img.shape[1]/2 + X)/2, (img.shape[0]/2 + Y)/2), (LAxis, SAxis), Angle
    
    
    ellp_img = cv2.cvtColor(th1,cv2.COLOR_GRAY2BGR)
    cv2.ellipse(ellp_img, ellipse, (0, 0, 255), 3)
    rect = cv2.RotatedRect((X,Y),(LAxis, SAxis),Angle)
    print("angle=",Angle)
    # new_ellipse = (X, Y), (LAxis, SAxis), 90
    # cv2.ellipse(ellp_img, new_ellipse, (0, 255, 0), 3)
    box = cv2.boxPoints(rect) # cv2.cv.BoxPoints(rect) for OpenCV <3.x
    box = np.intp(box)
    # cv2.drawContours(ellp_img,[box], 0, (0, 0, 255), 2)
    # cv2.circle(ellp_img, (int(X), int(Y)), 0, (0, 255, 0), 2)
    print(box)
    
    # new_ellipse = ((img.shape[1]/2 + X)/2, (img.shape[0]/2 + Y)/2), (LAxis, SAxis), Angle
    # cv2.ellipse(ellp_img, new_ellipse, (0, 255, 0), 3)
    # cv2.putText(ellp_img, "first", (1, 20), cv2.FONT_HERSHEY_SIMPLEX,0.9, (0, 0, 255), 1)
    # cv2.putText(ellp_img, "new", (1, 40), cv2.FONT_HERSHEY_SIMPLEX,0.9, (0, 255, 0), 1)
    cv2.imshow("ellipse", ellp_img)
    
    
    pt0, pt1, pt2, pt3 = box.tolist()
    lpt = [] # 用左上右上的點 找到上方矩形
    rpt = []
    for i in [0,1]:
       lpt.append((pt0[i] + pt3[i])//2)
       rpt.append((pt1[i] + pt2[i])//2)
    top_rect_box = [pt0,pt1,rpt,lpt]
    top_rect = np.intp(top_rect_box)
    dwn_rect_box = [lpt,rpt,pt2,pt3]
    dwn_rect = np.intp(dwn_rect_box)
    # print(box)
    # print(pt0,pt1,pt2,pt3) 左上開始為0點，順時鐘0123，左下為3

    cv2.waitKey(0)
    continue
    
    
    # 用top_rect 對白色點 做mask =============================================
    mask1 = np.zeros(img.shape[:2], dtype="uint8")
    cv2.drawContours(mask1,[top_rect], 0, (255, 255, 255), -1)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, kernel1)
    masked1 = cv2.bitwise_and(can, can, mask=mask1) # 用白點th1 用邊緣can
    mask_img1 = cv2.cvtColor(masked1,cv2.COLOR_GRAY2BGR) # 用來顯示的圖
    cv2.drawContours(mask_img1,[top_rect], 0, (0, 0, 255), 2)
    
    temp = cv2.transpose(masked1)
    inlier_pnts = np.argwhere(temp == 255) #計算在mask內白色點
    line = cv2.fitLine(inlier_pnts, cv2.DIST_L2, 0, 0.01, 0.01) # vx, vy, x0, y0
    line = [ i[0] for i in line.tolist()]
    p1 = [line[0]*1000+line[2], line[1]*1000+line[3]] # x0+vx*100, y0+vy*100
    p2 = [-line[0]*1000+line[2], -line[1]*1000+line[3]] # x0-vx*100, y0-vy*100
    pt = np.intp((p1,p2))
    # print(pt)
    cv2.line(mask_img1,*pt,(0,255,0),3)
    line_img = img.copy()
    cv2.line(line_img,*pt,(0,255,0),3)
    cv2.imshow("masked1 rect & line",mask_img1)
    # cv2.imshow("line1 on ori_img",line_img)
    
    
    # 用top_rect的center做一個新的ellipse 對白色點 做mask =============================================
    # print(top_rect_box)
    ct = [sum([b[0] for b in top_rect_box])//4, sum([b[1] for b in top_rect_box])//4]
    top_ellipse = ct, (LAxis//2, SAxis), Angle
    mask2 = np.zeros(img.shape[:2], dtype="uint8")
    cv2.ellipse(mask2, top_ellipse, (255, 255, 255), -1)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_DILATE, kernel2)
    masked2 = cv2.bitwise_and(can, can, mask=mask2) # 用白點th1 用邊緣can
    mask_img2 = cv2.cvtColor(masked2,cv2.COLOR_GRAY2BGR) # 用來顯示的圖
    cv2.ellipse(mask_img2, top_ellipse, (0, 0, 255), 2)
    
    temp = cv2.transpose(masked2)
    inlier_pnts = np.argwhere(temp == 255) #計算在mask內白色點
    line = cv2.fitLine(inlier_pnts, cv2.DIST_L2, 0, 0.01, 0.01) # vx, vy, x0, y0
    line = [ i[0] for i in line.tolist()]
    p1 = [line[0]*1000+line[2], line[1]*1000+line[3]] # x0+vx*100, y0+vy*100
    p2 = [-line[0]*1000+line[2], -line[1]*1000+line[3]] # x0-vx*100, y0-vy*100
    pt = np.intp((p1,p2))
    # print(pt)
    cv2.line(mask_img2,*pt,(0,255,0),3)
    line_img2 = img.copy()
    cv2.line(line_img2,*pt,(0,255,0),3)
    cv2.imshow("masked2 ellipse & line",mask_img2)
    # cv2.imshow("line2 on ori_img",line_img2)
    
    # 用dwn_rect 對白色點 做mask =============================================
    mask3 = np.zeros(img.shape[:2], dtype="uint8")
    cv2.drawContours(mask3,[dwn_rect], 0, (255, 255, 255), -1)
    # kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (10,10))
    mask3 = cv2.morphologyEx(mask3, cv2.MORPH_DILATE, kernel1)
    masked3 = cv2.bitwise_and(can, can, mask=mask3) # 用白點th1 用邊緣can
    mask_img3 = cv2.cvtColor(masked3,cv2.COLOR_GRAY2BGR) # 用來顯示的圖
    cv2.drawContours(mask_img3,[dwn_rect], 0, (0, 0, 255), 2)
    temp = cv2.transpose(masked3)
    inlier_pnts = np.argwhere(temp == 255) #計算在mask內白色點
    line = cv2.fitLine(inlier_pnts, cv2.DIST_L2, 0, 0.01, 0.01) # vx, vy, x0, y0
    line = [ i[0] for i in line.tolist()]
    p1 = [line[0]*1000+line[2], line[1]*1000+line[3]] # x0+vx*100, y0+vy*100
    p2 = [-line[0]*1000+line[2], -line[1]*1000+line[3]] # x0-vx*100, y0-vy*100
    pt = np.intp((p1,p2))
    # print(pt)
    cv2.line(mask_img3,*pt,(0,255,0),3)
    # line_img = img.copy() # 第一段define過
    cv2.line(line_img,*pt,(0,255,0),3)
    cv2.imshow("masked3 rect & line",mask_img3)
    cv2.imshow("line1,3 on ori_img",line_img)
    
    # 用dwn_rect的center做一個新的ellipse 對白色點 做mask =============================================
    # print(dwn_rect_box)
    ct = [sum([b[0] for b in dwn_rect_box])//4, sum([b[1] for b in dwn_rect_box])//4]
    dwn_ellipse = ct, (LAxis//2, SAxis), Angle
    mask4 = np.zeros(img.shape[:2], dtype="uint8")
    cv2.ellipse(mask4, dwn_ellipse, (255, 255, 255), -1)
    # kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10,10))
    mask4 = cv2.morphologyEx(mask4, cv2.MORPH_DILATE, kernel2)
    masked4 = cv2.bitwise_and(can, can, mask=mask4) # 用白點th1 用邊緣can
    mask_img4 = cv2.cvtColor(masked4,cv2.COLOR_GRAY2BGR) # 用來顯示的圖
    cv2.ellipse(mask_img4, dwn_ellipse, (0, 0, 255), 2)
    
    temp = cv2.transpose(masked4)
    inlier_pnts = np.argwhere(temp == 255) #計算在mask內白色點
    line = cv2.fitLine(inlier_pnts, cv2.DIST_L2, 0, 0.01, 0.01) # vx, vy, x0, y0
    line = [ i[0] for i in line.tolist()]
    p1 = [line[0]*1000+line[2], line[1]*1000+line[3]] # x0+vx*100, y0+vy*100
    p2 = [-line[0]*1000+line[2], -line[1]*1000+line[3]] # x0-vx*100, y0-vy*100
    pt = np.intp((p1,p2))
    # print(pt)
    cv2.line(mask_img4,*pt,(0,255,0),3)
    # line_img2 = img.copy()
    cv2.line(line_img2,*pt,(0,255,0),3)
    cv2.imshow("masked4 ellipse & line",mask_img4)
    cv2.imshow("line2,4 on ori_img",line_img2)

    

    
    cv2.waitKey(0)
    
    # for eps in np.linspace(0.001, 0.05, 10):
    # 	# approximate the contour
    #     peri = cv2.arcLength(c, True)
    #     approx = cv2.approxPolyDP(c, eps * peri, True)
    #     # draw the approximated contour on the image
    #     output = img.copy()
    #     cv2.drawContours(output, [approx], -1, (0, 255, 0), 3)
    #     text = "eps={:.4f}, num_pts={}".format(eps, len(approx))
    #     cv2.putText(output, text, (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX,
    #         0.9, (0, 255, 0), 2)
    #     # show the approximated contour image
    #     print("[INFO] {}".format(text))
    #     cv2.imshow("Approximated Contour", output)
    #     cv2.waitKey(0)
    
    continue
    
    cnts = sorted(cnts,key=cv2.contourArea,reverse=True)
    if(len(cnts) > 30):
        cnts = cnts[:int(len(cnts)*0.9)]
    cnts = np.concatenate(cnts)
    #print(cnts)
    
    
    rect = cv2.minAreaRect(cnts)
    print("中心坐标:", rect[0])
    print("宽度:", rect[1][0])
    print("长度:", rect[1][1])
    print("旋转角度:", rect[2])
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    print("四个顶点坐标为;", box)
    cv2.drawContours(img, [box], -1, (255,0,0), 2)    
    
    hull = cv2.convexHull(cnts)
    cv2.drawContours(img, [hull], -1, (0,255,0), 1)
    

    #check the ratio of the detected plate area to the bounding box
    # if (cv2.contourArea(approx)/(img.shape[0]*img.shape[1]) > .2):
    
    cv2.imshow("box",img)
    
    th2 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
    cv2.THRESH_BINARY,11,2)
    th3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    cv2.THRESH_BINARY,11,2)
    
    dst = cv2.cornerHarris(th1,2,3,0.04)
    img[dst > 0.01 * dst.max()] = [255,0,0]
    # if cv2.countNonZero(th1) < pixels / 2: # 白色部分 < 一半
    #     th1 = cv2.bitwise_not(th1)
    # if cv2.countNonZero(th2) < pixels / 2: # 白色部分 < 一半
    #     th1 = cv2.bitwise_not(th2)
    # if cv2.countNonZero(th3) < pixels / 2: # 白色部分 < 一半
    #     th1 = cv2.bitwise_not(th3)
        
    # element = cv2.getStructuringElement(cv2.MORPH_RECT,(15,8))
    # th1 = cv2.dilate(th1,element)
    
    titles = ['Original Image', 'Global Thresholding (v = 127)',
    'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
    images = [img, th1, th2, th3]
    for i in range(4):
        plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()