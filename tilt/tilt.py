import cv2
import os
import numpy as np
from matplotlib import pyplot as plt

def shi_tomashi(image):
    """
    Use Shi-Tomashi algorithm to detect corners
    Args:
        image: np.array
    Returns:
        corners: list
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, 4, 0.01, 100)
    corners = np.int0(corners)
    corners = sorted(np.concatenate(corners).tolist())
    print('\nThe corner points are...\n')

    im = image.copy()
    for index, c in enumerate(corners):
        x, y = c
        cv2.circle(im, (x, y), 30, 255, -1)
        character = chr(65 + index)
        print(character, ':', c)
        cv2.putText(im, character, tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 20, (0, 255, 255), 2, cv2.LINE_AA)

    plt.imshow(im)
    plt.title('Corner Detection: Shi-Tomashi')
    plt.show()
    return corners

os.chdir("tilt_plate_center")
names = os.listdir()
for name in names:
    img = cv2.imread(name)
    
    h,w,_ = img.shape
    # cv2.line(img,(int(w*3/4),0),(int(w*3/4),h),(0,0,255),8)
    # cv2.line(img,(int(w*1/4),0),(int(w*1/4),h),(0,0,255),8)
    
    v = h//3
    # L = cv2.getPerspectiveTransform(np.float32([(0,0),(w*1/2,0),(w*1/2,h),(0,h)]),np.float32([[0,0],[w,0],[w,h],[0,h]]))
    # M = cv2.getPerspectiveTransform(np.float32([(w*1/4,0),(w*3/4,0),(w*3/4,h),(w*1/4,h)]),np.float32([[0,0],[w,0],[w,h],[0,h]]))
    # R = cv2.getPerspectiveTransform(np.float32([(w*1/2,0),(w,0),(w,h),(w*1/2,h)]),np.float32([[0,0],[w,0],[w,h],[0,h]]))
    # Wider = cv2.getPerspectiveTransform(np.float32([[0,0],[w,0],[w,h],[0,h]]),np.float32([[0,0],[w*3,0],[w*3,h],[0,h]]))
    L = cv2.getPerspectiveTransform(np.float32([(w*1/4,v),(w*3/4,-v),(w*3/4,h-v),(w*1/4,h+v)]),np.float32([[0,0],[w,0],[w,h],[0,h]]))
    R = cv2.getPerspectiveTransform(np.float32([(w*1/4,-v),(w*3/4,v),(w*3/4,h+v),(w*1/4,h-v)]),np.float32([[0,0],[w,0],[w,h],[0,h]]))

    
    # new_right = cv2.warpPerspective(img.copy(),R,(w,h),flags=cv2.INTER_LINEAR)
    # new_left = cv2.warpPerspective(img.copy(),L,(w,h),flags=cv2.INTER_LINEAR)
    # new_mid = cv2.warpPerspective(img.copy(),M,(w,h),flags=cv2.INTER_LINEAR)
    # new_wider = cv2.warpPerspective(img.copy(),Wider,(w*3,h),flags=cv2.INTER_LINEAR)
    new_right = cv2.warpPerspective(img.copy(),R,(w,h),flags=cv2.INTER_LINEAR)
    new_left = cv2.warpPerspective(img.copy(),L,(w,h),flags=cv2.INTER_LINEAR)
    
    img = cv2.resize(img, (w//4, h//4), cv2.INTER_AREA)
    new_right = cv2.resize(new_right, (w//4, h//4), cv2.INTER_AREA)
    new_left = cv2.resize(new_left, (w//4, h//4), cv2.INTER_AREA)
    # new_mid = cv2.resize(new_mid, (w//8, h//8), cv2.INTER_AREA)
    # new_wider = cv2.resize(new_wider, (w*3//8, h//8), cv2.INTER_AREA)
    cv2.imshow("img",img)
    cv2.imshow("right",new_right)
    cv2.imshow("left",new_left)
    # cv2.imshow("mid",new_mid)
    # cv2.imshow("wider",new_wider)
    
    
    cv2.waitKey(0)
    # cv2.destroyAllWindows()
    

os.chdir("..")
    
    
    