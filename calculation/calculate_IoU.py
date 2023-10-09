import os
import cv2

# https://www.wongwonggoods.com/all-posts/python/python_opencv/python-opencv-iou/
def get_iou(bbox_ai, bbox_gt):
    iou_x = max(bbox_ai[0], bbox_gt[0]) # x
    iou_y = max(bbox_ai[1], bbox_gt[1]) # y
    iou_w = min(bbox_ai[2]+bbox_ai[0], bbox_gt[2]+bbox_gt[0]) - iou_x # w
    iou_w = max(iou_w, 0)
    print(f'{iou_w=}')
    iou_h = min(bbox_ai[3]+bbox_ai[1], bbox_gt[3]+bbox_gt[1]) - iou_y # h
    iou_h = max(iou_h, 0)
    print(f'{iou_h=}')

    iou_area = iou_w * iou_h
    print(f'{iou_area=}')
    all_area = bbox_ai[2]*bbox_ai[3] + bbox_gt[2]*bbox_gt[3] - iou_area
    print(f'{all_area=}')

    if all_area == 0: return 0
    return max(iou_area/all_area, 0)

TP, FP, TN, FN = 0,0,0,0
# video_num = 2 # 影片編號
cpp_ver = "org" # org wide thres 程式版本
model_ver = "m6" # m6, old 模型版本

for video_num in [1,2]:
    imgdir = f"gt_img_{video_num}" # 存圖片的資料夾
    gt_txt = f"groundtruth_xywh_{video_num}.txt" # 由yolo轉格式的groundtruth
    ai_txt = f"detected//detected_{cpp_ver}_{model_ver}_{video_num}.txt" # 各AI模型產生的預測結果
    gts = []
    ais = []

    # idx 為 0 ~ 640
    with open(gt_txt, "r") as f:
        gts = list(map(lambda i:list(map(int,i[:-1].split(" "))),f.readlines())) # 讀入後將每行先去除換行[:-1]，再用空白切分split，並map到int
    with open(ai_txt, "r") as f:
        ais = list(map(lambda i:list(map(int,i[:-1].split(" "))),f.readlines()))

    for i in range(len(gts)):
        # i+1.jpg
        # img = cv2.imread(f"{imgdir}//{i+1}.jpg")
        has_GT = gts[i][0] # 此圖有groundtruth
        has_AI = ais[i][0] # 此圖AI預測有結果
        gt_rect = gts[i][1:] if has_GT else [0,0,0,0]
        ai_rect = ais[i][1:] if has_AI else [0,0,0,0]
        print(gt_rect, ai_rect)
        
        # x, y, w, h = gt_rect
        # cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),1)
        # x, y, w, h = ai_rect
        # cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),1)
        
        iou = get_iou(ai_rect, gt_rect)
        if has_AI: # positive
            if iou >= 0.5: TP += 1 # true
            else: FP += 1 # false
        else: #negative
            if has_GT: FN += 1 # false
            else: TN += 1 # true
        
        # resized = cv2.resize(img, (img.shape[1]//2,img.shape[0]//2), interpolation = cv2.INTER_AREA)
        # cv2.imshow(str(i+1)+".jpg", resized)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

precision = TP / (TP + FP) # 確實有車牌/預測有車牌
recall = TP / (TP + FN) # 預測出的車牌/GroundTruth車牌數
print(f"TP:{TP} FP:{FP} FN:{FN} TN:{TN}")
print("precision: ", precision)
print("recall: ",recall)