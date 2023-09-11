import os

dw = 1920
dh = 1080

with open("groundtruth_yolo_2.txt","r") as f:
    lines = f.readlines()

for i in range(len(lines)):
    l = lines[i][:-1] #去掉換行
    ls = l.split(" ")
    if len(ls) > 1:
        _, x, y, w, h = map(float,ls)
        nx = int((x - w / 2) * dw) # xmin
        ny = int((y - h / 2) * dh) # ymin
        nw = int(w * dw) # width
        nh = int(h * dh) # height
        lines[i] = f"1 {nx} {ny} {nw} {nh}\n"
    else:
        lines[i] = "0\n"

with open("groundtruth_xywh_2.txt","w") as f:
    f.writelines(lines)
    