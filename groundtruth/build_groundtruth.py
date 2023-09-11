import os

dirs = ["groundtruth_641","groundtruth_2411"]

for d in dirs:
    os.chdir(d)
    lst = os.listdir()
    datas = []
    for i in range(1,len(lst)//2+1):
        print(i)
        with open(str(i)+".txt","r") as f:
            r = f.read().split(" ")
            r[0] = '1' if len(r)>1 else '0'
            # print(r)
            datas.append(" ".join(r)+'\n')
    os.chdir("..")
    with open(d+".txt","w") as f:
        f.writelines(datas)