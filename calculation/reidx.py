import os
dirs = ["groundtruth_641","groundtruth_2411"]

for d in dirs:
    os.chdir(d)
    lst = os.listdir()
    for filename in lst:
        if  not filename.endswith(".jpg"):
            continue
        filename = filename[:-4]
        newidx = filename.split("_")[1][1:]
        os.rename(filename+".jpg",str(newidx)+".jpg")
        os.rename(filename+".txt",str(newidx)+".txt")
    os.chdir("..")
    
    

