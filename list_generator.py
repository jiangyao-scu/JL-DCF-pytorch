import os

outer_path = r'D:\pytorch\JL-DCF\data\RGBDcollection_crop'
depth_path = 'depth'
gt_path = 'GT'

def test():
    folderlist = os.listdir(os.path.join(outer_path ,depth_path))
    listText = open(os.path.join(outer_path,'test.lst'), 'a')
    for file in folderlist:
        _,ex = os.path.splitext(file)
        if ex=='.jpg' or ex=='.png' or ex=='.bmp':
            path = file
            path = "RGB/"+path.replace("_Depth.png","_ori.jpg")+' '+"depth/"+path +'\n'
            #path = "RGB/" + path.replace(".bmp", ".jpg") + ' ' + "RGB/" + path.replace(".bmp" , ".jpg")+ '\n'
            listText.write(path)

def train():
    folderlist = os.listdir(os.path.join(outer_path,depth_path))
    listText = open(os.path.join(outer_path,'train.lst'), 'a')
    for file in folderlist:
        _,ex = os.path.splitext(file)
        if ex=='.jpg' or ex=='.png':
            path = file
            path = "LR/"+path.replace("Depth.png","ori.jpg")+' '+"depth/"+path + ' ' + "GT/"+path.replace('Depth','GT') +'\n'
            listText.write(path)


if __name__ == '__main__':
    train()
