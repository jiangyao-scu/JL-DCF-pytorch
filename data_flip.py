import os
import cv2

path = 'D:/work/python/RGBDcollection/'

def main():
    file_list = os.listdir(path+'GT/')
    for i, GT_name in enumerate(file_list):
        # print(i,GT_name)
        name = GT_name.split('GT')[0]
        depth_name = name + 'Depth.png'
        LR_name = name + 'ori.png'
        print(i, GT_name, depth_name, LR_name)
        GT = cv2.imread(path+'GT/'+GT_name)
        Depth = cv2.imread(path + 'depth/' + depth_name)
        LR = cv2.imread(path + 'LR/' + LR_name)
        FLIP(GT,Depth,LR,name)


def FLIP(GT,Depth,LR,name):
    times = [0,5,10,20]
    h = GT.shape[0]
    w = GT.shape[1]
    for i in times:
        flip(GT,Depth,LR,i,h,w,name)

def flip(GT,Depth,LR,locate,h,w,name):
    location = 'D:/work/python/RGBDCollection_e/'
    GT = cv2.resize(GT,(320,320))
    Depth = cv2.resize(Depth, (320, 320))
    LR = cv2.resize(LR, (320, 320))
    GT = GT[locate:300+locate,locate:300+locate,:]
    Depth = Depth[locate:300 + locate, locate:300 + locate, :]
    LR = LR[locate:300 + locate, locate:300 + locate]
    GT = cv2.resize(GT, (h, w))
    Depth = cv2.resize(Depth, (h, w))
    LR = cv2.resize(LR, (h, w))
    # cv2.imwrite(GT,location + name+'_'+locate+'_GT.png')
    # cv2.imwrite(Depth, location + name + '_' + locate + '_Depth.png')
    # cv2.imwrite(LR, location + name + '_' + locate + '_ori.png')

if __name__ == '__main__':
    main()