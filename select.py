import cv2
import os
import tarfile
import caffe
import numpy as np
from PIL import Image
import parse_xml as pXml
import time

def handDetect():

    start = time.clock()
    root='/wzx_ssd/caffe/python/' 
    deploy=root + 'deploy_.prototxt'  
    caffe_model=root + 'VGG_EGO_SSD_300x300_iter_9004.caffemodel'

    net = caffe.Net(deploy,caffe_model,caffe.TEST)

    for xmlNum in range(0,300):

        for i in range(0,6):

            camID='cam'+str(i)
            filePath=root+'b8_event_1-300/'+str(xmlNum)+'_event_data_metainfo.xml'
            camDict=pXml.load_xml(filePath)

            imgFile=camDict['file_list'][camID]['name'].replace('.tar.gz','')
            rstPath=root+'b8_hand_1-300/'+imgFile

            isExists=os.path.exists(rstPath)
            if not isExists:
               os.makedirs(rstPath) 

            tarPath=root+'b8_event_1-300/'+camDict['file_list'][camID]['name']
            print(tarPath)
            isExists=os.path.exists(tarPath)
            if isExists:
                tar = tarfile.open(tarPath)
                names = tar.getnames()
                for name in names:
                    tar.extract(name,path=root+'b8_event_1-300/')
                tar.close()

            flag=1
            # caffe.set_mode_gpu()
            # caffe.set_device(0)
            transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
            transformer.set_transpose('data', (2,0,1))
            transformer.set_mean('data', np.array([104, 117, 123]))
            transformer.set_raw_scale('data', 255)
            transformer.set_channel_swap('data', (2,1,0))

            selectIndex=0
            #print(camDict['frame_list'][camID]['img'])
            #print(imgFile)
            
           # end = time.clock()
           # print end-start
            for file in camDict['frame_list'][camID]['img']:
               
                imgRoot=root+'b8_event_1-300/'+imgFile+'/'+file
                imSrc=Image.open(imgRoot)
                
                im=caffe.io.load_image(imgRoot)
                net.blobs['data'].data[...] = transformer.preprocess('data',im)
                
                out = net.forward()
                
                # end = time.clock()
                # print end-start
                prob= net.blobs['detection_out'].data[0]
                img=cv2.imread(imgRoot)

                if prob[0,0,2]>0.4: 
                    # a=prob[0,0,:]
                    # x_min=img.shape[1]*a[3]
                    # x_max=img.shape[1]*a[5]
                    # y_min=img.shape[0]*a[4]
                    # y_max=img.shape[0]*a[6]

                    # cv2.rectangle(img,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(0,0,255),3)
               
                    # if prob[0,1,2]>0.4 and prob[0,0,2]-prob[0,1,2]<0.1:
                    #     if prob[0,1,3]>prob[0,0,3] and prob[0,1,4]<prob[0,0,4] and prob[0,1,5]>prob[0,0,5] and prob[0,1,6]<prob[0,0,6]:
                    #         handNum=1
                    #         print(imgRoot,handNum,prob[0,0,2],prob[0,1,2])
                    #     else:    
                    #         handNum=2
                    #         print(imgRoot,handNum,prob[0,0,2],prob[0,1,2])
                    #         a=prob[0,1,:]
                    #         x_min=img.shape[1]*a[3]
                    #         x_max=img.shape[1]*a[5]
                    #         y_min=img.shape[0]*a[4]
                    #         y_max=img.shape[0]*a[6]
                    #         cv2.rectangle(img,(int(x_min),int(y_min)),(int(x_max),int(y_max)),(255,0,0),3)
                
                    # else:
                    #     handNum=1
                    #     print(imgRoot,handNum,prob[0,0,2],prob[0,1,2])
                    
                    if selectIndex % 4 == 1:
                        txtPath='b8_hand_1-300/'+imgFile+'result.txt'
                        f=open(txtPath,'a+')
                        f.write(file+' '+str(handNum)+'\n')
                        f.close()
                        cv2.imwrite('b8_hand_1-300/'+imgFile+'/'+file,img)
                        print('select')
                        #break
                
                # cv2.imwrite('b8_hand_1-300/'+imgFile+'/'+file,img)
                else: 
                    handNum=0
                    print(imgRoot,handNum,prob[0,0,2],prob[0,1,2])
                
                selectIndex += 1
                # txtPath='b8_hand_1-300/'+imgFile+'/'+'result.txt'
                # if os.path.exists(txtPath) and flag:
                #     os.remove(txtPath)
                #     flag=0

                # f=open(txtPath,'a+')
                # f.write(file+' '+str(handNum)+'\n')
                # f.close()
                
                # cv2.imwrite('b8_hand_1-300/'+imgFile+'/'+file,img)

if __name__ == '__main__':
    import sys
    handDetect()
