import torch
import numpy as np
import pandas as pd
import cv2
import os, sys
import glob
import colorsys
import traceback

class ConvertToKNetFormat():

    def __init__(self, seqName = None, mode = None, FilePath = None, output_dir = None, image_dir = None, metaInfoDir = None):

        assert mode  in [None, "raw", "gt"], "Not valid mode. Value has to be None, 'raw', or 'gt'"

        self.seqName = seqName
        self.mode = mode
        self.image_dir= image_dir
        self.output_dir = output_dir
        self.FilePath = FilePath
        self.metaInfoDir = metaInfoDir


    def loadData(self):
        data = np.genfromtxt(self.FilePath, delimiter=',')
        if data.ndim == 1:  # Because in MOT we have different delimites in result files?!?!?!?!?!?
            data = np.genfromtxt(self.FilePath, delimiter=' ')
        if data.ndim == 1:  # Because
            print("Ooops, cant parse %s, skipping this one ... " % self.FilePath)

            return None
        # clean nan from results
        #data = data[~np.isnan(data)]
        nan_index = np.sum(np.isnan(data), axis = 1)
        data = data[nan_index==0]
        return data


    def convertDataFormat(self, detectionsData):
        formattedData = {}

        for line in detectionsData:
            id = line[1]
            bbLeft = line[2] # bb top left corner x coordinate
            bbTop = line[3] # bb top left corner y coordinate
            bbWidth = line[4] # bb width
            bbHeight = line[5] # bb height

            bbCentreX = bbLeft + (bbWidth/2)
            bbCentreY = bbTop - (bbHeight/2)

            scale = bbWidth*bbHeight
            aspectRatio = bbWidth/bbHeight

            stateVector = np.array([bbCentreX, bbCentreY, scale, aspectRatio]).reshape((4, 1))
            dataVector = np.concatenate((stateVector, np.zeros((3,1))), axis=0)

            # print(stateVector)

            if id in formattedData:
                formattedData[id].append(dataVector)
            else:
                formattedData[id] = [dataVector]

        return formattedData
    

    def restructureData(self, formattedData):
        # print(len(formattedData))
        finalData = torch.zeros(len(formattedData), 7, 600)

        for key, value in formattedData.items():
            example = torch.zeros(7, 600)
            for index, val in enumerate(formattedData[key]):
                # print(index)
                # print(torch.tensor(val), torch.tensor(val).shape)
                # print('\n\n')
                example[:, index] = torch.squeeze(torch.tensor(val), 1)
            finalData[int(key-1), :, :] = example
        # print(finalData.shape)

        trainInput = finalData[:, :, :599]
        trainOutput = finalData[:, :, 1:]

        # print(trainInput[1][0], trainInput[1][1])
        # print(trainOutput[1][0], trainOutput[1][1])
        return trainInput, trainOutput

    def drawResults(self, im = None, t = 0):
        self.draw_boxes = False

        maxConf = 1
        if self.mode == "det":
            maxConf = max(self.resFile[:,6])

        # boxes in this frame
        thisF=np.flatnonzero(self.resFile[:,0]==t)

        for bb in thisF:
            targetID = self.resFile[bb,1]
            IDstr = "%d" % targetID
            left=((self.resFile[bb,2]-1)*self.imScale).astype(int)
            top=((self.resFile[bb,3]-1)*self.imScale).astype(int)
            width=((self.resFile[bb,4])*self.imScale).astype(int)
            height=((self.resFile[bb,5])*self.imScale).astype(int)

            left=((self.resFile[bb,2]-1)).astype(int)
            top=((self.resFile[bb,3]-1)).astype(int)
            width=((self.resFile[bb,4])).astype(int)
            height=((self.resFile[bb,5])).astype(int)

            # normalize confidence to [0,5]
            rawConf=self.resFile[bb,6]
            conf=(rawConf)/maxConf
            conf = int(conf * 5)


            pt1=(left,top)
            pt2=(left+width,top+height)

            color = self.colors[int(targetID % len(self.colors))]
            color = tuple([int(c*255) for c in color])



            if  self.mode == "gt":
                label = self.resFile[bb, 7]
            else:
                label = 1


            # occluder
            if ((self.mode == "gt") & (self.showOccluder) & (int(label) in [9, 10, 11, 13])):

                overlay = im.copy()
                alpha = 0.7
                color = (0.7*255, 0.7*255, 0.7*255)
                cv2.rectangle(overlay,pt1,pt2,color ,-1)
                im = cv2.addWeighted(overlay, alpha, im, 1 - alpha, 0)

            else:
                cv2.rectangle(im,pt1,pt2,color,2)
                if not self.mode == "det":
                    cv2.putText(im,IDstr,pt1,cv2.FONT_HERSHEY_SIMPLEX,1, color = color)

        return im

    def generate_colors(self):
        """
        Generate random colors.
        To get visually distinct colors, generate them in HSV space then
        convert to RGB.
        """
        N = 30
        brightness = 0.7
        hsv = [(i / N, 1, brightness) for i in range(N)]
        colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
        perm = [15, 13, 25, 12, 19, 8, 22, 24, 29, 17, 28, 20, 2, 27, 11, 26, 21, 4, 3, 18, 9, 5, 14, 1, 16, 0, 23, 7, 6, 10]
        colors = [colors[idx] for idx in perm]
        return colors

    def convertVideo(self, extensions):
        for ext in extensions:
            print( self.outputName)
            outputNameNewExt = "%s%s" % (self.outputNameNoExt, ext)
            print("Convert Video to: %s" % outputNameNewExt)
            command = "ffmpeg -loglevel warning -y -i %s -c:v libvpx-vp9 -crf 30 -b:v 0 -b:a 128k -c:a libvorbis -cpu-used 8  %s" % (self.outputName, outputNameNewExt)
            os.system(command)

    def generateVideo(self, outputName = None, extensions = [], displayTime = False, displayName = False, showOccluder = False, fps = 25):

        self.showOccluder = showOccluder

        if outputName == None:
            outputName = self.seqName


        if not self.mode == "raw":
                #load File
                self.resFile = self.loadData()

        # get images Folder
        # check if image folder exists
        if not os.path.isdir(self.image_dir):
            print ("imgFolder does not exist")
            sys.exit()


        imgFile = "000001.jpg"
        img = os.path.join(self.image_dir,imgFile)
        print("image file" , img)
        im =  cv2.imread(img,1)
        height, width, c = im.shape

        self.imScale = 1
        if width > 800:
            self.imScale = .5
            width = int(width*self.imScale)
            height = int(height*self.imScale)

        # video extension
        extension = ".mp4"
        if self.mode:
            self.outputNameNoExt = os.path.join(self.output_dir, "%s-%s" % (outputName, self.mode))
        else:
            self.outputNameNoExt = os.path.join(self.output_dir, outputName)
        self.outputName = "%s%s" % (self.outputNameNoExt, extension)

        self.out = cv2.VideoWriter(self.outputName,cv2.VideoWriter_fourcc(*'mp4v'), fps, (width,height))

        print ("Output name: %s"%self.outputName)
        self.colors = self.generate_colors
        t=0

        for img in sorted(glob.glob(os.path.join(self.image_dir,"*.jpg"))):
            t+=1

            im = cv2.imread(img,1)

            if not self.mode == "raw":
                try:
                    im = self.drawResults(im, t)
                except Exception as e:
                    print(str(traceback.format_exc()))
            im=cv2.resize(im,(0,0),fx=self.imScale,fy=self.imScale)
        
            if displayTime:
                cv2.putText(im,"%d" % t,(25,50),cv2.FONT_HERSHEY_PLAIN,self.imScale*6,[255, 255, 255], thickness = 3)
            if displayName:
                text = "%s: %s" %(self.seqName, displayName)
                cv2.putText(im, text,(25,height - 25 ),cv2.FONT_HERSHEY_DUPLEX,self.imScale* 2,[255, 255, 255],  thickness = 2)

            if t == 1:
                cv2.imwrite("{}.jpg".format(self.outputNameNoExt), im)
                im_mini = cv2.resize(im, (0,0), fx=0.25, fy=0.25)
                cv2.imwrite("{}-mini.jpg".format(self.outputNameNoExt), im_mini)
            self.out.write(im)
        self.out.release()

        print("Finished: %s"%self.outputName)
        if not len(extensions)==0:
            print("Convert Video to : ", extensions)
            self.convertVideo(extensions)




visualizer = ConvertToKNetFormat(seqName = "MOT17-02",
    FilePath = "Data/MOT17DetLabels/train/MOT17-02/gt/gt.txt",
    image_dir = "data/MOT16/train/MOT16-02/img1",
    mode = "gt",
    output_dir = "Data/visualisation_output")

detectionsData = visualizer.loadData()
formattedData = visualizer.convertDataFormat(detectionsData)
trainInput, trainOutput = visualizer.restructureData(formattedData)
    # print(formattedData)

    # visualizer.generateVideo(
    #         displayTime = True,
    #         displayName = "traj",
    #         showOccluder = True,
    #         fps = 30)