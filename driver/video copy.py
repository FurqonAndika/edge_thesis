
from PyQt5.QtCore import pyqtSignal, QThread,pyqtSlot
import numpy as np
import cv2
import time

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray,str,int)

    def __init__(self, detect_fn, category):
        super().__init__()
        self._run_flag = True
        
    @pyqtSlot()
    def run(self):
        # capture from web cam
        try:
            self._run_flag = True
            if self.__type['type']=='Camera':
                try:
                    self.cap = cv2.VideoCapture(self.__type['port'], cv2.CAP_DSHOW)
                except Exception as e:
                    print('and',e)
                    self.error_signal.emit('error')
  
                start_time = time.time()
                while self._run_flag:
                    ret, frame = self.cap.read()
                    # crop
                    # frame = frame[int(frame.shape[0]/2):,:]
                    width = int(frame.shape[1] * self.scale_percent / 100)
                    height = int(frame.shape[0] * self.scale_percent / 100)
                    dim = (width, height)
                    
                 

                    if frame.shape[1]>600:
                        frame = cv2.resize(frame,dim, interpolation = cv2.INTER_AREA)
                        print(frame.shape)
                    if ret:
                        frame = cv2.flip(frame,1)
                        frame,label,score= self.detect(frame)
                        # frame = frame
                        # label = ""
                        # score = 0
                        # frame = cv_img
                        end_time = time.time()
                        fps = int(1/(end_time - start_time))
                        start_time = end_time
                        # cv2.putText(frame, f"FPS: {fps}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1)
                        self.change_pixmap_signal.emit(frame,str(label),int(score))
                        # self.result_signal.emit(str(label),int(score))         
                # shut down capture system
                self.cap.release()

            elif self.__type['type']=='Internal File':
                file = self.__type['file']
                extension = os.path.splitext(file)
                try:
                    extension = extension[1]
                    image_type = ['.jpg','.jpeg','.png','.JPG','.JPEG']
                    video_type = ['.avi','.mp4']
                    if extension in image_type:
                        frame = cv2.imread(file)
                        width = int(frame.shape[1] * self.scale_percent / 100)
                        height = int(frame.shape[0] * self.scale_percent / 100)
                        dim = (width, height)
                  
                        if frame.shape[1]>600:
                            frame = cv2.resize(frame,dim, interpolation = cv2.INTER_AREA)
                            print(frame.shape)
                        frame,label,score= self.detect(frame)
                        self.change_pixmap_signal.emit(frame,str(label),int(score))
                        # self.result_signal.emit(str(label),int(score)) 

                    elif extension in video_type:
                        self.cap = cv2.VideoCapture(file)
                        start_time = time.time()
                        while self._run_flag:
                            ret, frame = self.cap.read()
                            width = int(frame.shape[1] * self.scale_percent / 100)
                            height = int(frame.shape[0] * self.scale_percent / 100)
                            dim = (width, height)
              
                            if frame.shape[1]>600:
                                frame = cv2.resize(frame,dim, interpolation = cv2.INTER_AREA)
                                print(frame.shape)
                            if ret:
                                frame,label,score= self.detect(frame)
                                # frame = cv_img
                                end_time = time.time()
                                fps = int(1/(end_time - start_time))
                                start_time = end_time
                                # cv2.putText(frame, f"FPS: {fps}", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1)
                                self.change_pixmap_signal.emit(frame,str(label),int(score))
                                # self.result_signal.emit(str(label),int(score))      

                            # shut down capture system
                        self.cap.release()

                    else:
                        self.error_signal.emit("Error File Extension Not Supported use (jpg,JPGH,JPEG,jpeg,png,avi,mp4)")

                except Exception as e:
                    print('nad',e)  
                    self.error_signal.emit("Error Please stop and choose your File")

    
        except Exception as e:
            print(e)


    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
     