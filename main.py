
import sys
import os

from PyQt5.QtGui import *
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import *
from PyQt5.QtWidgets import QLabel, QMainWindow, QApplication,QSizeGrip
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtCore import *

from PyQt5 import *
from PyQt5 import uic

from driver.video import VideoThread
from driver.upload import upload_file
from driver.serialgps import Serial_GPS
import numpy as np
import cv2
import psutil

import requests

import time
import datetime
import threading 
from driver.threadhttp import Worker_Http

#mqtt driver
from driver.uploadmqtt import connect_mqtt, publish

class UI(QMainWindow):
        def __init__(self):

                super(UI, self).__init__()  # Call the inherited classes __init__ method

                self.ui = uic.loadUi('menu.ui', self)  # Load the .ui file

                # self.setWindowFlag(QtCore.Qt.FramelessWindowHint)
                self.showMaximized()
                # self.showFullScreen()
                # self.setWindowFlag(QtCore.Qt.Tool)
                self.setWindowOpacity(1)

                self.setWindowTitle('Dashboard')
                # self.setWindowIcon(QIcon('UI/aset/icon/logo.png'))

                # size grip
                self.gripSize = 10
                self.grip =QSizeGrip(self)
                self.grip.resize(self.gripSize, self.gripSize)

                 #stacked init
                self.stacked_page_init()    

                #buton init
                self.button_init()
                
                #label init
                self.label_init()


                # self.resizeEvent = self.label_camera_resize
                # self.label_camera.resizeEvent =self.camera_resize
                
                #table_init
                self.table_init()
                self.count = 0

                self.combo_box_init()
                self.frame_init()

                #load model
                try:
                
                    # self.camera_1 = VideoThread(source='vid3.mp4',model_path="yolov8n/best320.onnx",detect = 1)
                    # self.camera_1.start()
                    # self.camera_1.change_pixmap_signal.connect(self.update_image)

                    self.camera_2 = VideoThread(source='vid3.mp4',model_path="yolov8n/best320.onnx",detect =True)
                    self.camera_2.start()
                    self.camera_2.change_pixmap_signal.connect(self.update_image_2)
                except Exception as e:
                    print(e)


                #mqtt initiation
                self.client = connect_mqtt(broker='broker.emqx.io',port = 1883)
                #self.client = connect_mqtt(broker='10.8.107.83',port = 1883)
                self.client.loop_start()
                self.client.loop_stop()

                self.total = 0
                self.label_camera_1.hide()
                self.lat = 0
                self.lng = 0
                self.http = Worker_Http()
                self.session = requests.Session()
                self.serial_gps = Serial_GPS()
                self.serial_gps.start()
                self.serial_gps.progress.connect(self.update_gps)
                self.show()  # Show the GUI

        def update_gps(self,data):
            try:
                data =eval(data)
                self.lat = data['lat']
                self.lng = data['lng']
                self.http.lat = self.lat
                self.http.lng = self.lng
                print(lat,lng)
            except Exception as e:
                self.lat = 0
                self.lng =0
                self.http.lat = self.lat
                self.http.lng = self.lng
            
                #print('gpserror',e)

        
        def frame_init(self):
                pass

        # @pyqtSlot(np.ndarray)
        # def update_image(self, cv_img):
                # """Updates the image_label with a new opencv image"""
                # qt_img= self.convert_cv_qt(cv_img)
                # self.label_camera_1.setPixmap(qt_img)


        def upload_http(self):
            try:
                cv_img = open('pic.jpg','rb')
                file_uploaded ={"file_uploaded":cv_img}
                data = {
                    "latitude":self.lat,
                    "langitude" :self.lng,
                    "status" :self.status,
                    "damage_type" :self.damage_type,
                    "location":"(0,0)",
                    "city": ""
                }
                
                req = self.session.post('http://10.8.107.88:8000/api/upload/',files=file_uploaded, data = data)
                print(req.text)

            except Exception as e:
                print(e)

            
        @pyqtSlot(np.ndarray,str,int,str,float)
        def update_image_2(self, cv_img,label,score,status,iou):
                """Updates the image_label with a new opencv image"""
                qt_img= self.convert_cv_qt(cv_img)
                # damage = road_damage
                #print(label, score)
                self.label_camera_2.setPixmap(qt_img)
                self.damage_type = label
                self.status = status
                print(iou)
                
                #http
                self.http.damage_type = self.damage_type
                self.http.status = self.status
                
                #send image via http request
                cv2.imwrite('pic.jpg',cv_img)
                self.http.location = ""
                self.http.city = ""
                self.http.cv_img = open('pic.jpg','rb')

                if score==0 and status=='undetected':
                    self.count +=1
                    if self.count>=6:
                        self.http.start()
                        self.count=0
                    
                    

                if iou<0.8 and status =="detected":  #gambar pada object yang berbeda
                    #pass
                    self.http.start()
                    #self.thread_http.start()
                    #self.upload_http()


                    
                '''
                cv2.imwrite('pic.jpg',cv_img)
                try:
                    cv_img = open('pic.jpg','rb')
                    file_uploaded ={"file_uploaded":cv_img}
                    req = requests.post('http://10.8.100.28:8000/api/upload/',files=file_uploaded)
                    print(req.text)

                except Exception as e:
                    print(e)
                '''
                
                '''
                #send image via mqtt
                self.total+=1
               
                if label!='unknown':
                    data = {
                    'image':cv_img.tostring(),
                    "status":"detected",
                    'image':'image',
                    'damage':label,
                    'score':score,
                    'time' :str(datetime.datetime.now().minute)+'-'+str(datetime.datetime.now().second),
                    'total':self.total
                    }
                    publish(self.client,data=data,topic='road_damage_itb')
                else:
                    data = {
                    'image':cv_img.tostring(),
                    "status":"undetected",
                    'image':'image',
                    'damage':'unknown',
                    'score':0,
                    'time' :str(datetime.datetime.now().minute)+'-'+str(datetime.datetime.now().second),
                    'total':self.total
                    }
                    publish(self.client,data=data,topic='road_damage_itb')
                '''
        def convert_cv_qt(self, cv_img):
                rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch *w
                convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                p = convert_to_Qt_format.scaled(320, 320, Qt.KeepAspectRatio)

                # ram_info = psutil.virtual_memory()
                # print('available ram:',ram_info.available/1024/1024/1024,'GB')

                # cpu_percent = psutil.cpu_percent()
                # print('cpu usage:',cpu_percent,'%')
                
                    
                return QPixmap.fromImage(p)
        
        
        def combo_box_init(self):
              pass   

        def button_init(self):
                pass
        
        def label_init(self):
              self.label_camera_1 = self.findChild(QLabel,'label_camera')
              self.label_camera_2 = self.findChild(QLabel,'label_camera2')

        def table_init(self):
                pass

        def stacked_page_init(self):
                pass

if __name__ == '__main__':
        try:
                # QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
                # app = QApplication([])  # Create an instance of QtWidgets.QApplication
                app = QApplication(sys.argv)
                window = UI()  # Create an instance of our class
                app.exec_()  # Start the application

        except Exception as errormsg:
                print('ui ', errormsg)






