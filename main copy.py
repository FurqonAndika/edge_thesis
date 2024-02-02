
import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

from PyQt5 import *
from PyQt5 import uic
import PyQt5
import os

import io
import folium
from PyQt5 import QtWidgets, QtWebEngineWidgets

# chart
from PyQt5.QtChart import QChart, QChartView, QPieSeries, QPieSlice
from PyQt5.QtGui import QPainter, QPen
from PyQt5.QtCore import Qt

from driver.video import VideoThread
from driver.camera import list_ports, get_available_cameras

import pyautogui
import numpy as np

import tensorflow as tf
import cv2
from object_detection.utils import label_map_util
label_map_util.tf = tf.compat.v1
tf.gfile = tf.io.gfile



class UI(QMainWindow):
        def __init__(self):

                super(UI, self).__init__()  # Call the inherited classes __init__ method

                #get monitor size
                width, height = pyautogui.size()
                print("monitor : ",width, height)

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
                self.grip = QtWidgets.QSizeGrip(self)
                self.grip.resize(self.gripSize, self.gripSize)

                #lineedit
                self.line_edit_init()

                 #stacked init
                self.stacked_page_init()    

                #buton init
                self.button_init()

                self.display_width = 1280
                self.display_height = 720
                
                
                #label init
                self.label_init()


                # self.resizeEvent = self.label_camera_resize
                # self.label_camera.resizeEvent =self.camera_resize
                
                #table_init
                self.table_init()
                # 
                self.widget_init()

                self.combo_box_init()
                self.frame_init()

                #load model 
                self.category_index = label_map_util.create_category_index_from_labelmap("model50%/saved_model/label_map.pbtxt",use_display_name=True)
                tf.config.threading.set_intra_op_parallelism_threads(1)
                self.detect_fn = tf.saved_model.load('model50%/saved_model')
                self.thread = VideoThread(self.detect_fn, self.category_index)
                self.Worker = QThread(self) 
                self.thread.moveToThread(self.Worker)
                self.Worker.start()
                self.type = self.combobox_type.currentText()
                self.thread.set_type({'type':self.type})

                # fix error pyqtslot
                self.damage_label = ""

                self.show()  # Show the GUI
                

              
        def resizeEvent(self, event):
                # QtWidgets.QMainWindow.resizeEvent(self, event)
                # self.display_width, self.display_height =self.label_camera.width(),self.label_camera.height()
                # print(self.display_width)
                old = event.oldSize()
                new = QSize(self.label_camera.geometry().width(),self.label_camera.geometry().height())
                try:
                       self.display_width, self.display_height= new.width(), new.height()
                except Exception as e:
                       print(e)
                
                QMainWindow.resizeEvent(self, event)
        
        def frame_init(self):
                self.frame_file = self.findChild(QFrame, 'frame_11')
                self.frame_camera = self.findChild(QFrame, 'frame_camera')
                self.frame_combo_camera = self.findChild(QFrame, 'frame_12')
                self.frame_combo_camera.setHidden(True)

        

        def combo_box_init(self):
               self.combobox_type = self.findChild(QComboBox,'comboBox')
               self.combobox_camera = self.findChild(QComboBox,'comboBox_camera')
               list_cam = get_available_cameras()
               self.camera_list=list_cam.values()
               self.combobox_camera.addItems(self.camera_list)
               self.combobox_camera.currentIndexChanged.connect(self.selectionchangecamera)
            

               type = [ 'Internal File','Camera']
               self.combobox_type.addItems(type)
               self.combobox_type.currentIndexChanged.connect(self.selectionchange)
        
        def selectionchangecamera(self,i):
                self.port = self.combobox_camera.itemText(i)
                self.thread.set_type({'type':self.type,"port":list(self.camera_list).index(self.port)})


        def selectionchange(self,i):
                self.type = self.combobox_type.itemText(i)
                self.thread.set_type({'type':self.type,'port':0})
                if self.type =='Camera':
                        self.frame_file.setHidden(True)
                        self.frame_combo_camera.setHidden(False)
                else:
                       self.frame_file.setHidden(False)
                       self.frame_combo_camera.setHidden(True)

        def button_init(self):
                self.button_dashboard = self.findChild(QPushButton, 'button_dashboard')
                self.button_map = self.findChild(QPushButton, 'button_map')
                self.button_home = self.findChild(QPushButton,'bt_menu')
                self.button_input_data = self.findChild(QPushButton,'button_input_data')
                self.button_choose_file = self.findChild(QPushButton,'button_choose')
                self.button_start = self.findChild(QPushButton,'button_start')
                self.button_stop = self.findChild(QPushButton,'button_stop')

                # action
                self.button_map.clicked.connect(lambda:self.stacked_page.setCurrentWidget(self.map))
                self.button_dashboard.clicked.connect(self.show_dashboard)
                self.button_home.clicked.connect(self.bar)
                self.button_input_data.clicked.connect(lambda:self.stacked_page.setCurrentWidget(self.input_file))
                self.button_choose_file.clicked.connect(self.choose_file)
                self.button_start.clicked.connect(self.start_detect)
                self.button_stop.clicked.connect(self.stop_detect)
                self.button_stop.setVisible(False)

        def stop_detect(self):
                self.damage_label = ""
                self.combobox_type.setEnabled(True)
                self.combobox_camera.setEnabled(True)
                self.frame_file.setEnabled(True)
                self.thread.stop()
                self.label_camera.clear()
                self.button_start.setVisible(True)
                self.button_stop.setVisible(False)

        def start_detect(self):
                self.damage_label = ""
                self.table_detection.setRowCount(0)
                self.label_file_name.setText("")
                self.frame_file.setEnabled(False)
                self.combobox_type.setEnabled(False)
                self.combobox_camera.setEnabled(False)
                self.button_start.setVisible(False)
                self.button_stop.setVisible(True)
                self.thread.start()
                self.thread.change_pixmap_signal.connect(self.update_image)
                self.thread.error_signal.connect(self.update_error)
                # self.thread.result_signal.connect(self.update_result)
                
        
        # @pyqtSlot(str,int)
        # def update_result(self,label,score):
        #         if label !="":
        #                 print(label)
        #                 self.add_value_to_table(str(label),str(score),str('longlat'))
        
        def update_error(self, string):
                print(string)
                self.label_camera.setText(string)

        @pyqtSlot(np.ndarray,str,int)
        def update_image(self, cv_img,label, score):
                """Updates the image_label with a new opencv image"""
                qt_img= self.convert_cv_qt(cv_img)
                self.label_camera.setPixmap(qt_img)
                if label !="" and label != self.damage_label:
                        print(label)
                        self.add_value_to_table(str(label),str(score),str('longlat'))
                self.damage_label = label




        def close_event(self, event):
                self.thread.stop()
                event.accept()
                
        def convert_cv_qt(self, cv_img):
                rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch *w
                convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                p = convert_to_Qt_format.scaled(self.display_width, self.display_height, Qt.KeepAspectRatio)
                return QPixmap.fromImage(p)

        def choose_file(self):
                try:
                        self.label_file_name.setText("")
                        dlg = QFileDialog()
                        dlg.setFileMode(QFileDialog.FileMode.ExistingFiles)
                        dlg.setFileMode(QFileDialog.AnyFile)
                        # dlg.setFilter("Images/Video (*.jpg *.png *.jpeg *.mp4 *.avi)")    
                        dlg.setNameFilter("Images (*.png *.jpg *.jpeg *.mp4 *.avi)")                            
                        if dlg.exec_():
                                filenames = dlg.selectedFiles()
                                print(filenames[0])
                                self.label_file_name.setText(filenames[0])
                                self.thread.set_type({'type':self.type,'file':filenames[0]})
                except Exception as e:
                      print(e)

        def widget_init(self):
            self.widget_dashboard = self.findChild(QWidget, 'widget_3')
            self.widget_dashboard_ = self.findChild(QWidget, 'widget_2')
            self.addWidget=True
        
        def show_dashboard(self):
            self.stacked_page.setCurrentWidget(self.dashboard)
            
            # self.widget_dashboard = self.findChild(QWidget, 'widget_3')
            # self.widget_dashboard_ = self.findChild(QWidget, 'widget_2')
            
            # font
            font=QFont()
            font.setPixelSize(20)
            font.setPointSize(20)

            series = QPieSeries()
            series.append("Lubang", 80)
            series.append("Retak", 10)
            series.append("Retak Buaya", 20)
            series.append("Polisi tidur", 25)
            series.append("lain-lain", 30)

     
            #adding slice
            slice = QPieSlice()
            slice = series.slices()[2]
            slice.setExploded(True)
            slice.setLabelVisible(True)
            slice.setPen(QPen(Qt.darkGreen, 2))
            slice.setBrush(Qt.green)


            slice.setLabelFont(font)

     
            chart = QChart()
            chart.legend().hide()
            chart.addSeries(series)
            chart.createDefaultAxes()
            chart.setAnimationOptions(QChart.SeriesAnimations)
            chart.setTitle("Kerusakan Jalan")
     
            chart.legend().setVisible(True)
            chart.legend().setAlignment(Qt.AlignBottom)
    
            chartview = QChartView(chart)
            chartview.setRenderHint(QPainter.Antialiasing)
            

            series2 = QPieSeries()
            series2.append("Lubang", 80)
            series2.append("Retak", 10)
            series2.append("Retak Buaya", 20)
            series2.append("Polisi tidur", 25)
            series2.append("lain-lain", 30)
     
            #adding slice
            slice2 = QPieSlice()
            slice2 = series2.slices()[2]
            slice2.setExploded(True)
            slice2.setLabelVisible(True)
            slice2.setPen(QPen(Qt.darkGreen, 2))
            slice2.setBrush(Qt.green)
     
            chart2= QChart()
            chart2.legend().hide()
            chart2.addSeries(series2)
            chart2.createDefaultAxes()
            chart2.setAnimationOptions(QChart.SeriesAnimations)
            chart2.setTitle("Kerusakan Jalan")
     
            chart2.legend().setVisible(True)
            chart2.legend().setAlignment(Qt.AlignBottom)

            chartview_2 =  QChartView(chart2)
            chartview_2.setRenderHint(QPainter.Antialiasing)
            # self.setCentralWidget(chartview)
            layout = QGridLayout()
            layout_2 = QGridLayout()
            # layout = QVBoxLayout()
    
            if self.addWidget:
                try:
                    layout.addWidget(chartview,0,0)
                    layout_2.addWidget(chartview_2,0,1)
                    self.widget_dashboard.setLayout(layout)
                    self.widget_dashboard_.setLayout(layout_2)
                    self.addWidget = False
                except Exception as e:
                    print(e)

        def handle_button_group_text(self,button):
                pass


        def label_init(self):
                self.label_title = self.findChild(QLabel, 'label_3')
                self.label_logo = self.findChild(QLabel, 'label')
                self.label_map = self.findChild(QWidget,'widget')
                self.label_file_name = self.findChild(QLabel,'label_file_name')
                self.label_camera= self.findChild(QLabel,'label_camera')
                self.label_camera.resize(self.display_width, self.display_height) 
                self.label_camera.setMinimumWidth(720)
                self.label_camera.setMinimumHeight(320) 
                self.label_camera.adjustSize() 


                # location =[0.8077983825856107, 102.02653538131426]   #siak
                location =[-6.890550542243903, 107.61077159665338]
                m = folium.Map(location=location, tiles="OpenStreetMap", zoom_start=20)
                # m = folium.Map(location=[0.8077983825856107, 102.02653538131426], tiles="Stamen Terrain", zoom_start=13)
                folium.Marker(location=location,popup='Retak Horizontal ' +str (location),icon=folium.Icon(color='red', icon='envelope')).add_to(m)
                folium.LayerControl().add_to(m)
                data = io.BytesIO()
                m.save(data, close_file=False)
                self.label_map.setHtml(data.getvalue().decode())
                self.label_map.resize(640, 480)

                # self.label_title.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
                # self.label_logo.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
                # w.show()

        def line_edit_init(self):
                pass

        def table_init(self):
                self.table_detection = self.findChild(QTableWidget,'tableWidget')
                self.table_detection.setColumnCount(4)
                self.table_detection.setHorizontalHeaderLabels(['No','Damage Type','Accuracy','Longlat'])
                self.table_detection.horizontalHeader().setDefaultAlignment(Qt.AlignCenter)
                self.table_detection.horizontalHeader().setFixedHeight(30)
                self.table_detection.horizontalHeader().setStretchLastSection(True)
                self.table_detection.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

                self.table_detection.resizeColumnsToContents()
                self.table_detection.resizeRowsToContents()
                self.table_detection.show()

        def add_value_to_table(self,value1,value2,value3):
                rowPosition = self.table_detection.rowCount()
                self.table_detection.insertRow(rowPosition)
                self.table_detection.setItem(rowPosition , 0,QTableWidgetItem(str(self.table_detection.rowCount())))
                self.table_detection.setItem(rowPosition , 1,QTableWidgetItem(str(value1)))
                self.table_detection.setItem(rowPosition , 2, QTableWidgetItem(str(value2)))
                self.table_detection.setItem(rowPosition , 3, QTableWidgetItem(str(value3)))

        def stacked_page_init(self):
                self.stacked_page = self.findChild(QStackedWidget,'stackedWidget')
                self.stacked_page.setCurrentWidget(self.page_welcome)

        def bar(self):
            self.frame_nav_bar = self.findChild(QFrame, 'frame_nav_bar')
            if True:
                width = self.frame_nav_bar.width()
                normal = 0
                if width == 0:
                    extender = 180
                else:
                    extender = normal
                self.animacion = QPropertyAnimation(self.frame_nav_bar, b'minimumWidth')
                self.animacion.setDuration(500)
                self.animacion.setStartValue(width)
                self.animacion.setEndValue(extender)
                self.animacion.setEasingCurve(QEasingCurve.InOutQuart)
                self.animacion.start()        






if __name__ == '__main__':
        try:
                # QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
                # app = QApplication([])  # Create an instance of QtWidgets.QApplication
                app = QApplication(sys.argv)
                window = UI()  # Create an instance of our class
                app.exec_()  # Start the application

        except Exception as errormsg:
                print('ui ', errormsg)






