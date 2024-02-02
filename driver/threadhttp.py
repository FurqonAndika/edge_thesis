from PyQt5.QtCore import  QThread, pyqtSignal
import requests

class Worker_Http(QThread):
    progress =  pyqtSignal(str)
    city = ""
    location = ""
    lat = 0
    lng =0
    cv_img = ""
    damage_type = ""
    status = ""
    session = requests.Session()
    
    def run(self):
        try:
            #cv_img = open('pic.jpg','rb')
            file_uploaded ={"file_uploaded":self.cv_img}
            data = {
                "latitude":self.lat,
                "langitude" :self.lng,
                "status" :self.status,
                "damage_type" :self.damage_type,
                "location":self.location,
                "city": self.city
            }
            
            #req = self.session.post('http://10.8.107.88:8000/api/upload/',files=file_uploaded, data = data)
            req = self.session.post('https://72e6-167-205-0-218.ngrok-free.app/api/upload/',files=file_uploaded, data = data)
            print(req.text)
            self.progress.emit(req.text)
        except Exception as e:
            print(e)

    
       
             
