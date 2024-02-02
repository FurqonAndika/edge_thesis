from PyQt5.QtCore import  QThread, pyqtSignal
import serial
import serial.tools.list_ports


class Serial_GPS(QThread):
    port = None
    progress =  pyqtSignal(str)
    

    def get_port(self):
        comlist = serial.tools.list_ports.comports()
        connected_com = []
        for elemen in comlist:
            connected_com.append(elemen.device)

        available_port = connected_com   
        available_port.reverse()
        #print('available_ports:',available_port)
        for com in available_port:
            try:
                self.ser = serial.Serial(com, 9600, timeout=1)
                if self.ser.isOpen():
                    self.port = com
                    #print('com:',com)
                    break
            except Exception as e:
                print('e',e)

    def run(self):
        self.get_port()
        self.count = 0
        while True:
            try:
                # Reading all bytes available bytes till EOL  
                line = self.ser.readline().decode('utf-8').rstrip()
                #print('line:',line)
                if not line:
                    self.count +=1
                    if self.count >5:
                        self.get_port()
                # Converting Byte Strings into unicode strings
                # string_gps = line.decode()
                # string_gps = string_gps.strip()
                # print(string_gps)
                self.progress.emit(line)
            except Exception as e:
                print('error:',e)
                self.progress.emit("0")
                self.get_port()


class GPS:
    def __init__(self,baudrate=9600):
        self.baudrate = baudrate
        self.ser = None
        self.error = 0
   
    def get_port(self):
        comlist = serial.tools.list_ports.comports()
        connected_com = []
        for elemen in comlist:
            connected_com.append(elemen.device)

        available_port = connected_com   
        available_port.reverse()
        #print('available_ports:',available_port)
        for com in available_port:
            try:
                ser = serial.Serial(com, self.baudrate, timeout=1)
                if ser.isOpen():
                    port = com
                    self.ser = ser
                    self.error=0
                    print('com',com)
                    break
                    #print('com:',com)
                else:
                    print('cant get port')
            except Exception as e:
                print('e',e)

    def read_line(self):
        try:
            line = self.ser.readline().decode('utf-8').rstrip()
            return line
        except Exception as e:
            self.error +=1
            if self.error>5:
                self.get_port()
            print('error gps:' ,e)

if __name__=="__main__":
    gps = GPS()
    gps.get_port()
    while True:
        latlng = gps.read_line()
        print(latlng)
    
