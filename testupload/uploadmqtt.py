









# python 3.6

import random
import time

from paho.mqtt import client as mqtt_client
import cv2
from pprint import pprint

import json

import numpy 
from json import JSONEncoder


#broker = 'broker.emqx.io'
broker = "172.30.112.1"
broker = "192.168.43.220"
broker = "192.168.137.1"  #bisa pixel lan
broker = "192.168.1.13"
#broker = 'broker.emqx.io'
# broker = 'localhost'
# broker = IPAddr

port = 1883
topic = "road_damage_itb"
MQTT_KEEPALIVE_INTERVAL=45
# Generate a Client ID with the publish prefix.
client_id = f'publish-{random.randint(0, 100000)}'
username = 'emqx'
password = 'public'

print('connect to', broker)

class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
    
def connect_mqtt():
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(client_id)
    client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.connect(broker, port,MQTT_KEEPALIVE_INTERVAL)
    #client.connect(broker, port)
    return client


def publish(client):
    start = time.time()
    msg_count = 1
    waktu =[]
    while True:
        start = time.time()
        # data = get_data()
        data = cv2.imread('frame.jpg')
        # data = data[:300]  
        # shape = data.shape
        # print(shape)
        # data = numpy.array([1,2,3,4,5])
        data = {
            "image":data.tostring(),
            "shape":data.shape,
            'title':str(msg_count)
            
            # "image":data
        }
        # msg = f'{data}{msg_count}'
        # msg = f"{data}"
        # msg = json.dumps(data)
        # msg = json.dumps(data,cls=NumpyArrayEncoder)
        # print(msg)
        result = client.publish(topic,str('data'),qos=0)
        # data = data.reshape(shape)
        # print(data)
        # result: [0, 1]
        status = result[0]
        if status == 0:
            # print(f"Send `{msg}` to topic `{topic}`")
            pass
            #print("send to client")
            
            
        else:
            print(f"Failed to send message to topic {topic}")
        msg_count += 1

        waktu.append(time.time()-start)
        # print(time.time()-start)

        time.sleep(0.1)
        if msg_count > 10:
            break
            
    print('average:',sum(waktu)/len(waktu))
    # print(time.time()-start)
    #test gamabr 0.051
    # 12.747712135314941 100 gambar json

'''
0.1205141544342041
0.0867149829864502
0.0564877986907959
0.06016349792480469
0.05648374557495117
0.05272197723388672
0.052265167236328125
0.0525209903717041
0.05725669860839844
0.05186867713928223
average: 0.06468505859375

0.012657642364501953
0.022929668426513672
0.07375383377075195
0.02298879623413086
0.02359294891357422
0.023519515991210938
0.023351192474365234
0.02854442596435547
0.023425579071044922
0.02332329750061035
average: 0.027804112434387206


'''



def run():
    
    client = connect_mqtt()
    client.loop_start()
    #waktu = []
    
    #for i in range(10):
        #start = time.time()
    publish(client)
        #print(time.time()-start)
        #waktu.append(time.time()-start)
    #print('average:',sum(waktu)/len(waktu))
    client.loop_stop()
    
'''
0.1352064609527588
0.17798972129821777
0.1560206413269043
0.15689921379089355
0.15768980979919434
0.15263652801513672
0.1532764434814453
0.20621395111083984
0.1510457992553711
0.15099310874938965
average: 0.16000452041625976
'''
    
'''
connect to 192.168.43.220
0.2571287155151367
0.1570756435394287
0.15691256523132324
0.16016888618469238
0.15837931632995605
0.15590858459472656
0.15625786781311035
0.1560499668121338
0.15614533424377441
0.16025543212890625
average: 0.1676029920578003

'''

'''
pixel
connect to 192.168.137.100
0.1504216194152832
0.15640664100646973
0.15358185768127441
0.1497669219970703
0.1534430980682373
0.1525883674621582
0.20899295806884766
0.1547069549560547
0.15270209312438965
0.15234017372131348
average: 0.15872633457183838

'''

if __name__ == '__main__':
    run()

