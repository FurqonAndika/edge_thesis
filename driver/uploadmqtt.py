
# python 3.6

import random
from paho.mqtt import client as mqtt_client
import cv2
import json
import numpy 
from json import JSONEncoder



MQTT_KEEPALIVE_INTERVAL=45
# Generate a Client ID with the publish prefix.
client_id = f'publish-{random.randint(0, 100000)}'
username = 'emqx'
password = 'public'


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, numpy.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)
    
def connect_mqtt(broker, port):
    def on_connect(client, userdata, flags, rc):
        if rc == 0:
            print("Connected to MQTT Broker!")
        else:
            print("Failed to connect, return code %d\n", rc)

    client = mqtt_client.Client(client_id)
    client.username_pw_set(username, password)
    client.on_connect = on_connect
    client.connect(broker, port,MQTT_KEEPALIVE_INTERVAL)
    # client.connect(broker, port)
    return client


#data as numpy image
def publish(client,data,topic):
    # data = {
        # "image":data.tostring(),
        # "shape":data.shape,
    # }
    result = client.publish(topic,str(data),qos=0)
    status = result[0]
    if status == 0:
        # print(f"Send `{msg}` to topic `{topic}`")
        #$print("send to client")
        pass
    else:
        pass
        #print(f"Failed to send message to topic {topic}")


    
def run():
    client = connect_mqtt(broker='broker.emqx.io',port = 1883)
    client.loop_start()
    publish(client,data=image,topic='road_damage_itb')
    client.loop_stop()
   

if __name__ == '__main__':
    run()

