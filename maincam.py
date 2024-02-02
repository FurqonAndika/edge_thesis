import cv2
import numpy as np
import onnxruntime 
from driver.video import detect_frame
from driver.serialgps import GPS
import time
import requests
import threading
import pandas as pd

def main():
    session = requests.Session()
    for i in range(20):
        #cap = cv2.VideoCapture(i,cv2.CAP_V4L2)
        cap = cv2.VideoCapture(i,cv2.CAP_DSHOW)
        if cap.isOpened()==True:
            print(i)
            break
    #cap = cv2.VideoCapture('result1.avi')
    #cap = cv2.VideoCapture('videotest.mp4')
    opt_session = onnxruntime.SessionOptions()
    opt_session.enable_mem_pattern = False
    opt_session.enable_cpu_mem_arena = False
    opt_session.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

    model_path = "yolov8n/best320.onnx"

    EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    ort_session = onnxruntime.InferenceSession(model_path, providers=EP_list)
    model_inputs = ort_session.get_inputs()
    input_names = [model_inputs[i].name for i in range(len(model_inputs))]
    input_shape = model_inputs[0].shape
    model_output = ort_session.get_outputs()
    output_names = [model_output[i].name for i in range(len(model_output))]
    
    box_1 = [[0,0],[0,0],[0,0],[0,0]]
    
    # Define classes 
    CLASSES = ['D00','D10','D20','D40']
    detect = True
    gps = GPS()
    gps.get_port()
    

    #save video
    #result = cv2.VideoWriter('result.avi',cv2.VideoWriter_fourcc('F','M','P','4'),10, (int(cap.get(3)),int(cap.get(4))))

    df = pd.DataFrame(columns=['iterations','latitude','longitude','scores','road_damage','status','fps'])
    iterations = []
    lats =[]
    lons = []
    scores_arr=[]
    road_damages=[]
    status_arr =[]
    fps_arr = []
    it =0

    while cap.isOpened():
        start = time.time()
        ret, frame = cap.read()
        #result.write(frame)
        scale = 90
        width = int(frame.shape[1]*scale/100)
        height = int(frame.shape[2]*scale/100)
        dimention = (width, height)
        #frame = cv2.resize(frame,dimention,interpolation = cv2.INTER_AREA) 
        #frame = cv2.imread('1.jpg')
        #frame = frame [:100,0:100]

        #gps
        try:
            latlng = gps.read_line()
            print(latlng)
            latlng = eval(str(latlng))
            lat = latlng['lat']
            lng = latlng ['lng']        
        except:
            lat =0
            lng =0
        #lat = "-6.914744"
        #lng = "107.609810"
        iterations.append(it)
        lats.append(lat)
        lons.append(lng)
        
        if detect==True:
            #image_draw,str(label),int(scores[max_val]*100),str(status),iou)
            start = time.time()
            original_frame,frame,label,scores,status, iou = detect_frame(frame, input_shape, ort_session,CLASSES,box_1,output_names,input_names)
            send = False
            threshold =20
            if scores>threshold:
                print('scores:' ,scores)
                cv2.imwrite('pic.jpg',frame)
                send = True
                #send as detected status
                
            elif scores>5 and scores<=threshold:
                print('under:', scores)
                frame = original_frame
                status = 'undetected'
                cv2.imwrite('pic.jpg',frame)
                send = True
                #send as undetected
                
            else:
                frame = original_frame
                status ='undetected'

            status_arr.append(status)
            road_damages.append(label)
            scores_arr.append(scores)
         
            
            cv2.putText(frame,f'scores:{scores}', (10,100),cv2.FONT_HERSHEY_SIMPLEX,0.60, [225, 255, 255],thickness=1)

            #send image
            if send:
                cv_img = open('pic.jpg','rb')
                file_uploaded ={"file_uploaded":cv_img}
                data = {
                    "latitude":lat,
                    "langitude" :lng,
                    "status" :status,
                    "damage_type" :label,
                    "location":'',
                    "city": '',
                    "scores":scores
                }
                try:
                    def req():
                        req = session.post('http://192.168.137.1:8000/api/upload/',files=file_uploaded, data = data)
                        print(req.text)
                    #thread = threading.Thread(target=req)
                    #thread.start()
                    pass
                except Exception as e:
                    print('req_error', e)
        #print('fps', 1/(time.time()-start))
        fps = 1/(time.time()-start)
        fps_arr.append(fps)
    
        cv2.putText(frame,f'fps:{fps}', (10,40),cv2.FONT_HERSHEY_SIMPLEX,0.60, [225, 255, 255],thickness=1)
        cv2.putText(frame,f'lat:{lat}', (10,60),cv2.FONT_HERSHEY_SIMPLEX,0.60, [225, 255, 255],thickness=1)
        cv2.putText(frame,f'lng:{lng}', (10,80),cv2.FONT_HERSHEY_SIMPLEX,0.60, [225, 255, 255],thickness=1)

        #cv2.namedWindow('frame', cv2.WND_PROP_FULLSCREEN)
        #cv2.setWindowProperty('frame', cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
        cv2.imshow('frame', frame)

        it +=1
        
        key = cv2.waitKey(1)
        if key ==ord('q'):
            break

        elif key==ord('d'):
            detect = True
        
        elif key==ord('u'):
            detect = False

    #['iterations','latitude','longitude'])
    
    df['iterations'] = iterations
    df['latitude'] = lats
    df['longitude'] = lons
    df['scores'] =scores_arr
    df['road_damage']=road_damages
    df['status']  =status_arr
    df['fps']= fps_arr
    #df.to_csv('gpsloger.csv',index=False)

    cap.release()
    #result.release()
    cv2.destroyAllWindows()


if __name__=="__main__":
    main()
