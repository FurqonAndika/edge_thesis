
from PyQt5.QtCore import pyqtSignal, QThread,pyqtSlot
import numpy as np
import cv2
import time
import onnxruntime 
from shapely.geometry import Polygon

def calculate_iou(box_1, box_2):
    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)
    iou = poly_1.intersection(poly_2).area/poly_1.union(poly_2).area
    return iou

def compute_iou(box, boxes):
    # Compute xmin, ymin, xmax, ymax for both boxes
    xmin = np.maximum(box[0], boxes[:, 0])
    ymin = np.maximum(box[1], boxes[:, 1])
    xmax = np.minimum(box[2], boxes[:, 2])
    ymax = np.minimum(box[3], boxes[:, 3])

    # Compute intersection area
    intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

    # Compute union area
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union_area = box_area + boxes_area - intersection_area

    # Compute IoU
    iou = intersection_area / union_area

    return iou



def xywh2xyxy(x):
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2
    y[..., 1] = x[..., 1] - x[..., 3] / 2
    y[..., 2] = x[..., 0] + x[..., 2] / 2
    y[..., 3] = x[..., 1] + x[..., 3] / 2
    return y

def nms(boxes, scores, iou_threshold):
    sorted_indices = np.argsort(scores)[::-1]
    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes

class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray,str,int,str,float)

    def __init__(self, source,model_path,detect=True):
        super().__init__()
        self._run_flag = True
        self.source = source
        self.model_path = model_path
        self.detect = detect
        
    @pyqtSlot()
    def run(self):
        cap = cv2.VideoCapture(self.source)
        start = time.time()
        opt_session = onnxruntime.SessionOptions()
        opt_session.enable_mem_pattern = False
        opt_session.enable_cpu_mem_arena = False
        opt_session.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

        # model_path = 'model yolov8/best.onnx'

        EP_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        ort_session = onnxruntime.InferenceSession(self.model_path, providers=EP_list)
        model_inputs = ort_session.get_inputs()
        input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        input_shape = model_inputs[0].shape
        model_output = ort_session.get_outputs()
        output_names = [model_output[i].name for i in range(len(model_output))]
        
        box_1 = [[0,0],[0,0],[0,0],[0,0]]
        
        # Define classes 
        CLASSES = ['D00','D10','D20','D40']
        while self._run_flag==True:
            start = time.time()
            #image = cv2.imread(source)
            ret,image = cap.read()
            #image = image [320:]
            
            #compress
            #image = cv2.resize(image, None, fx=0.5, fy=0.5)
            try:
           
                if self.detect == True:
                    image_height, image_width = image.shape[:2]
                    input_height, input_width = input_shape[2:]
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    #resized = cv2.resize(image_rgb, (input_width, input_height))
                    resized = cv2.resize(image_rgb, (input_width, input_height), interpolation = cv2.INTER_AREA)
                        
                    # Scale input pixel value to 0 to 1
                    input_image = resized / 255.0
                    input_image = input_image.transpose(2,0,1)
                    input_tensor = input_image[np.newaxis, :, :, :].astype(np.float32)
                    
                    outputs = ort_session.run(output_names, {input_names[0]: input_tensor})[0]
                    predictions = np.squeeze(outputs).T
                    
                    conf_thresold = 0.1
                    # Filter out object confidence scores below threshold
                    scores = np.max(predictions[:, 4:], axis=1)
                    predictions = predictions[scores > conf_thresold, :]
                    scores = scores[scores > conf_thresold]
                    #print(scores)
                    # Get the class with the highest confidence
                    class_ids = np.argmax(predictions[:, 4:], axis=1)
                    # print(class_ids)
                    # print(scores)
                    
                    # Get bounding boxes for each object
                    boxes = predictions[:, :4]
                    #rescale box
                    input_shape = np.array([input_width, input_height, input_width, input_height])
                    boxes = np.divide(boxes, input_shape, dtype=np.float32)
                    boxes *= np.array([image_width, image_height, image_width, image_height])
                    boxes = boxes.astype(np.int32)

                   
                    image_draw = image.copy()     
                    max_val = np.argmax(scores)
                    bbox = xywh2xyxy(boxes[max_val]).round().astype(np.int32).tolist()
                    cls_id = int(class_ids[max_val])
                    cls = CLASSES[cls_id]
                    color = (0,255,0)
                    # print(tuple(bbox[:2]))
                    cv2.rectangle(image_draw, tuple(bbox[:2]), tuple(bbox[2:]), color, 2)   
                    #cv2.putText(image_draw,f'{cls}:{int(scores[max_val]*100)},fps:{fps}', (bbox[0], bbox[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.60, [225, 255, 255],thickness=1)
                    cv2.putText(image_draw,f'{cls}:{int(scores[max_val]*100)} %', (bbox[0], bbox[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.60, [225, 255, 255],thickness=1)

                
                    
                    image_draw = cv2.resize(image_draw, (image_width,image_height ))
                    #cv2.imwrite('result.jpg',image_draw)
                    
                    #label
                    if len(predictions) >0 :
                        class_ids = np.argmax(predictions[:, 4:], axis=1)
                        label = CLASSES[class_ids[np.argmax(scores)]]
                        status = "detected"
                        #print(label,int(scores[max_val]*100))

                        #iou
                        xmin, ymin = bbox[:2]
                        xmax, ymax = bbox[2:]
                        box_2 = [[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]]
                        iou = calculate_iou(box_1,box_2)
                        box_1 = box_2
                        
                    else :
                        label = ""
                        status ="undetected"
                        iou = 0
                        
                    self.change_pixmap_signal.emit(image_draw,str(label),int(scores[max_val]*100),str(status),iou)
               
                else:   #detection false
                    label=""
                    score=0
                    status = "undetected"
                    self.change_pixmap_signal.emit(image,label,score,status,0)
      
            except Exception as e:
                print('error',e)
                self.change_pixmap_signal.emit(image,"",0,"undetected",0)   
        

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False




def detect_frame(frame, input_shape, ort_session,CLASSES,box_1,output_names,input_names):
    try:
        image = frame
        #input_shape,ort_session, class
        

        
        image_height, image_width = image.shape[:2]
        input_height, input_width = input_shape[2:]
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #resized = cv2.resize(image_rgb, (input_width, input_height))
        resized = cv2.resize(image_rgb, (input_width, input_height), interpolation = cv2.INTER_AREA)
            
        # Scale input pixel value to 0 to 1
        input_image = resized / 255.0
        input_image = input_image.transpose(2,0,1)
        input_tensor = input_image[np.newaxis, :, :, :].astype(np.float32)
        
        outputs = ort_session.run(output_names, {input_names[0]: input_tensor})[0]
        predictions = np.squeeze(outputs).T
        
        conf_thresold = 0.02
        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > conf_thresold, :]
        scores = scores[scores > conf_thresold]
        #print(scores)
        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)
        # print(class_ids)
        # print(scores)
        
        # Get bounding boxes for each object
        boxes = predictions[:, :4]
        #rescale box
        input_shape = np.array([input_width, input_height, input_width, input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([image_width, image_height, image_width, image_height])
        boxes = boxes.astype(np.int32)

       
        image_draw = image.copy()     
        max_val = np.argmax(scores)
        bbox = xywh2xyxy(boxes[max_val]).round().astype(np.int32).tolist()
        cls_id = int(class_ids[max_val])
        cls = CLASSES[cls_id]
        color = (0,255,0)
        # print(tuple(bbox[:2]))
        cv2.rectangle(image_draw, tuple(bbox[:2]), tuple(bbox[2:]), color, 2)   
        #cv2.putText(image_draw,f'{cls}:{int(scores[max_val]*100)},fps:{fps}', (bbox[0], bbox[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.60, [225, 255, 255],thickness=1)
        cv2.putText(image_draw,f'{cls}:{int(scores[max_val]*100)} %', (bbox[0], bbox[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.60, [225, 255, 255],thickness=1)


        
        image_draw = cv2.resize(image_draw, (image_width,image_height ))
        #cv2.imwrite('result.jpg',image_draw)
        
        #label
        if len(predictions) >0 :
            class_ids = np.argmax(predictions[:, 4:], axis=1)
            label = CLASSES[class_ids[np.argmax(scores)]]
            status = "detected"
            #print(label,int(scores[max_val]*100))

            #iou
            xmin, ymin = bbox[:2]
            xmax, ymax = bbox[2:]
            box_2 = [[xmin,ymin],[xmax,ymin],[xmax,ymax],[xmin,ymax]]
            iou = calculate_iou(box_1,box_2)
            box_1 = box_2
            
        else :
            label = "nolabel"
            status ="undetected"
            iou = 0
            
        return (image,image_draw,str(label),int(scores[max_val]*100),str(status),iou)

    except Exception as e:
        print('exception', e)
        label = "nolabel"
        status ="undetected"
        iou = 0
        scores=0
        return frame,frame,str(label),scores,str(status),iou
