import requests as rq
import time
start = time.time()

# file_uploaded = open('vid3.mp4','rb')
# url = 'http://10.8.100.28:8000/api/upload/'
# file_uploaded ={"file_uploaded":file_uploaded}
# #request = rq.get(url)
# #print(request.text)
# request = rq.post(url, files=file_uploaded)

# print(dir(request))
# print(request.text)
# print(request.links)
# print(time.time()-start)

def upload_file(url,filepath):
    file_uploaded = open(filepath,'rb')
    file_uploaded =  {"file_uploaded":file_uploaded}
    request = rq.post(url, files=file_uploaded)
    #print(request.text)


if  __name__=="__main__":
    print('a')
    waktu =[]
    for i in range(10):
        start = time.time()
      
        cv_img = open('frame.jpg','rb')
        file_uploaded ={"file_uploaded":cv_img}
        
        req = rq.post('http://192.168.43.220:8000/api/upload/',files=file_uploaded)
        #print(req.text)

        print(time.time()-start)
        waktu.append(time.time()-start)
        #0.03 second 1 pic
    print(sum(waktu)/len(waktu))


    '''
    0.18800616264343262
    0.09780263900756836
    0.10197162628173828
    0.07647013664245605
    0.08888959884643555
    0.10402870178222656
    0.11475658416748047
    0.08525419235229492
    0.10785603523254395
    0.09099292755126953
    average:0.10572588443756104

    '''
