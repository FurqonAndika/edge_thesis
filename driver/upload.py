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
    request = rq.post(url, files=file_uploaded)
    print(request.text)


if  "__name__"=="_main__":
    print('a')
