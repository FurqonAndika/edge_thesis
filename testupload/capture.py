import cv2

cap = cv2.VideoCapture(10)

while True:
    ret, frame = cap.read()

    if ret:
        cv2.imshow('frame',frame)

    if cv2.waitKey(1)==ord('q'):
        break
    elif cv2.waitKey(1)==ord('w'):
        print('w')
        cv2.imwrite('frame'+str(frame.shape)+'.jpg',frame)

cap.release()
cv2.destroyAllWindows()
    
