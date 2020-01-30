import cv2,time

video=cv2.VideoCapture(0,cv2.CAP_DSHOW)  
#method that triggers video capturing i.e turn on the webcam...
#the parameter can be either index [0-one camera, 1-second camera....and so on] or video file path...
#0 here indicates that i have a webcam in the laptop..

a=0
while True:
    a+=1
    check, frame=video.read()
    print(check) #print a boolean value...TRUE if video recording is successul else FALSE...
    print(frame) #print the numpy array

    grey=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    #time.sleep(3) #this will provide 3 seconds delay...

    cv2.imshow("Capturing",grey)
    key=cv2.waitKey(1) #this will show each frame with an interval of 1 sec b/w each frame...
    if(key==ord('q')): #on pressing "q" the while loop will stop...
        break
#print(key,ord('q'))
video.release()
#release the camera i.e turn off the webcam...
cv2.destroyAllWindows()
print(a) #to check for total number of frames...