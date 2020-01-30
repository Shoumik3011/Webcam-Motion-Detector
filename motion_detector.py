import cv2,time,pandas
from datetime import datetime   #import datetime class from datetime library....

first_frame=None          #to store the first frame of the video..
status_list=[None,None]
timedur=[]
datfr=pandas.DataFrame(columns=["Start Time","End Time"]) #create a dataframe using pandas....

video=cv2.VideoCapture(0,cv2.CAP_DSHOW)  
#method that triggers video capturing i.e turn on the webcam...
#the parameter can be either index [0-one camera, 1-second camera....and so on] or video file path...
#0 here indicates that i have a webcam in the laptop..

while True:
    check, frame=video.read()
    #print(check) #print a boolean value...TRUE if video recording is successul else FALSE...
    status=0

    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray=cv2.GaussianBlur(gray,(21,21),0)         #to smooth the image in order to remove noice from image and increase accuracy while comapring images 

    if first_frame is None:
        first_frame=gray
        continue

    delta_frame=cv2.absdiff(first_frame,gray)

    threshold_frame=cv2.threshold(delta_frame,100,255,cv2.THRESH_BINARY)[1]  
    #"0" corresponds to '255' value i.e white....
    #threshold method returns a tuple..we are putting [1] since we want the 2nd value of the threshold tuple...  

    threshold_frame=cv2.dilate(threshold_frame,None,iterations=2)   #to make the tthreshold frame smoother..remove black holes from big white areas..                                               

    (cnts,_)=cv2.findContours(threshold_frame.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)  #find contours of all the distinct objects in the image...
   
    for contours in cnts:
        if(cv2.contourArea(contours)<10000):
            continue
        status=1
        (x,y,wd,ht)=cv2.boundingRect(contours) 
        cv2.rectangle(frame,(x,y),(x+wd,y+ht),(0,255,0),3)
    status_list.append(status)
    if status_list[-1]==1 and status_list[-2]==0:
        timedur.append(datetime.now())    #current time(when status changed from 0 to 1) will be stored into the list...
    if status_list[-1]==0 and status_list[-2]==1:
        timedur.append(datetime.now())    #current time(when status changed from 1 to 0) will be stored into the list...

    cv2.imshow("Gray Frame",gray)
    cv2.imshow("Delta Frame",delta_frame)
    cv2.imshow("Threshold Frame",threshold_frame)
    cv2.imshow("Frame",frame)
    #print(gray) #print the numpy array
    #print(delta_frame)
    
    key=cv2.waitKey(1) #this will show each frame with an interval of 1 sec b/w each frame...

    if(key==ord('q')): #on pressing "q" the while loop will stop...
        if(status==1):
            timedur.append(datetime.now())    #to store the end time in case when the window is quitted when the object was in front of the window...
        break
  
#print(status_list)
print(timedur)
for i in range(0,len(timedur),2):
    datfr=datfr.append({"Start Time":timedur[i],"End Time":timedur[i+1]},ignore_index=True)

datfr.to_csv("timedur.csv")     #exporting the data frame to a csv file...

video.release()
#release the camera i.e turn off the webcam...
cv2.destroyAllWindows()
