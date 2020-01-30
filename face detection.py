import cv2

face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")  
#this method will create a face cascade object in python...
#this will create a cascade classifier object and you can use this cascade classifier object of the face feature to search for face in the image..

img=cv2.imread("IMG_0399.JPG")  #no 2nd parameter indicates that the image is being read as coloured...

gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

 #we are reading the image in greyscale to increase accuracy of opencv to detect face in the image
 #this will convert the coloured image into greyscale image...

faces=face_cascade.detectMultiScale(gray_img,
scaleFactor=1.35, 
minNeighbors=5 )    

#this will search for the frontalface xml file in our image and will retrun the coordintes(x,y) of the upper left corner of the face and the (height,width) of the face rectangle in the image....               
#scalefactor=1.05 tells python to decrease the scale by 5% for the next face search..a small value means high accuracy... 

for x,y,wd,ht in faces:
    img=cv2.rectangle(img,(x,y),(x+wd,y+ht),(0,255,0),3)

#rectangle(image object,(tuple of starting point of the rectangle),(tuple of lowest right corner of the rectangle),color of rectangle(B,G,R) format,width of the rectangle).....
#rectangle of green color sincs(blue=0,green=255(full),red=0)..


#if you want to resize the image if it does not fits to your screen...
resized=cv2.resize(img,(int(img.shape[1]/2),int(img.shape[0]/2)))
print(type(faces))
print(faces)
cv2.imshow("new img",resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
