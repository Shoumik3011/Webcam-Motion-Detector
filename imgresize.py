import cv2
import glob

images=glob.glob("*.jpg")
for image in images:
    imag=cv2.imread(image,0)
    newimg=cv2.resize(imag,(100,100))
    cv2.imshow("display",newimg)
    print(newimg.shape)
    cv2.waitKey(500)
    cv2.destroyAllWindows()
    #cv2.imwrite("resized_"+image,newimg)
    print(images)

    