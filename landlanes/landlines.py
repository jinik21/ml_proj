import cv2
import numpy as np

def canny(image):
    gray=cv2.cvtColor(image,cv2.COLOR_RGB2GRAY)
    blur=cv2.GaussianBlur(gray,(5,5),0)
    cannyimage=cv2.Canny(blur,50,150)
    return cannyimage
def roi(image):
    height=image.shape[0]
    polygons=np.array([(200,height),(1100,height),(550,200)])
    mask=np.zeros_like(image)
    cv2.fillPoly( mask,np.int32([polygons]), 255)
    masked_image=cv2.bitwise_and(image,mask)
    return masked_image

def display_lines(image,lines):
    line_image=np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2=line.reshape(4)
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    return line_image


def make_cord(image,line_parameters):
    try:
        slope,interscept =line_parameters
    except TypeError:
        slope,interscept=1,1
    y1=image.shape[0]
    y2=int(y1*(3/5))
    x1 =int((y1-interscept)/slope)
    x2=int((y2-interscept)/slope)
    return np.array([x1,y1,x2,y2])



def avg_slope(image,lines):
    left_fit=[]
    right_fit=[]
    for line in lines:
        x1,y1,x2,y2=line.reshape(4)
        parameters=np.polyfit((x1,x2),(y1,y2),1)
        slope=parameters[0]
        intercept=parameters[1]
        if slope < 0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))

    left_fit_avg=np.average(left_fit,axis=0)
    right_fit_avg=np.average(right_fit,axis=0)
    left_line=make_cord(image,left_fit_avg)
    right_line=make_cord(image,right_fit_avg) 
    return np.array([left_line,right_line])


"""
img= cv2.imread('test_image.jpg')
lane_image=np.copy(img)
canny_img=canny(img)
cropped_img=roi(canny_img)
lines=cv2.HoughLinesP(cropped_img,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
avglines = avg_slope(lane_image,lines)
line_image=display_lines(lane_image,avglines)
final_image=cv2.addWeighted(lane_image,0.8,line_image,1,1)
cv2.imshow("result",final_image)
cv2.waitKey(0)
"""

cap=cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
    _,frame =cap.read()
    canny_img=canny(frame)
    cropped_img=roi(canny_img)
    lines=cv2.HoughLinesP(cropped_img,2,np.pi/180,100,np.array([]),minLineLength=40,maxLineGap=5)
    avglines = avg_slope(frame,lines)
    line_image=display_lines(frame,avglines)
    final_image=cv2.addWeighted(frame,0.8,line_image,1,1)
    cv2.imshow("result",final_image)
    if cv2.waitKey(5) == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()

