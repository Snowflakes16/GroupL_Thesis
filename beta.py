
from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2

pig_id=[[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],]

#############################################################################################################    
def order_points(pts):
    xSorted = pts[np.argsort(pts[:, 0]), :]

    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost

    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]

    return np.array([tl, tr, br, bl], dtype="float32")
##############################################################################################################

###############################################################################################################

def four_point_transform(image, pts):
 
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

   
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    return warped
#############################################################################################################    
def midpoint(ptA, ptB):
  return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
def detect_color1(ptA,ptB):
  return ((ptA[0] + ptB[0]) * 0.8, (ptA[1] + ptB[1]) * 0.8)
def topbot():
  #####TOPBOT SIDE THRESHOLDING####
  extpointsx=[extLeft_x,extRight_x,extTop_x,extBot_x]
  extpointsy=[extLeft_y,extRight_y,extTop_y,extBot_y]
  threshold_tltrX=[]
  threshold_tltrY=[]
  #########TLTR###################
  for epx in extpointsx:
    threshold_tltrX.append(abs(epx-tltrX))
  for epy in extpointsy:
    threshold_tltrY.append(abs(epy-tltrY))
  ########BLBR#################
  threshold_blbrX=[]
  threshold_blbrY=[]
  for epx in extpointsx:
    threshold_blbrX.append(abs(epx-blbrX))
  for epy in extpointsy:
    threshold_blbrY.append(abs(epy-blbrY))
  ########CHECKTIPARRAY##########
  #########TLBL##################
  tltrx_checktip=min(threshold_tltrX)
  tltry_checktip=min(threshold_tltrY)
  
  cntrtltrx=1
  cntrtltry=1
  for ctr_tltrx in threshold_tltrX:
    if(tltrx_checktip==ctr_tltrx):
      tltrx_tip=cntrtltrx
      break
    else:
      cntrtltrx=cntrtltrx+1
  for ctr_tltry in threshold_tltrY:
    if(tltry_checktip==ctr_tltry):
      tltry_tip=cntrtltry
      break
    else:
      cntrtltry=cntrtltry+1
  if(tltrx_tip==tltry_tip):
    return 3
  else:
    return 4


  print("COUNTER:",cntrtltrx," ",cntrtltry)
def leftright():
  #LEFT AND RIGHT SIDE THRESHOLDING
  extpointsx=[extLeft_x,extRight_x,extTop_x,extBot_x]
  extpointsy=[extLeft_y,extRight_y,extTop_y,extBot_y]
  threshold_tlblX=[]
  threshold_tlblY=[]
  #########TLBL############
  FLAG=0
  for epx in extpointsx:
    threshold_tlblX.append(abs(epx-tlblX))
  for epy in extpointsy:
    threshold_tlblY.append(abs(epy-tlblY))
  ########TRBR###########
  threshold_trbrX=[]
  threshold_trbrY=[]
  for epx in extpointsx:
    threshold_trbrX.append(abs(epx-trbrX))
  for epy in extpointsy:
    threshold_trbrY.append(abs(epy-trbrY))

  ########CHECKTIPARRAY##########
  #########TLBL############
  tlblx_checktip=min(threshold_tlblX)
  tlbly_checktip=min(threshold_tlblY)
  cntrtlblx=1 
  cntrtlbly=1
  for ctr_tlblx in threshold_tlblX:
    if(tlblx_checktip==ctr_tlblx):
      tlblx_tip=cntrtlblx
      break
    else:
      cntrtlblx=cntrtlblx+1
  for ctr_tlbly in threshold_tlblY:
    if(tlbly_checktip==ctr_tlbly):
      tlbly_tip=cntrtlbly
      break
    else:
      cntrtlbly=cntrtlbly+1

  if(tlblx_tip==tlbly_tip):
    return 1
  else:
    return 2

  print("COUNTER:",cntrtlblx," ",cntrtlbly)
def pig_coord(pig_loc,pig_id):
  
  pigcoordinates[0][pig_id] = pig_loc[0]
  pigcoordinates[1][pig_id] = pig_loc[1]

  return(pigcoordinates)
def color_picker(coord,pig_loc):
  #3=blue;2=green;1=red
  ret, img = cap.read()

  colors=[]
  for x in coord:
    print(x)
    B,G,R = img[x]
    #for checking if naka basa ba siya if red/green/blue.
    if R > G and R > B:
      colors.append(1)
      print("RED")
    elif G > R and G > B:
      colors.append(2)
      print("Green")

    elif B > R and B > G:
      colors.append(3)
      print("Blue")

    else:
      print("Cant identify")
  print(colors)
  index=0
  flag=0
  #for cntr1 in color_oder:
  # if(flag!=0):
  color_seq= [[1,2,3],
      [1,3,2],
      [3,2,1],
      [3,1,2],
      [2,3,1],
      [2,1,3],
      [1,2,1],
      [1,3,1],
      [3,1,3], 
      [3,2,3]]
  for ctr in color_seq:
    if(colors==color_seq[index]):
      pig_id[index][0] = pig_loc[0]
      pig_id[index][1] = pig_loc[1]
      print("PIG_ID:",index)
      break
    else:
      index=index+1
def get_data():
  return(pig_id)

###################################################################################################
cap = cv2.VideoCapture('rrr.avi')


if (cap.isOpened()== False): 
  print("Error opening video stream or file")
  
while (cv2.waitKey(0)):
  ret, img2 = cap.read()
   
  rows,cols,channels = img2.shape
  roi = img2[0:rows,0:cols]
  kernel = np.ones((5,5), np.uint8)   #no idea wtf this does

  
  img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) #converts image to grayscale

  ret,mask = cv2.threshold(img2gray,207,255,cv2.THRESH_BINARY) # converts image to binary (BW)
  #w3w = cv2.adaptiveThreshold(mask,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,59,2) # adaptive threshold w/ gaussian method

  #blur = cv2.GaussianBlur(w3w,(5,5),0)
  #t3,th3 = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
  #cv2.imshow('inv mask',th3)
  
  mask_inv = cv2.bitwise_not(mask) # invert mask B AND W
  img2_fg = cv2.bitwise_and(img2, img2, mask=mask_inv)

  blur = cv2.blur(img2_fg,(2,2))

 # cv2.imshow('inv mask',blur) # ok here
  


  hsv = cv2.cvtColor( blur, cv2.COLOR_BGR2HSV)

  lower_red = np.array([0,50,50])
  upper_red = np.array([2,185,255])
  mask0 = cv2.inRange(hsv, lower_red, upper_red)

  # upper mask (170-180) RED
  lower_red = np.array([170,50,50])
  upper_red = np.array([180,255,255])
  mask1 = cv2.inRange(hsv, lower_red, upper_red)

  # lower mask (0-10) BLUE
  lower_blue = np.array([101,50,0])
  upper_blue = np.array([150,210,255])
  mask2 = cv2.inRange(hsv, lower_blue, upper_blue)

  # upper mask (170-180) BLUE
  # lower_blue = np.array([130,50,50])
  # upper_blue = np.array([110,255,255])
  # mask3 = cv2.inRange(hsv, lower_blue, upper_blue)

  # lower mask (0-10) GREEN
  lower_green = np.array([46,30,0])
  upper_green = np.array([100,255,255])
  mask4 = cv2.inRange(hsv, lower_green, upper_green)



  # join my masks
  mask = mask0+mask1+mask2+mask4

  # set my output img to zero everywhere except my mask
  output_img = img2_fg.copy()
  output_img = cv2.bitwise_and(output_img, output_img, mask= mask)

  # or your HSV image, which I *believe* is what you want
  output_hsv = hsv.copy()
  output_hsv = cv2.bitwise_and(output_img, output_img, mask= mask)

  image = output_hsv
  width = 3

  #hsv_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  #hsv_gray = cv2.GaussianBlur(hsv_gray, (7, 7), 0)
 # cv2.imshow("Image", hsv_gray)
 # cv2.waitKey(0)
  edged = cv2.Canny(image, 200, 100)
  edged = cv2.dilate(edged, None, iterations=1)
  edged = cv2.erode(edged, None, iterations=1)
#  cv2.imshow("Image", edged)
 #cv2.waitKey(0)
  contr= cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  contr = contr[1]
  ct_cord=cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)



  # 'pixels per metric' calibration variable
  pixelsPerMetric = None
  TIP=(0,0)

  pigIDcntr=0
  # loop over the each contours 
  for c in contr:

    # Threshold for countor size
    if cv2.contourArea(c) < 100:
      continue


    # rotated bounding box of each contour
    orig = image.copy()
    box = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")
    
    # order the points in the contour 
    # top-left, top-right, bottom-right, and bottom-left
    box = order_points(box)
    #extract coordinates from the rouded box
    (tl, tr, br, bl) = box
    #  midpoints between the top-left, top-right points, top-righT and bottom-right
    (tltrX, tltrY) = midpoint(tl, tr)
    (blbrX, blbrY) = midpoint(bl, br)
    (tlblX, tlblY) = midpoint(tl, bl)
    (trbrX, trbrY) = midpoint(tr, br)


    # distance between the midpoints
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))
    
    # for init the pixel ratio
    # calculate the ratio of pixels to supplied metric
    if pixelsPerMetric is None:
      pixelsPerMetric = dB / width

    #################################COLOR DETECTION############################################
    (cdtltrX,cdtltrY) = detect_color1 (tl,tr)
    (cdblbrX,cdblbrY) = detect_color1 (bl,br)
    (cdtlblX,cdtlblY) = detect_color1 (tl,bl)
    (cdtrbrX,cdtrbrY) = detect_color1 (tr,br)

    cX = np.average(box[:, 0])
    cY = np.average(box[:, 1])
    # cv2.circle(orig, (int(cX), int(cY)), 5, (255, 0, 0), -1)

    #cv2.circle(orig, (int(cX+35), int(cY-50)), 5, (255, 0, 0), -1)
    #(upleftx,uplefty) = midpoint((tlblX,tlblY),(tltrX,tltrY))
    #(uprightx,uprighty) = midpoint((trbrX,trbrY),(tltrX,tltrY))
    #(color1X,color1Y)=midpoint((upleftx,uplefty),(uprightx,uprighty))
    (ctopX,ctopY)=midpoint((cX,cY),(tlblX,tltrY))

    ##########################EXTREME#POINTS####################################################
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    (extLeft_x, extLeft_y)=extLeft

    extRight = tuple(c[c[:, :, 0].argmax()][0])
    (extRight_x,extRight_y)=extRight

    extTop = tuple(c[c[:, :, 1].argmin()][0])
    (extTop_x,extTop_y)=extTop

    extBot = tuple(c[c[:, :, 1].argmax()][0])
    (extBot_x,extBot_y)=extBot  
    ################FROM TOP###################################################################
    (color1_topX,color1_topY)=midpoint((cX,cY),(extTop_x,extTop_y))
    (color2_topX,color2_topY)=(cX,cY)
    (tempc3_topX,tempc3_topY)=midpoint((cX,cY),(blbrX,blbrY))
    (color3_topX,color3_topY)=midpoint((tempc3_topX,tempc3_topY),(blbrX,blbrY))


    ################FROM BOT###################################################################
    (color1_botX,color1_botY)=midpoint((cX,cY),(extBot_x,extBot_y))
    (color2_botX,color2_botY)=(cX,cY)
    (tempc3_botX,tempc3_botY)=midpoint((cX,cY),(tltrX,tltrY))
    (color3_botX,color3_botY)=midpoint((tempc3_botX,tempc3_botY),(tltrX,tltrY))
    ################FROM LEFT###################################################################
    (color1_leftX,color1_leftY)=midpoint((cX,cY),(extLeft_x,extLeft_y))
    (color2_leftX,color2_leftY)=(cX,cY)
    (tempc3_leftX,tempc3_leftY)=midpoint((cX,cY),(trbrX,trbrY))
    (color3_leftX,color3_leftY)=midpoint((tempc3_leftX,tempc3_leftY),(trbrX,trbrY))
    ################FROM RIGHT###################################################################
    (color1_rightX,color1_rightY)=midpoint((cX,cY),(extRight_x,extRight_y))
    (color2_rightX,color2_rightY)=(cX,cY)
    (tempc3_rightX,tempc3_rightY)=midpoint((cX,cY),(tlblX,tlblY))
    (color3_rightX,color3_rightY)=midpoint((tempc3_rightX,tempc3_rightY),(tlblX,tlblY))
    cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),(255, 0, 255), 2)
    cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),(255, 0, 255), 2)
    ################################Tip#Detect##################################################
  
    if(dA>dB):
      
      if(topbot()==3):
        #print(topbot())
        #side_tltr=cv2.line(orig, tuple(tl),tuple(tr), (0,255,0), 2)
        #line1=cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),(255, 0, 255), 2)
        #line2=cv2.line(orig, (int(cdtlblX), int(cdtlblY)), (int(cdtrbrX), int(cdtrbrY)),(255, 0, 255), 2)
        #line_intersection(line1,line2)
        print("TOP")
        cv2.circle(orig, (int(color1_topX), int(color1_topY)), 5, (0, 255, 0), -1)
        cv2.circle(orig, (int(color2_topX), int(color2_topY)), 5, (0, 255, 0), -1)
        cv2.circle(orig, (int(color3_topX), int(color3_topY)), 5, (0, 255, 0), -1)
        cv2.putText(orig, "TIP",(int(extTop_x - 30), int(extTop_y - 10)), cv2.FONT_HERSHEY_SIMPLEX,0.65, (255, 255, 255), 2)
        cv2.circle(orig, (int(extTop_x), int(extTop_y)),5 , (0, 0, 255), -1)
        
        colorseq_top1=(int(color1_topX),int(color1_topY))
        colorseq_top2=(int(color2_topX),int(color2_topY))
        colorseq_top3=(int(color3_topX),int(color3_topY))
        colorseq_top=[colorseq_top1,colorseq_top2,colorseq_top3]

        color_picker(colorseq_top,extTop)
        #################FEEDER#DISTANCE#####################
        #print("DISTANCE FROM FEEDER:",dist.euclidean((extTop_x, extTop_y), (627,528)))
      elif(topbot()==4):
        #print(topbot())
        #side_blbr=cv2.line(orig, tuple(bl),tuple(br), (0,255,0), 2)
        cv2.circle(orig, (int(color1_botX), int(color1_botY)), 5, (0, 255, 0), -1)
        cv2.circle(orig, (int(color2_botX), int(color2_botY)), 5, (0, 255, 0), -1)
        cv2.circle(orig, (int(color3_botX), int(color3_botY)), 5, (0, 255, 0), -1)      

        cv2.putText(orig, "TIP",(int(extBot_x - 30), int(extBot_y- 10)), cv2.FONT_HERSHEY_SIMPLEX,0.65, (255, 255, 255), 2)
        cv2.circle(orig, (int(extBot_x), int(extBot_y)),5 , (0, 0, 255), -1)
        
        colorseq_bot1=(int(color1_botX),int(color1_botY))
        colorseq_bot2=(int(color2_botX),int(color2_botY))
        colorseq_bot3=(int(color3_botX),int(color3_botY))
        colorseq_bot=[colorseq_bot1,colorseq_bot2,colorseq_bot3]

        #################FEEDER#DISTANCE#####################
        #print("DISTANCE FROM FEEDER:",dist.euclidean((extBot_x, extBot_y), (627,528)))
        #PickColor(colorseq_bot)
        color_picker(colorseq_bot,extBot)
    else:
      
      if(leftright()==1):
        #print(leftright())
        #side_tlbl=cv2.line(orig, tuple(tl),tuple(bl), (0,255,0), 2)
        cv2.circle(orig, (int(color1_leftX), int(color1_leftY)), 5, (0, 255, 0), -1)
        cv2.circle(orig, (int(color2_leftX), int(color2_leftY)), 5, (0, 255, 0), -1)
        cv2.circle(orig, (int(color3_leftX), int(color3_leftY)), 5, (0, 255, 0), -1)

        cv2.putText(orig, "TIP",(int(extLeft_x + 10), int(extLeft_y)), cv2.FONT_HERSHEY_SIMPLEX,0.65, (255, 255, 255), 2)
        cv2.circle(orig, (int(extLeft_x), int(extLeft_y)),5 , (0, 0, 255), -1)
        #################FEEDER#DISTANCE#####################
        #print("DISTANCE FROM FEEDER:",dist.euclidean((extLeft_x, extLeft_y), (627,528)))
      # PickColor(colorseq_left)
        colorseq_left1=(int(color1_leftX),int(color1_leftY))
        colorseq_left2=(int(color2_leftX),int(color2_leftY))
        colorseq_left3=(int(color3_leftX),int(color3_leftY))
        colorseq_left=[colorseq_left1,colorseq_left2,colorseq_left3]
        color_picker(colorseq_left,extLeft)

      elif(leftright()==2):
        #print(leftright())
        #side_trbr=cv2.line(orig, tuple(tr),tuple(br), (0,255,0), 2)
        cv2.circle(orig, (int(color1_rightX), int(color1_rightY)), 5, (0, 255, 0), -1)
        cv2.circle(orig, (int(color2_rightX), int(color2_rightY)), 5, (0, 255, 0), -1)
        cv2.circle(orig, (int(color3_rightX), int(color3_rightY)), 5, (0, 255, 0), -1)

        cv2.putText(orig, "TIP",(int(extRight_x + 10), int(extRight_y)), cv2.FONT_HERSHEY_SIMPLEX,0.65, (255, 255, 255), 2)
        cv2.circle(orig, (int(extRight_x), int(extRight_y)),5 , (0, 0, 255), -1)

        colorseq_right1=(int(color1_rightX),int(color1_rightY))
        colorseq_right2=(int(color2_rightX),int(color2_rightY))
        colorseq_right3=(int(color3_rightX),int(color3_rightY))
        colorseq_right=[colorseq_right1,colorseq_right2,colorseq_right3]
        color_picker(colorseq_right,extRight)
    cv2.imshow("Image", orig)
    cv2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  print(get_data())      
cap.release()
cv2.destroyAllWindows()



  #add = img2+img1                               #slight useful kind of addition
  #add = cv2.add(img1,img2)                      #useless shit
  #add = cv2.addWeighted(img2, 0.4,img2, 0.6, 0) #useful shit

