import numpy as np
import pandas as pd
import random
import cv2
import sys

mod_dicts = {}
scale = 6
WIDTH,HEIGHT = 800,600
random.seed(121)
COLOR = [(0,0,255),(0,125,255),(0,200,225),(0,255,0),(125,255,0),(255,255,0),(255,0,0),(255,0,125),(255,0,255),(255,125,255)]

def cooling_edge(mod_dicts,pixel_mm):
    poly = []
    pair = []
    for i in range(1,len(mod_dicts)+1):
        pts_i = mod_dicts[i]["polys"]
        for i1 in range(i+1,len(mod_dicts)+1):
            pts_i1 = mod_dicts[i1]["polys"]
            p = 0
            for ip in pts_i:
                for jp in pts_i1:
                    if abs(ip[0]-jp[0])<10 and abs(ip[1]-jp[1])<10:
                        pair.append(ip)
                        p +=1
            if len(pair) > 1:
                poly.append([(i,i1),pair])
                l = 0
                for il in range(len(pair)-1):
                    x0,y0 = pair[il]
                    x1,y1 = pair[il+1]
                    l += np.sqrt((x1-x0)**2+(y1-y0)**2)
                l = l/pixel_mm
                mod_dicts[i]["pairs"].append([(i,i1),pair,l])
                mod_dicts[i1]["pairs"].append([(i1,i),pair,l])
                pair =[]
    return poly

def calc_area_peri(polys):
    s1,s2,sl = 0,0,0
    for i in range(len(polys)):
        x1,y1 = polys[i]
        if i < len(polys)-1 :
            x2,y2 = polys[i+1]
        else :
            x2,y2 = polys[0]
        s1 += x1*y2
        s2 += y1*x2
        sl += np.sqrt((x2-x1)**2+(y2-y1)**2)
    area = 0.5*abs(s1-s2)
    perimeter = sl
    return area,perimeter

def update_modulus(mod_dicts,mat,pixel_mm):
    rate = 4.4 if mat == "FCD" else 3.3
    poly = cooling_edge(mod_dicts,pixel_mm)
    for idx,v in mod_dicts.items():
        pts = mod_dicts[idx]["polys"]
        pts = np.array(pts)
        area,perimeter = calc_area_peri(pts)
        area = area/(pixel_mm**2)
        perimeter = perimeter/(pixel_mm)
        cl = 0
        for pair,pts,l in mod_dicts[idx]["pairs"]:
            cl = cl+l
        # cl = cl/pixel_mm
        modulus = 0.1*area/(perimeter-cl)
        sot = rate*(modulus**2)
        sht = 0.5*sot
        mod_dicts[idx]["area"] = area
        mod_dicts[idx]["perimeter"] = perimeter
        mod_dicts[idx]["modulus0"] = 0.1*area/perimeter
        mod_dicts[idx]["modulus1"] = modulus
        mod_dicts[idx]["coolingL"] = cl
        mod_dicts[idx]["SOT"] = sot
        mod_dicts[idx]["SHT"] = sht
    return poly,mod_dicts

def draw_feeding(img,mod_dicts):
    for idx in range(len(mod_dicts)):
        pairs = mod_dicts[idx+1]["pairs"]
        for pair,pts,l in pairs:
            mo1,mo2 = pair
            sot = mod_dicts[mo1]["SOT"]
            sht = mod_dicts[mo2]["SHT"]
            center1 = mod_dicts[mo1]["center"]
            center2 = mod_dicts[mo2]["center"]
            dist = np.sqrt((center1[0]-center2[0])**2+(center1[1]-center2[1])**2)
            dx,dy = center2[0] - center1[0],center2[1] - center1[1]
            if sot >= sht :
                cv2.line(img,center1,(int(center1[0]+0.3*dx),int(center1[1]+0.3*dy)),(255,50,50),1)
                cv2.circle(img,(int(center1[0]+0.3*dx),int(center1[1]+0.3*dy)),2,(255,50,50),-1)

def show_modulus(img,mod_dicts,polys):
    for idx in range(len(mod_dicts)):
        points = mod_dicts[idx+1]["polys"]
        points = np.array(points)
        x,y = mod_dicts[idx+1]["center"]
        # print(points)
        cv2.fillPoly(img,[points],COLOR[idx])
        cv2.circle(img, (x,y), 4, (100, 100, 100), -1)
        cv2.putText(img,f'{idx+1}({mod_dicts[idx+1]["modulus0"]:.2f},{mod_dicts[idx+1]["modulus1"]:.2f})',(x-30,y+30),
                    cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1)
    for pair,points in polys:
        pts = np.array(points)
        cv2.polylines(img,[pts],False,(0,0,0),2)
    draw_feeding(img,mod_dicts)
    # cv2.imshow('image',img)
    # cv2.waitKey(0)
    # cv2.imwrite('modulus_feeding.jpg',img)
    return img

def modulus_diagram(img,mod_dicts):
    df = pd.DataFrame(data=mod_dicts.values(),index=mod_dicts.keys())
    m = np.array(df.loc[:,"modulus1"])
    sot = np.array(df.loc[:,"SOT"])
    df.to_csv('modulus_data.csv')
    offset = 5

    # img = np.zeros((HEIGHT,WIDTH,3), np.uint8)+255
    max_mo = max(m)

    nw = len(m) if len(m)> 5 else 5

    w = int((WIDTH-200)/nw) - offset

    for i in range(len(m)):
        h = int(m[i]*(HEIGHT-200)/max_mo)
        x = 100+ i*(w+offset)
        y = HEIGHT - 100 - h
        cv2.rectangle(img,(x,y),(x+w,y+h),(20,125,255),-1)
        cv2.putText(img,"SOT",(int(x+0.3*w),int(y+0.5*h-10)),cv2.FONT_HERSHEY_PLAIN,1.0,(0,0,0),1)
        cv2.putText(img,str(round(sot[i],1))+" min",(int(x+0.15*w),int(y+0.5*h+10)),cv2.FONT_HERSHEY_PLAIN,0.8,(0,0,0),1)
        cv2.putText(img,str(round(m[i],2)),(int(x+0.3*w),y-10),cv2.FONT_HERSHEY_PLAIN,1.0,(0,0,0),1)
        cv2.putText(img,str(i+1),(int(x+0.4*w),y+h+20),cv2.FONT_HERSHEY_PLAIN,1.0,(0,0,0),1)
    cv2.putText(img,"Modulus diagram",(int(0.2*WIDTH),50),cv2.FONT_HERSHEY_PLAIN,2.5,(255,0,0),1)  
    cv2.line(img,(100,HEIGHT-100),(WIDTH-80,HEIGHT-100),(0,0,0),2)
    cv2.line(img,(100,HEIGHT-100),(100,80),(0,0,0),2)
    # cv2.imshow('image',img)
    # cv2.waitKey(0)
    # cv2.imwrite('modulus_diagram.jpg',img)
    return img

def record_video(fname="record.mp4",width=WIDTH,height=HEIGHT,fps=30):
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Create VideoWriter object. We use the same properties as the input camera.
    # Last argument is False to write the video in grayscale. True otherwise (write the video in color)
    out = cv2.VideoWriter(fname, fourcc, fps, (width, height), True)   #(1466, 1030)
    return out

from imutils import contours,resize

def read_dwg_img(image):
    # Load image, grayscale, Otsu's threshold
    # image = cv2.imread(img_path)
    image = resize(image,height=600)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3,3), 0)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    # Filter using contour hierarchy
    cnts, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    hierarchy = hierarchy[0]
    smoothen_contours = []
    polys = []
    factor = 0.005
    ct = 0
    for component in zip(cnts, hierarchy):
        currentContour = component[0]
        currentHierarchy = component[1]
        x,y,w,h = cv2.boundingRect(currentContour)
        epsilon = factor * cv2.arcLength(currentContour, True)
        cnt = cv2.approxPolyDP(currentContour, epsilon, True)
        
        # Has inner contours which means it is IN
        if currentHierarchy[2] < 0:
            # cv2.putText(image, 'IN', (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36,255,12), 2)
            if w < 10 or h < 10:
                continue
            ct +=1
            smoothen_contours.append(cnt)
            cv2.drawContours(image, [cnt], 0, color=(0,125,255), thickness=2)
            
            # print(cnt)
            # cv2.imshow('image', image)
            # cv2.waitKey()
        # No child which means it is OUT
        # elif currentHierarchy[3] < 0:
        #     # cv2.putText(image, 'OUT', (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36,255,12), 2)
        #     cv2.drawContours(image, [cnt], 0, color=(0,0,255), thickness=1)
    # cv2.imshow('thres',thresh)
    # cv2.imshow('image', image)
    # cv2.waitKey()
    (smoot_cnts, _) = contours.sort_contours(smoothen_contours, method="left-to-right")
    # print(len(smoot_cnts))
    smoot_cnts = list(smoot_cnts)
    scale_bx = smoot_cnts.pop()
    x,y,w,h = cv2.boundingRect(scale_bx)
    for poly in smoot_cnts:
        pts = poly.reshape(-1,2)
        polys.append(list(pts))
    return polys,h

def main_dwg_img(img):
    idx = 0
    mod_dicts = {}

    polys,scale_px = read_dwg_img(img)
    scale = scale_px/10
    for cnt in polys:
        idx +=1
        cntarr = np.array(cnt)
        M = cv2.moments(cntarr)
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
        # print(center)
        area = cv2.contourArea(cntarr)/(scale**2)
        perimeter = cv2.arcLength(cntarr,True)/scale
        modulus = 0.1*area/perimeter
        # print(cntarr)
        mod_dicts[idx] = {  "polys":cntarr,
                                        "area":area,
                                        "perimeter":perimeter,
                                        "modulus0":modulus,
                                        "modulus1":modulus,
                                        "center":center,
                                        "pairs": [],
                                        "coolingL":0,
                                        "SOT":0,
                                        "SHT":0
                                        }
    # print(scale)
    print(mod_dicts)

    init_img = np.zeros((HEIGHT,WIDTH,3), np.uint8)+225
    dwg_img = init_img.copy()
    # cv2.namedWindow('image') 
    poly,mod_dicts = update_modulus(mod_dicts,"FCD",scale)
    print(mod_dicts)
    img_mod = show_modulus(dwg_img,mod_dicts,poly)
    dwg_img = init_img.copy()
    img_dig = modulus_diagram(dwg_img,mod_dicts)
    return img_mod,img_dig

if __name__ == "__main__" :
    path = './images/mt.jpg'
    main_dwg_img(path)