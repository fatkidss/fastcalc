import math
import numpy as np

class CalcArea:
    data = []
    density = 6.9e-6 # metal density
    def __init__(self,h,f,q,p,c,name,gthickness=4):
        self.h = h  # head 
        self.f = f  # loss factor
        self.q = q  # flow rate
        self.p = p  # casting height over gate
        self.c = c # total casting height
        # self.d = 6.9e-6 # metal density
        self.name = name
        self.gate_h = gthickness

    def velocity_ave(self):
        he = self.h - self.p**2/(2*self.c)
        v = self.f*140*math.sqrt(he)
        return v
    
    def velocity_graph(self):
        he = self.h - np.linspace(0,self.p,50)
        v = self.f*140*np.sqrt(he)
        return (he,v)
    
    def choke(self):
        v = self.velocity_ave()
        area = self.q/(CalcArea.density*v)
        return area
    
    def size(self):
        sname = self.name.split("_")
        ratio = float(sname[1])
        if sname[0][0].upper() == 'C':
            area = self.choke()
            a = math.sqrt(area/2)
            width = a*1.35
            h = width/math.cos(math.pi/6)
            return [self.name, area, a, (a*0.65,a*1.35) , h]
        if sname[0][0].upper() == 'R':
            area = self.choke()*ratio
            a = math.sqrt(area/2)
            width = a*1.35
            h = a*2
            return [self.name, area, a, (a*0.65,a*1.35), h]
        if sname[0][0].upper() == 'I':
            area = self.choke()*ratio
            h = self.gate_h
            width = int(area/h)
            return [self.name, area, width, (width-1,width+1), h]
        return [self.name, 0, 0, (0,0), 0]
    
    def ranold(self):
        name, area, width, wminmax, height = self.size()
        v = self.q/(CalcArea.density*area)
        ran = 6.8*v*area/(2*(width+height))
        return ran
    
    def erosion(self):
        name, area, width, wminmax, height = self.size()
        v = self.q/(CalcArea.density*area)
        sx = CalcArea.density*(v**2)
        sy = CalcArea.density*v*(140)*math.sqrt(self.c-self.p)
        return sx,sy

    def show(self):
        name, area, width, wminmax, height = self.size()
        ran = self.ranold()
        sx,sy = self.erosion()
        print(f'name:{name} , area:{area:.0f} mm2 , width:{width:.0f} : ({wminmax[0]:.0f},{wminmax[1]:.0f} ) mm , height:{height:.1f} mm,\nranold no.:{ran:.0f} stress(sx,sy): ({sx:.2f},{sy:.2f})')

    def save(self):
        name, area, width, (wmin,wmax), height = self.size()
        ran = self.ranold()
        sx,sy = self.erosion()
        CalcArea.data.append([round(self.h,1),round(self.f,2),round(self.q,2),round(self.p,1),round(self.c,1),name,
                              round(area,2),round(width,2),round(wmin,0),round(wmax,0),round(height,2),round(ran,0),round(sx,1),round(sy,1)])
        return [round(self.h,1),round(self.f,2),round(self.q,2),round(self.p,1),round(self.c,1),
                                name,round(area,2),round(width,2),(wmin,wmax) ,round(height,2),round(ran,0),(sx,sy)]
    
    def remove(self):
        CalcArea.data.pop()
        # del CalcRiser.data[-1]

class CalcRiser:
    data = []
    density = 7.2e-6
    ratio = {'FC':{'casting':3, 'neck':0.35, 'riser':1.2, 'TS':300},
            'FCD':{'casting':4, 'neck':0.45, 'riser':1.4, 'TS':550}
            }
    hot_fac = 0.17
    cold_fac = 0.12

    def __init__(self,mat,cwt,cmod,cold=False,nhigh=0.0):
        self.mat = mat
        self.cwt = cwt
        self.cmod = cmod
        self.cold = cold
        self.nhigh = nhigh

    def calModulus(self):
        mat = 'FCD' if self.mat[:3].upper() == 'FCD' else 'FC'
        nmod = self.cmod*self.ratio[mat]['neck']
        rmod = self.cmod*self.ratio[mat]['riser']
        return nmod, rmod
    
    def sizeNeck(self):
        nmod, rmod = self.calModulus()
        width = 6*10*nmod
        hight = width/2
        try:
            if self.nhigh != 0.0 :
                w = 2*10*nmod*self.nhigh/(self.nhigh-2*10*nmod)
                h = self.nhigh
                if w > 0:
                    width = w
                    hight = h
                else :
                    print('can not create width --> use automatic calculate neck height')
        except:
            print('can not create width --> use automatic calculate neck height')
        length = 1.3*hight
        return width, hight, length
    
    def sizeRiser(self):
        nmod, rmod = self.calModulus()
        base = 5.5*10*rmod
        top = base - 4*10*self.cmod
        h = 1.5*base
        wt = math.pi*h/12*(base**2+base*top+top**2)*CalcRiser.density
        return base,top,h,wt
    
    def enoughRiser(self):
        mat = 'FCD' if self.mat[:3].upper() == 'FCD' else 'FC'
        base,top,h,wt = self.sizeRiser()
        if self.cold :
            factor =  round((wt*CalcRiser.cold_fac)/(self.cwt*self.ratio[mat]['casting']/100),2)
        else :
            factor =  round((wt*CalcRiser.hot_fac)/(self.cwt*self.ratio[mat]['casting']/100),2)
        return factor
    
    def neckForce(self,lx=30,ly=30):
        mat = 'FCD' if self.mat[:3].upper() == 'FCD' else 'FC'
        s = CalcRiser.ratio[mat]['TS']
        width, hight, length = self.sizeNeck()
        fx = s*width*(hight**2)/6/lx
        fy = s*hight*(width**2)/6/ly
        return fx,fy

    def show(self):
        width, hight, length = self.sizeNeck()
        base,top,h,wt = self.sizeRiser()
        factor = self.enoughRiser()
        fx,fy = self.neckForce()
        txtR = f'Riser size --> BaseDia {base:.2f} mm : TopDia {top:.2f} mm : height {h:.2f} mm : Weight {wt:.2f} kg\n'
        txtN = f'Neck size ---> Width {width:.0f} mm : Hight {hight:.0f} mm : Length {length:.0f} mm\n'
        txtF = f'Riser feed ratio: {factor:.2f}'
        txtP = f'Packing force (fx,fy): {fx:.2f} , {fy:.2f}'
        print(txtN+txtR+txtF+txtP)

    def save(self):
        nmod, rmod = self.calModulus()
        width, hight, length = self.sizeNeck()
        base,top,h,wt = self.sizeRiser()
        factor = self.enoughRiser()
        fx,fy = self.neckForce()
        CalcRiser.data.append([self.mat,round(self.cwt,2),round(self.cmod,2),self.cold,
                          round(nmod,3),round(rmod,3),
                          round(width,2),round(hight,2),round(length,2),
                          round(base,2),round(top,2),round(h,2),round(wt,2),round(factor,2),
                          round(fx,0),round(fy,2)
                          ])
        return [self.mat,round(self.cwt,2),round(self.cmod,2),self.cold,
                          round(nmod,3),round(rmod,3),
                          round(width,2),round(hight,2),round(length,2),
                          round(base,2),round(top,2),round(h,2),round(wt,2),round(factor,2),
                          round(fx,0),round(fy,2)
                          ]
    def remove(self):
        CalcRiser.data.pop()
        # del CalcRiser.data[-1]

    
    




        

        











    
        
    



