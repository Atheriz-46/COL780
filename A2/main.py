import cv2 as cv
import pandas as pd 
import numpy as np 
import math
import os
import re


class Transformation:
    def transform(self,p):
        pass
    
    def inv_transform(self,p):
        pass

class Projective(Transformation):
    def __init__(self,p):
        self.H = p 
    def transform(self,img,box,dp=np.zeros((3,3))):
        x,y,w,h = box
        p = self.H 
        b = bilinear_interpolate(img)
        dp.resize((3,3),refcheck=False)
        def func(x_,y_):
            res=(p+dp)@np.array([x_,y_,1])
            res = res/res[2]
            return b(res[0],res[1])
        Iw = np.zeros((h+1,w+1))
        for j in range(h+1):
            for i in range(w+1):
                Iw[j,i] = func(y+j,x+i)
        return Iw
    def get_box(self,box):
        x,y,w,h = box
        coords = self.H@np.array([[y,x,1],[y,x+w,1],[y+h,x+w,1],[y+h,x,1]]) 
        coords = coords/coords[:,-1]
        y,x,_ = np.min(coords,axis=1)       
        h,w,_ = np.max(coords,axis=1)      
        return [mat.floor(y),math.floor(x),math.ceil(y-h),math.ceil(x-w)] 
    def select(self,dp):
        print(dp)
        self.H += dp.resize((3,3),refcheck=False)


def bilinear_interpolate(img):
    def image(x,y):
        if y>=img.shape[1] or y<0 or x>=img.shape[0] or x<0:
            raise IndexError
        i,j = int(x),int(y)
        a,b = x-i,y-j
        return (1-a)*((1-b)*img[i,j]+b*img[i,j+1]) + a*((1-b)*img[i+1,j]+b*img[i+1,j+1])
    return image

def NSSE(t,f):
    return -np.linalg.norm(f-t)**2
def NCC(t,f):
    return np.sum(f*t)/np.std(f)

def IOU(box1,box2):
    boxA,boxB = box1,box2
    boxA[2:]+=boxA[:2]
    boxB[2:]+=boxB[:2]
    xA = max(boxA[0],boxB[0])
    yA = max(boxA[1],boxB[1])
    xB = min(boxA[2],boxB[2])
    yB = min(boxA[3],boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def block_based(dir,p_0,delp,n_p,outfile,metric = NSSE):
    sample = [np.linspace(-dp,dp,n) for dp,n in zip(delp,n_p)]
    sample = np.meshgrid(*sample)
    inp_path = os.path.join(dir,'img')
    gt = np.genfromtxt(os.path.join(dir,'groundtruth_rect.txt'), delimiter=',')
    gt = np.int32(gt)
    box = gt[0]
    gt = gt[1:]
    files = sorted(os.listdir(inp_path))
    files = files[1:]
    template = cv.imread(os.path.join(inp_path,files[0]))
    template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
    # 
    template_cut = template[box[1]:box[1]+box[3]+1,box[0]:box[0]+box[2]+1]
    homo = Projective(p_0)
    output = []
    sIOU,n = 0.,0.
    for file in files:
        if file[-4:] in ['.jpg']:
            frame = cv.imread(os.path.join(inp_path,file))
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            if frame is None: break             
            frame_no = int(re.search(r'[0-9]+',file)[0])
   #################
            max_,best_p=None, None
            for dp in sample:
                try:
                    Iw = homo.transform(frame,box,dp)
                    k = metric(template_cut,Iw)
                    if(not max_ or k>max_): 
                        best_p = dp
                        max_ = k 
                except:
                    continue
            homo.select(best_p)
            box_t = homo.get_box(box)
            output.append(box_t)

            ''' Score '''
            sIOU += IOU(box_t,box)
            n+=1.
            
            frame = cv.rectangle(frame,(box_t[1],box_t[0]) ,(box_t[1]+box_t[3],box_t[2]+box_t[0]) , (255,0,0), 2)
            frame = cv.rectangle(frame,(gt[frame_no-2,1],gt[frame_no-2,0]) ,(gt[frame_no-2,1]+gt[frame_no-2,3],gt[frame_no-2,2]+gt[frame_no-2,0]) , (0,255,0), 2)
            cv.imshow('Image',frame)
            if cv.waitKey(60) & 0xFF == ord('q'):
                break
            

    
    numpy.savetxt(outfile, np.array(output), delimiter=",")
    print('mIOU score: '+ str(sIOU/n*100))


'''Needs lots of work'''
class LK:
    def __init__(self,template,box ,p = np.eye(3)):
        self.geometry = Projective(p)
        self.template_img = template
        self.box = box
        self.template =  template[box[1]:box[1]+box[3]+1,box[0]:box[0]+box[2]+1]
        self.p = p

    
    def fit(self,img):
        self.img = img
        while(np.linalg.norm(dp)>self.tol):
            t1 = np.matmul(self.del_I(),self.del_W()) 
            H  = np.einsum('ijkl,ijkm->lm',t1,t1)
            Iw = self.geometry.transform(img,self.box)
            dp = np.linalg.inv(H).dot(np.einsum('ijkl,ij->lk',t1,self.template-Iw))
            self.p+=dp.resize((3,3),refcheck=False)
        return self.geometry.get_box(self.box)
    def del_I(self):
        '''h x w x 1 x 2'''
        dx = np.array([[-1,0,1],
                    [-2,0,2],
                    [-1,0,1]])
    
        I = np.array([[cv.filter2D(Iw, -1, dx, borderType=cv.BORDER_CONSTANT),
            cv.filter2D(Iw, -1, dx.T, borderType=cv.BORDER_CONSTANT)]])
        return I.transpose((2,3,0,1))
    def del_W(self):
        x = np.arange(self.box[0],self.box[0]+self.box[2]+1)
        y = np.arange(self.box[1],self.box[1]+self.box[3]+1)
        xx,yy = np.meshgrid(x,y,sparse =True)
        
        return self.Wp_affine(xx,yy)
    def Wp_affine(self,x,y):
        return np.array([[x,0,y,0,1,0],[0,x,0,y,0,1]]).transpose((2,3,0,1))#2 x p x h x w
    
    def Wp_projective(self,x,y):
        xp = p.dot(np.array([x,y,1]))
        xp,yp,c = xp
        xp,yp = xp/c,yp/c 
        return (np.array([[x,y,1,0,0,0,-x*xp,-y*xp],[0,0,0,x,y,1,-x*yp,-y*yp]])/c).transpose((2,3,0,1))

delp = [0.5*1e-2]*8
n_p = [5]*8
block_based('./A2/BlurCar2',np.eye(3),delp,n_p,'./A2/BlurCar2/outfile')
# def LK():
    # pass
