import cv2 as cv
# import pandas as pd 
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
        # y,x,h,w = box
        p = self.H 
        b = bilinear_interpolate(img)
        dp =dp.copy()
        dp.resize((3,3),refcheck=False)
        
        def func(x_,y_):
            res=(p+dp)@np.array([x_,y_,1])
            x_,y_,_ = res/res[2]
            return b(x_,y_)
        Iw = np.zeros((h+1,w+1))
        for j in range(h+1):
            for i in range(w+1):
                Iw[j,i] = func(x+i,y+j)
        return Iw
    def get_box(self,box):
        x,y,w,h = box
        coords = (self.H @ np.array([[x,y,1],[x+w,y,1],[x+w,y+h,1],[x,y+h,1]]).T).T
        print(coords)
        print(self.H)
        coords = coords/coords[:,-1:]
        # print(np.min(coords,axis=0))
        x_,y_,_ = np.min(coords,axis=0)       
        w_,h_,_ = np.max(coords,axis=0)      
        ans = [math.floor(x_),math.floor(y_),math.ceil(w_-x_+1),math.ceil(h_-y_+1)]
        print(box)
        print(ans)
        print()
        return ans 
    def select(self,dp):
        # print(dp)
        # print(dp)
        dp =dp.copy()
        dp.resize((3,3),refcheck=False)
        # print(dp)
        self.H += dp


def bilinear_interpolate(img):
    def image(x,y):

        if y>=img.shape[0]-1 or y<0 or x>=img.shape[1]-1 or x<0:
            # print(x,y)
            raise IndexError
        j,i = int(x),int(y)
        b,a = x-i,y-j
        return (1-a)*((1-b)*img[i,j]+b*img[i,j+1]) + a*((1-b)*img[i+1,j]+b*img[i+1,j+1])
    return image

def NSSE(t,f):
    return -np.linalg.norm(f-t)**2
def NCC(t,f):
    return np.sum(f*t)/np.std(f)

def IOU(box1,box2):
    boxA,boxB = box1.copy(),box2.copy()
    # print(boxA,boxB)
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
    sample = np.stack(np.meshgrid(*sample)).reshape((-1,len(delp)))

    # print(sample.shape,len(sample))
    # return
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
            max_,best_p=None, np.zeros(6)
            # print('lol')
            for dp in sample:
                # print(dp.shape)
                try:
                    Iw = homo.transform(frame,box,dp)
                    k = metric(template_cut,Iw)
                    if(not max_ or k>max_): 
                        best_p = dp
                        max_ = k 
                    # print('good')
                except IndexError:
                    # print('bad')

                    continue
            homo.select(best_p)
            box_t = homo.get_box(box)
            # print(box_t,best_p)
            output.append(box_t)

            ''' Score '''
            sIOU += IOU(box_t,box)
            n+=1.
            
            frame = cv.rectangle(frame,(box_t[1],box_t[0]) ,(box_t[1]+box_t[3],box_t[2]+box_t[0]) , (255,0,0), 5)
            frame = cv.rectangle(frame,(gt[frame_no-2,1],gt[frame_no-2,0]) ,(gt[frame_no-2,1]+gt[frame_no-2,3],gt[frame_no-2,2]+gt[frame_no-2,0]) , (0,255,0), 2)
            cv.imshow('Image',frame)
            if cv.waitKey(60) & 0xFF == ord('q'):
                break
            

    
    np.savetxt(outfile, np.array(output), delimiter=",",fmt = "%d")
    print('mIOU score: '+ str(sIOU/n*100))


'''Needs lots of work'''
class LK:
    def __init__(self,template,box ,p = np.eye(3),tol = 0.1):
        self.geometry = Projective(p)
        self.template_img = template
        self.box = box
        self.template =  template[box[1]:box[1]+box[3]+1,box[0]:box[0]+box[2]+1]
        # print(self.template.shape,box)
        self.p = p
        self.tol = tol

    
    def fit(self,img):
        self.img = img
        Iw = self.geometry.transform(img,self.box)
        t1 = np.matmul(self.del_I(Iw),self.del_W()) 
        H  = np.einsum('ijkl,ijkm->lm',t1,t1)
        dp = np.linalg.inv(H).dot(np.einsum('ijkl,ij->lk',t1,self.template-Iw))
        dp.resize((3,3),refcheck=False)
        self.p=self.p + dp
        self.geometry.select(dp)
        while(np.linalg.norm(dp)>self.tol):
            # print(np.linalg.norm(dp))
            Iw = self.geometry.transform(img,self.box)
            t1 = np.matmul(self.del_I(Iw),self.del_W()) 
            H  = np.einsum('ijkl,ijkm->lm',t1,t1)
            
            dp = np.linalg.inv(H).dot(np.einsum('ijkl,ij->lk',t1,self.template-Iw))
            # except ValueError:
            #     print(np.linalg.inv(H))
            #     print(np.einsum('ijkl,ij->lk',t1,self.template-Iw))
            # print(self.p)
            dp.resize((3,3),refcheck=False)
            self.p=self.p + dp
            self.geometry.select(dp)
        print('-'*20)
        return self.geometry.get_box(self.box)
    def del_I(self,Iw):
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
        xx,yy = np.meshgrid(x,y)
        
        return self.Wp_affine(xx,yy)
    def Wp_affine(self,x,y):
        shape = x.shape
        k = np.stack((np.stack((x,np.zeros(shape),y,np.zeros(shape),np.ones(shape),np.zeros(shape))),
                    np.stack((np.zeros(shape),x,np.zeros(shape),y,np.zeros(shape),np.ones(shape)))))
        # print(k.shape,x.shape,y.shape)
        return k.transpose((2,3,0,1))#2 x p x h x w
    
    # def Wp_projective(self,x,y):
    #     xp = p.dot(np.array([x,y,1]))
    #     xp,yp,c = xp
    #     xp,yp = xp/c,yp/c 
    #     return (np.array([[x,y,1,0,0,0,-x*xp,-y*xp],[0,0,0,x,y,1,-x*yp,-y*yp]])/c).transpose((2,3,0,1))

def lk_tracker(dir,outfile):
    inp_path = os.path.join(dir,'img')
    gt = np.genfromtxt(os.path.join(dir,'groundtruth_rect.txt'), delimiter=',')
    gt = np.int32(gt)
    # box = gt[0]
    box = [227,207,122,99]
    gt = gt[1:]
    files = sorted(os.listdir(inp_path))
    files = files[1:]
    template = cv.imread(os.path.join(inp_path,files[0]))
    template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)
    # print(template.shape)
    # return
    output = []
    sIOU,n = 0.,0.
    tracker = LK(template,box)
    for file in files:
        if file[-4:] in ['.jpg']:
            frame = cv.imread(os.path.join(inp_path,file))
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            if frame is None: break             
            frame_no = int(re.search(r'[0-9]+',file)[0])



            box_t = tracker.fit(frame)
            # print(len(box_t))
            output.append(box_t)

            ''' Score '''
            sIOU += IOU(box_t,box)
            print(IOU(box_t,box))
            n+=1.
            
            frame = cv.rectangle(frame,(box_t[1],box_t[0]) ,(box_t[1]+box_t[3],box_t[2]+box_t[0]) , (255,0,0), 2)
            frame = cv.rectangle(frame,(gt[frame_no-2,1],gt[frame_no-2,0]) ,(gt[frame_no-2,1]+gt[frame_no-2,3],gt[frame_no-2,2]+gt[frame_no-2,0]) , (0,255,0), 2)
            cv.imshow('Image',frame)
            if cv.waitKey(60) & 0xFF == ord('q'):
                break
            

    
    np.savetxt(outfile, np.array(output), delimiter=",",fmt='%d')
    print('mIOU score: '+ str(sIOU/n*100))



delp = [0,0,60,0,0,60]
n_p = [1,1,3,1,1,3]
# block_based('./A2/BlurCar2',np.eye(3),delp,n_p,'./A2/BlurCar2/outfile')
# block_based('.\A2\data\BlurCar2',np.eye(3),delp,n_p,'.\A2\data\BlurCar2\outfile')


lk_tracker('.\A2\data\BlurCar2','.\A2\data\BlurCar2\outfile')
# def LK():
    # pass
