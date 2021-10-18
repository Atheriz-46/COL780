import cv2 as cv
import pandas as pd 
import numpy as np 
import math


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
        dp.resize((3,3))
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
        self.H += dp.resize((3,3))


def bilinear_interpolate(img):
    def image(x,y):
        if y>=img.shape[1]-1 or y<0 or x>=img.shape[0]-1 or x<0:
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
    pass

def block_based(dir,p_0,delp,n_p,outfile,metric = NSSE):
    sample = [np.linspace(-dp,dp,n) for dp,n in zip(delp,n_p)]
    sample = np.meshgrid(*sample)
    imp_path = os.path.join(dir,'img')
    gt = genfromtxt(os.path.join(dir,'groundtruth_rect.txt'), delimiter=',')
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
            max_,bset_p=None, None
            for dp in sample:
                Iw = homo.transform(frame,box,dp)
                k = metric(template_cut,Iw)
                if(k>max_ or not max_): 
                    best_p = dp
                    max_ = k 
            homo.select(dp)
            box_t = homo.get_box(box)
            output.append(box_t)

            ''' Score '''
            sIOU += IOU(box_t,box)
            n+=1.
            '''
            frame = cv.rectangle(frame,(box_t[1],box_t[0]) ,(box_t[1]+box_t[3],box_t[2]+box_t[0]) , (255,0,0), 2)
            frame = cv.rectangle(frame,(gt[frame_no-2,1],gt[frame_no-2,0]) ,(gt[frame_no-2,1]+gt[frame_no-2,3],gt[frame_no-2,2]+gt[frame_no-2,0]) , (0,255,0), 2)
            cv.imshow('Image',frame)
            if cv.waitKey(60) & 0xFF == ord('q'):
                break
            '''

    
    numpy.savetxt(outfile, np.array(output), delimiter=",")
    print('mIOU score: %.6f%', sIOU/n*100)


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
            self.p+=dp
        return self.geometry.get_box(self.box)
    def del_I(self):
        pass
    def del_W(self):
        pass

# def LK():
    # pass