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
    def __init__(self,p = np.eye(3)):
        self.H = p 
        # self.H = p 
    def transform(self,p,img,box):
        x,y,w,h = box
        # y,x,h,w = box
        # p = self.H 
        b = bilinear_interpolate(img)
        # dp =dp.copy()
        # dp.resize((3,3),refcheck=False)
        print(box)
        def func(x_,y_):
            res=p@np.array([x_,y_,1])
            # print(x_,y_)
            xp,yp,_ = res/res[2]
            # print(x_,y_)
            # print()
            return b(xp,yp)
        
        Iw = np.zeros((h+1,w+1))
        for j in range(h+1):
            for i in range(w+1):
                Iw[j,i] = func(x+i,y+j)
            # print(i)
        # print(x,y)
        return Iw
    def get_box(self,box):
        x,y,w,h = box
        coords = (self.H @ np.array([[x,y,1],[x+w,y,1],[x+w,y+h,1],[x,y+h,1]]).T).T
        # print(coords)
        # print(self.H)
        coords = coords/coords[:,-1:]
        # print(np.min(coords,axis=0))
        x_,y_,_ = np.min(coords,axis=0)       
        w_,h_,_ = np.max(coords,axis=0)      
        ans = [math.floor(x_),math.floor(y_),math.ceil(w_-x_+1),math.ceil(h_-y_+1)]
        # print(box)
        # print(ans)
        # print()
        return ans 
    def select(self,dp):
        # print(dp)
        # print(dp)
        dp =dp.copy()
        dp.resize((3,3),refcheck=False)
        # print(dp)
        self.H = dp


def bilinear_interpolate(img):
    def image(x,y):

        if y>=img.shape[0]-1 or y<0 or x>=img.shape[1]-1 or x<0:
            # print(x,y)
            raise IndexError
            # return 0
        i,j = math.floor(x),math.floor(y)
        a,b = x-i,y-j
        return (1-a)*((1-b)*img[j,i]+b*img[j+1,i]) + a*((1-b)*img[j,i+1]+b*img[j+1,i+1])
    return image

def NSSE(t,f):
    # print(1-np.linalg.norm(f-t)**2/f.size/255.**2)
    return 1-np.linalg.norm(f-t)**2/f.size/255.**2
def NCC(t,f):
    # print(np.sum(f*t)/np.linalg.norm(f**2)/np.linalg.norm(t**2))#/f.size)
    # print(np.sum(f*t),(f*t).shape,np.std(f).shape,np.std(t),f.size)
    return np.sum(f*t)/np.linalg.norm(f**2)/np.linalg.norm(t**2)#/f.size

def IOU(box1,box2):
    boxA,boxB = box1.copy(),box2.copy()
    # print(boxA,boxB)
    # boxA[2:]+=boxA[:2]
    # boxB[2:]+=boxB[:2]
    xA = max(boxA[0],boxB[0])
    yA = max(boxA[1],boxB[1])
    xB = min(boxA[2]+boxA[0],boxB[2]+boxB[0])
    yB = min(boxA[3]+boxA[1],boxB[3]+boxB[1])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2]  + 1) * (boxA[3] + 1)
    boxBArea = (boxB[2]  + 1) * (boxB[3] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # print(iou,boxA,boxB)
    return iou



def block_based(dir,p_0,delp,n_p,outfile,metric = NSSE,interactive = []):
    sample = [np.linspace(-dp,dp,n) for dp,n in zip(delp,n_p)]
    sample = np.stack(np.meshgrid(*sample)).reshape((-1,len(delp)))

    # print(sample.shape,len(sample))
    # return
    inp_path = os.path.join(dir,'img')
    gt = np.genfromtxt(os.path.join(dir,'groundtruth_rect.txt'), delimiter=',')
    gt = np.int32(gt)
    
    # gt = gt[1:]
    files = sorted(os.listdir(inp_path))
    files = files[1:]
    template = cv.imread(os.path.join(inp_path,files[0]))
    template = cv.cvtColor(template,cv.COLOR_BGR2GRAY)
    if interactive:
        box = interactive
    else:
        box = gt[0]
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
            mat = cv.matchTemplate(frame,template_cut,cv.TM_CCORR_NORMED)

            x,y = cv.minMaxLoc(mat)[3]
            box_t = [x,y,box[2],box[3]]
            # print(box_t,best_p)
            output.append(box_t)

            ''' Score '''
            sIOU += IOU(box_t,gt[frame_no-1])
            
            n+=1.
            
            # frame = cv.rectangle(frame,(box_t[1],box_t[0]) ,(box_t[1]+box_t[3],box_t[2]+box_t[0]) , (255,0,0), 5)
            # frame = cv.rectangle(frame,(gt[frame_no-2,1],gt[frame_no-2,0]) ,(gt[frame_no-2,1]+gt[frame_no-2,3],gt[frame_no-2,2]+gt[frame_no-2,0]) , (0,255,0), 2)
            frame = cv.rectangle(frame,(box_t[0],box_t[1]) ,(box_t[0]+box_t[2],box_t[3]+box_t[1]) , (255,0,0), 2)
            if not interactive:
                frame = cv.rectangle(frame,(gt[frame_no-1,0],gt[frame_no-1,1]) ,(gt[frame_no-1,0]+gt[frame_no-1,2],gt[frame_no-1,3]+gt[frame_no-1,1]) , (0,255,0), 2)

            cv.imshow('Image',frame)
            if cv.waitKey(60) & 0xFF == ord('q'):
                break
            

    
    np.savetxt(outfile, np.array(output), delimiter=",",fmt = "%d")
    if not interactive:
        print('mIOU score: '+ str(sIOU/n*100))


'''Needs lots of work'''
class LK:
    def __init__(self, template, box, p = np.eye(3), tol = 0.84):
        self.template_img = template
        self.box = box
        self.template =  template[box[1]:box[1]+box[3]+1,box[0]:box[0]+box[2]+1]
        cv.imshow("Template",self.template)
        # print(self.template.shape,box)
        self.p = p
        self.tol = tol

    
    def fit(self,img):
        self.img = img
        self.geometry = Projective(np.eye(3))
        err = 3000
        self.p = np.eye(3)
        # self.p = np.eye(3)
        # Iw = np.ones(self.template.shape)
        # Iw[0,0] = 0
        # dp = np.zeros(6)
        # Iw = self.geometry.transform(img,self.box)
        # t1 = np.matmul(self.del_I(Iw),self.del_W()) 
        # H  = np.einsum('ijkl,ijkm->lm',t1,t1)
        # term = np.einsum('ijkl,ij->lk',t1,self.template-Iw)
        
        # dp = np.linalg.inv(H).dot(term)
        # dp.resize((3,3),refcheck=False)
        # # print(dp)
        # self.p=self.p + dp
        # self.geometry.select(dp)
        # NCC(self.template,Iw)<self.tol and 
        while(err>0.7):
            
            # print(np.linalg.norm(dp))
            try:
                Iw = self.geometry.transform(self.p,img,self.box)
            # print(img.type())
            # rotated_image = cv.warpAffine(src=img, M=self.p, dsize=img.shape)
            # box = self.box
            # Iw = rotated_image[box[1]:box[1]+box[3]+1,box[0]:box[0]+box[2]+1]
            # cv.imshow('Iw',Iw/255)
            # cv.waitKey(0)

                t1 = np.matmul(self.del_I(img),self.del_W())
            # print(self.p)

            # print(Iw.shape,self.del_W().shape)
                '''
                H  = np.einsum('ijkl,ijkm->lm',t1,t1)
                
                dp = np.linalg.inv(H).dot(np.einsum('ijkl,ij->lk',t1,self.template-Iw))
                '''
            # print(t1.shape,np.linalg.pinv(t1).shape,(self.template-Iw).shape)
                dp = np.einsum('ijkl,ij->lk',np.linalg.pinv(t1),self.template-Iw)
            
            # exit()
            # except ValueError:
            #     print(np.linalg.inv(H))
            #     print(np.einsum('ijkl,ij->lk',t1,self.template-Iw))
                print(dp)
                dp.resize((3,3),refcheck=False)
                self.p=self.p + dp
                print(self.p)
                err = np.linalg.norm(dp)
                print(err)

            except IndexError:
                err = 0
                
                
            self.geometry.select(self.p)
            
        print('-'*20)
        print(self.box)
        return self.geometry.get_box(self.box)
    def del_I(self,Iw):
        '''h x w x 1 x 2'''
        dx = np.array([[-1,0,1],
                    [-2,0,2],
                    [-1,0,1]])
    
        # I = np.array([[]])
        # return I.transpose((2,3,0,1))
        Ix,Iy = cv.filter2D(Iw, -1, dx, borderType=cv.BORDER_CONSTANT), cv.filter2D(Iw, -1, dx.T, borderType=cv.BORDER_CONSTANT)
        # print(Ix.shape,Iy.shape,Iw.shape)
        # cv.imshow('Ix',Ix)
        # cv.waitKey(0)
        Ix = self.geometry.transform(self.p,Ix,self.box)
        Iy = self.geometry.transform(self.p,Iy,self.box)
        I = np.array([[Ix,Iy]])
        # print(I.shape)
        # exit()

        return I.transpose((2,3,0,1))

    def del_W(self):
        x = np.arange(self.box[0],self.box[0]+self.box[2]+1)
        y = np.arange(self.box[1],self.box[1]+self.box[3]+1)
        xx,yy = np.meshgrid(x,y)
        
        # return self.Wp_translation(xx,yy)
        return self.Wp_affine(xx,yy)



    def Wp_translation(self,x,y):
        shape = x.shape
        # print(x.shape)
        # print(np.stack((np.ones(shape),np.zeros(shape))).shape)
        # k = np.stack((np.stack((x,np.zeros(shape),y,np.zeros(shape),np.ones(shape),np.zeros(shape))),
                    # np.stack((np.zeros(shape),x,np.zeros(shape),y,np.zeros(shape),np.ones(shape)))))
        k = np.stack((np.stack((np.ones(shape),np.zeros(shape))),
                    np.stack((np.zeros(shape),np.ones(shape)))))
        # k = np.eye(3)
                    # np.stack((np.zeros(shape),x,n)
        # print(k.shape,x.shape,y.shape)
        return k.transpose((2,3,0,1))
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
    box = gt[0]
    # box = [227,207,122,99]
    # gt = gt[1:]
    print(gt.shape)
    files = sorted(os.listdir(inp_path))
    template = cv.imread(os.path.join(inp_path,files[0]))
    # files = files[1:]
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
            print(box_t)
            output.append(box_t)

            ''' Score '''

            sIOU += IOU(box_t,gt[frame_no-1])
            n+=1.
            
            # frame = cv.rectangle(frame,(box_t[1],box_t[0]) ,(box_t[1]+box_t[3],box_t[2]+box_t[0]) , (255,0,0), 2)
            frame = cv.rectangle(frame,(box_t[0],box_t[1]) ,(box_t[0]+box_t[2],box_t[3]+box_t[1]) , (255,0,0), 2)
            # frame = cv.rectangle(frame,(gt[frame_no-1,1],gt[frame_no-1,0]) ,(gt[frame_no-1,1]+gt[frame_no-1,3],gt[frame_no-1,2]+gt[frame_no-1,0]) , (0,255,0), 2)
            frame = cv.rectangle(frame,(gt[frame_no-1,0],gt[frame_no-1,1]) ,(gt[frame_no-1,0]+gt[frame_no-1,2],gt[frame_no-1,3]+gt[frame_no-1,1]) , (0,255,0), 2)
            cv.imshow('Image',frame)
            # cv.waitKey(0)
            if cv.waitKey(60) & 0xFF == ord('q'):
                break
            

    
    np.savetxt(outfile, np.array(output), delimiter=",",fmt='%d')
    print('mIOU score: {0:.4f}'.format(sIOU/n))



delp = [1,1,60,1,1,60]
n_p = [10,10,3,10,10,3]
# block_based('./A2/BlurCar2',np.eye(3),delp,n_p,'./A2/BlurCar2/outfile')
block_based('.\A2\data\BlurCar2',np.eye(3),delp,n_p,'.\A2\data\BlurCar2\outfile')#,interactive = [0,0,100,100])


