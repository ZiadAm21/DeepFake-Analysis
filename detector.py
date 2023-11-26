import numpy as np
import os
import cv2
from imageio import imread
from skimage.transform import resize
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import Model,Sequential,load_model
import pandas as pd
import h5py
import glob
import sys
import scipy.io
import time 
import matplotlib.pyplot as plt 

rootdir = 'YourPath/DeepFakeDetective/'
image_name_video = []
# Load the cascade
face_cascade = cv2.CascadeClassifier(os.path.join(
    rootdir, 'haarcascade_frontalface_default.xml'))

for f in [f for f in os.listdir(os.path.join(rootdir, 'TestData/'))]:
    if "DS_Store" in f: 
        continue
    
    carpeta= os.path.join(rootdir, 'TestData/', f)
    cap = cv2.VideoCapture(carpeta)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    nFrames = cap.get(7)
    max_frames = int(nFrames)
    ruta_parcial = os.path.join(rootdir, 'DeepFrames',f)
    if not(os.path.exists(ruta_parcial)):
        os.mkdir(ruta_parcial);
    ruta_parcial2 = os.path.join(rootdir, 'RawFrames',f)
    if not(os.path.exists(ruta_parcial2)):
        os.mkdir(ruta_parcial2);
    
    L = 36
    C_R=np.empty((L,L,max_frames))
    C_G=np.empty((L,L,max_frames))
    C_B=np.empty((L,L,max_frames))
    
    D_R=np.empty((L,L,max_frames))
    D_G=np.empty((L,L,max_frames))
    D_B=np.empty((L,L,max_frames))
    
    D_R2=np.empty((L,L,max_frames))
    D_G2=np.empty((L,L,max_frames))
    D_B2=np.empty((L,L,max_frames))
    
    medias_R = np.empty((L,L))
    medias_G = np.empty((L,L))
    medias_B = np.empty((L,L))
    
    desviaciones_R = np.empty((L,L))
    desviaciones_G = np.empty((L,L))
    desviaciones_B = np.empty((L,L))
    
    imagen = np.empty((L,L,3))
    
    medias_CR = np.empty((L,L))
    medias_CG = np.empty((L,L))
    medias_CB = np.empty((L,L))
    
    desviaciones_CR = np.empty((L,L))
    desviaciones_CG = np.empty((L,L))
    desviaciones_CB = np.empty((L,L))
    ka            = 1
    
    
    while(cap.isOpened() and ka< max_frames):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        #rectangle around the faces
        for (x, y, w, h) in faces:
            # face = cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face = frame[y:y + h, x:x + w]
            
       
        face = cv2.resize(face, (L,L), interpolation = cv2.INTER_AREA)
        # cv2.imshow('img', face)
        # cv2.waitKey()
        C_R[:,:,ka] = face[:,:,0]
        C_G[:,:,ka] = face[:,:,1]
        C_B[:,:,ka] = face[:,:,2]
        
        
        if ka > 1:
            D_R[:,:,ka-1] = ( C_R[:,:,ka] - C_R[:,:,ka-1] ) / ( C_R[:,:,ka] + C_R[:,:,ka-1] );
            D_G[:,:,ka-1] = ( C_G[:,:,ka] - C_G[:,:,ka-1] ) / ( C_G[:,:,ka] + C_G[:,:,ka-1] );
            D_B[:,:,ka-1] = ( C_B[:,:,ka] - C_B[:,:,ka-1] ) / ( C_B[:,:,ka] + C_B[:,:,ka-1] );
        ka = ka+1
     
    
        
    for i in range(0,L):
        for j in range(0,L):
            medias_R[i,j]=np.mean(D_R[i,j,:]) 
            medias_G[i,j]=np.mean(D_G[i,j,:]) 
            medias_B[i,j]=np.mean(D_B[i,j,:]) 
            desviaciones_R[i,j]=np.std(D_R[i,j,:]) 
            desviaciones_G[i,j]=np.std(D_G[i,j,:]) 
            desviaciones_B[i,j]=np.std(D_B[i,j,:]) 
            
    for i in range(0,L):
        for j in range(0,L):
            medias_CR[i,j]=np.mean(C_R[i,j,:]) 
            medias_CG[i,j]=np.mean(C_G[i,j,:]) 
            medias_CB[i,j]=np.mean(C_B[i,j,:]) 
            desviaciones_CR[i,j]=np.std(C_R[i,j,:]) 
            desviaciones_CG[i,j]=np.std(C_G[i,j,:]) 
            desviaciones_CB[i,j]=np.std(C_B[i,j,:])         
            
    for k in range(0,max_frames):
        D_R2[:,:,k] = (C_R[:,:,k] - medias_CR)/(desviaciones_CR+000.1)
        D_G2[:,:,k] = (C_G[:,:,k] - medias_CG)/(desviaciones_CG+000.1)
        D_B2[:,:,k] = (C_B[:,:,k] - medias_CB)/(desviaciones_CB+000.1)
     


    for k in range(0,max_frames):
        
        imagen[:,:,0] = D_R2[:,:,k]
        imagen[:,:,1] = D_G2[:,:,k]
        imagen[:,:,2] = D_B2[:,:,k]

        imagen= np.uint8(imagen)
        
        nombre_salvar= os.path.join(ruta_parcial2,str(k)+'.png')
        cv2.imwrite(nombre_salvar, imagen)
        

    for k in range(0,max_frames):
        
        D_R[:,:,k] = (D_R[:,:,k] - medias_R)/(desviaciones_R+000.1)
        D_G[:,:,k] = (D_G[:,:,k] - medias_G)/(desviaciones_G+000.1)
        D_B[:,:,k] = (D_B[:,:,k] - medias_B)/(desviaciones_B+000.1)
        
    for k in range(0,max_frames):
        
        imagen[:,:,0] = D_R[:,:,k]
        imagen[:,:,1] = D_G[:,:,k]
        imagen[:,:,2] = D_B[:,:,k]
        
        imagen= np.uint8(imagen)

        nombre_salvar= os.path.join(ruta_parcial,str(k)+'.png')
        cv2.imwrite(nombre_salvar, imagen)            
        
        
    cap.release()
    cv2.destroyAllWindows()
print("Exiting...")

def load_test_motion(carpeta):
    X_test = []
    images_names = []
    image_path = carpeta
    print(carpeta)
    print('Read test images')
    for f in [f for f in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, f))]:
        carpeta= os.path.join(image_path, f)
        print(carpeta)
        for imagen in [imagen for imagen in os.listdir(carpeta) if os.path.isfile(os.path.join(carpeta, imagen))]:
            imagenes = os.path.join(carpeta, imagen)
            img = cv2.resize(cv2.imread(imagenes, cv2.IMREAD_COLOR), (36, 36))
            img = img.transpose((-1,0,1))
            X_test.append(img)
            images_names.append(imagenes)
    return X_test, images_names

def load_test_attention(carpeta):
    X_test = []
    images_names = []
    image_path = carpeta
    print('Read test images')
    for f in [f for f in os.listdir(image_path) if os.path.isdir(os.path.join(image_path, f))]:
        carpeta= os.path.join(image_path, f)
        print(carpeta)
        for imagen in [imagen for imagen in os.listdir(carpeta) if os.path.isfile(os.path.join(carpeta, imagen))]:
            imagenes = os.path.join(carpeta, imagen)
            img = cv2.resize(cv2.imread(imagenes, cv2.IMREAD_COLOR), (36, 36))
            img = img.transpose((-1,0,1))
            X_test.append(img)
            images_names.append(imagenes)
    return X_test, images_names

np.set_printoptions(threshold=np.inf)
data = []
batch_size = 128
model = load_model(os.path.join(rootdir, 'DeepFakesON-Phys_CelebDF_V2.h5'))
print(model.summary())
input("Press Enter to continue...")

carpeta_deep= os.path.join(rootdir, "DeepFrames")
carpeta_raw= os.path.join(rootdir, "RawFrames")

test_data, images_names = load_test_motion(carpeta_deep)
test_data2, images_names = load_test_attention(carpeta_raw)

test_data = np.array(test_data, copy=False, dtype=np.float32)
test_data2 = np.array(test_data2, copy=False, dtype=np.float32)

#Note: model.predict only works on GPU
predictions = model.predict([test_data, test_data2], batch_size=batch_size, verbose=1)

bufsize = 1
nombre_fichero_scores = 'deepfake_scores.txt'
fichero_scores = open(nombre_fichero_scores,'w',buffering=bufsize)
fichero_scores.write("img;score\n")
for i in range(predictions.shape[0]):
    fichero_scores.write("%s" % images_names[i]) #fichero
    fichero_scores.write(";%s\n" % predictions[i]) #scores predichas
