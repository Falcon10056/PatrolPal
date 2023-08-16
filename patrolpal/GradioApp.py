#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


# !pip install deepface
# !pip install retina-face
# !pip install ray
# !pip install moviepy
# !pip install imgaug
# !pip install h5py
# !pip install pycocotools
# !pip install cython
# !pip install tensorflow-gpu
# !pip install opencv-python
# !pip install scikit-image
# !pip install scikit-learn
# !pip install pandas
# !pip install matplotlib
# !pip install scipy
# !pip install numpy
# !pip install gradio
# !pip install imageio_ffmpeg


# In[1]:


import cv2
import os
import shutil
from pathlib import Path
import keras
from deepface import DeepFace
from retinaface import RetinaFace
import cv2
# from google.colab import files
from zipfile import ZipFile
import os
import shutil
from datetime import date
# from google.colab.patches import cv2_imshow
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import smtplib
import gradio as gr
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import time as tm
import ray
import numpy as np
import glob
from PIL import Image
ray.shutdown()
ray.init()


# In[2]:


# # from google.colab import drive
# # drive.mount('/content/drive',force_remount=True)
# !pwd


# In[19]:


@ray.remote
def func1(frame_name,filename,alpha,beta,output_storage_path,queryPath,detected_op_path,detected_freq,flag):
  nd=[]
  print(filename,"FILENAME")
  frames = []

  cap = cv2.VideoCapture(filename)
  print(cap)
  dflag=0
  while True:
    text_file = open(output_storage_path + "/person_coordinates.txt", 'a')
    ret, frame = cap.read()
    #print("hello")
    if(not ret):
      print("Please check your video quality or Your video file maybe corrupted!")
      break
    if(ret):

      if(frame_name%25==0):
        frame=cv2.addWeighted(frame,alpha,np.zeros(frame.shape,frame.dtype),0,beta)
        if(flag==1):
          cv2.imwrite(output_storage_path+filename.split("/")[-1][:-6]+".mp4"+"/"+str(frame_name)+".jpg", frame)
          print(output_storage_path+filename.split("/")[-1][:-6]+".mp4"+"/"+str(frame_name)+".jpg")
        else:
          cv2.imwrite(output_storage_path+filename.split("/")[-1]+"/"+str(frame_name)+".jpg", frame)
          print(output_storage_path+filename.split("/")[-1]+"/"+str(frame_name)+".jpg")
        try:
          if(flag==1):
            obj = DeepFace.verify(output_storage_path+filename.split("/")[-1][:-6]+".mp4"+"/"+str(frame_name)+".jpg", queryPath, model_name = 'ArcFace', detector_backend = 'retinaface')
          else:
            obj = DeepFace.verify(output_storage_path+filename.split("/")[-1]+"/"+str(frame_name)+".jpg", queryPath, model_name = 'ArcFace', detector_backend = 'retinaface')

          if(obj["verified"]==True or obj['distance']<.8):            
            cv2.imwrite(detected_op_path+str(frame_name)+".jpg",frame)
            print("Person of Interest detected in Frame No.",frame_name)
            detected_freq+=1
            text_file.write(str(frame_name)+"\n")
            frames.append(frame_name)

        except:

          print("no face")
          nd.append(frame_name)
        
        if(detected_freq>=2):
          break
    frame_name+=1
    text_file.close()
  return nd, frame_name


# In[20]:


def patrolPal(file_obj, query_obj):
    # uploaded = files.upload()
    #print(uploaded)
    # filename = next(iter(file_obj))
    filename = file_obj.name
    print(filename)
    print(file_obj.name, query_obj.name)
    list_of_cctv_vids=[]

    if not os.path.isdir('content/drive/MyDrive/person-modeltest/person-identify/'):
        os.makedirs('content/drive/MyDrive/person-modeltest/person-identify/Detected Frames/')
        os.makedirs('content/drive/MyDrive/person-modeltest/person-identify/output/')
        
    shutil.unpack_archive(filename, "content/drive/MyDrive"+filename)
    for filepath in os.listdir("content/drive/MyDrive"+filename):
        f = os.path.join(filename, filepath)
        # checking if it is a file
        print(f)
        list_of_cctv_vids.append("content/drive/MyDrive"+f)

    #print("Files inside the uploaded Zip Folder : ")
    print(list_of_cctv_vids)

    #print(uploaded)
    queryFileName = query_obj.name
    queryPath=queryFileName

    print(queryPath,"is the query image uploaded")

    #Redundant frame removal

    #list_of_cctv_vids=['/content/drive/MyDrive/case 5 files.zip/av-roundabout_pz3dg56a_new.mp4', '/content/drive/MyDrive/case 5 files.zip/cctv-1_new.mp4']

    # temp_list=[]
    # for i in list_of_cctv_vids:
    #     j=i[:-4]+'_new.'+i[-3:]
    #     os.popen("ffmpeg -i {input} -vf mpdecimate=hi=8000:lo=8000:frac=1:max=500 {output}".format(input = i, output = j))
    #     temp_list.append(j)
    
    # tm.sleep(120)

    # list_of_cctv_vids=temp_list
    print(list_of_cctv_vids)

    # for i in list_of_cctv_vids:
    #     vid2=i
    #     cam2 = cv2.VideoCapture(vid2)
    #     video_length = int(cam2.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    #     print ("Number of frames in modified video: ", video_length)

        
    #Folder hierarchy module
    today=date.today()
    directory = "Detected Frames/"+str(today)
    parent_dir = "content/drive/MyDrive/person-modeltest/person-identify/"
    path = os.path.join(parent_dir, directory)
    CHECK_FOLDER = os.path.isdir(path)
    if not CHECK_FOLDER:
        os.mkdir(path)
        print("Date Directory created")
    else:
        print("Directory already exists")

    
    for i in list_of_cctv_vids:
      temp=i.split("/")[-1]
      newfolder=os.path.join(path+"/", temp)
      CHECK_FOLDER = os.path.isdir(newfolder)
      if not CHECK_FOLDER:
        os.mkdir(newfolder)
        print(temp," Directory created")
      else:
        print("Directory already exists")

    path="content/drive/MyDrive/person-modeltest/person-identify/output/"
    for i in list_of_cctv_vids:
        temp=i.split("/")[-1]
        newfolder=os.path.join(path+"/", temp)
        CHECK_FOLDER = os.path.isdir(newfolder)
        if not CHECK_FOLDER:
            os.mkdir(newfolder)
            print("New Output Directory created")
        else:
            print("Output Directory already exists")

    #creating a directory to save only the frames where the person is detected

    path="content/drive/MyDrive/person-modeltest/person-identify/Detected Frames/"+str(today)
    for i in list_of_cctv_vids:
        temp=i.split("/")[-1]+"/"+"Face Recognition"
        newfolder=os.path.join(path+"/", temp)
        CHECK_FOLDER = os.path.isdir(newfolder)
        if not CHECK_FOLDER:
            os.mkdir(newfolder)
            print("New Face Recognition Directory created")
        else:
            print("Face Recognition Directory already exists")
        temp=i.split("/")[-1]+"/"+"Person Identification"
        newfolder=os.path.join(path+"/", temp)
        CHECK_FOLDER = os.path.isdir(newfolder)
        if not CHECK_FOLDER:
            os.mkdir(newfolder)
            print("New Person Identification Directory created")
        else:
            print("Person Identification Directory already exists")

    #creating textfiles to save the coordinate positions of the detected person
    path="content/drive/MyDrive/person-modeltest/person-identify/output/"
    for i in list_of_cctv_vids:
        temp=i.split("/")[-1]+"/"+"person_coordinates.txt"
        my_file = open(os.path.join(path, temp),"w+")
        my_file.close()
    
    #to maintain a list containing frames not detected by facial recognition module
    not_detected=[[]]*len(list_of_cctv_vids)
    #setting parameters for video frames enhancement
    alpha = 1.33  
    beta = -50
    for v in range(len(list_of_cctv_vids)):
        print("Analyzing video ",list_of_cctv_vids[v].split("/")[-1])
        detected_freq=0
        #list to contain information about the split up seconds of video into two halves
        times=[]

        output_storage_path="content/drive/MyDrive/person-modeltest/person-identify/output/"
        detected_op_path="content/drive/MyDrive/person-modeltest/person-identify/Detected Frames/"+str(date.today())+"/"+list_of_cctv_vids[v].split("/")[-1]+"/Face Recognition/"
        #cap = cv2.VideoCapture(input_video_path)

        #loading the video
        data = cv2.VideoCapture(list_of_cctv_vids[v])

        # count the number of frames in th evideo
        frames = data.get(cv2.CAP_PROP_FRAME_COUNT)
        #total no.of frames in the video
        fps = int(data.get(cv2.CAP_PROP_FPS))
        print("fps", fps)

        # calculate dusration of the video
        cutoff_len=0
        #total seconds of video
        seconds = int(frames / fps)

        #list to contain the two new split video's address
        img_pths=[]
        if(seconds>20):
            #mid point of the video in seconds
            first_half=seconds//2
            #storing 0-mid an mid-end info in times list
            times.append("0-"+str(first_half))
            times.append(str(first_half)+"-"+str(seconds))
            
            for time in times:
                #starting time of each subclip
                starttime = int(time.split("-")[0])
                #ending time of subclip
                endtime = int(time.split("-")[1])
                #no. of seconds in subclip
                cutoff=(endtime-starttime)*fps
                if(times.index(time)==0):
                    cutoff_len=cutoff
                #creating the subclip andsaving to the videos folder
                ffmpeg_extract_subclip(list_of_cctv_vids[v], starttime, endtime, targetname=list_of_cctv_vids[v][:-4]+"#"+str(times.index(time)+1)+".mp4")
                print("Path of the subclip no.",str(times.index(time)+1)+" : "+list_of_cctv_vids[v][:-4]+"#"+str(times.index(time)+1)+".mp4")
                img_pths.append(list_of_cctv_vids[v][:-4]+"#"+str(times.index(time)+1)+".mp4")
            tm.sleep(120)
            
            # running the analysis module parallely forboth subclips
            impl1=func1.remote(0,img_pths[0],1.33,-50,output_storage_path,queryPath,detected_op_path,0,1)
            impl2=func1.remote(cutoff_len,img_pths[1],1.33,-50,output_storage_path,queryPath,detected_op_path,0,1)
            
            x,y=ray.get([impl1,impl2])
            print(x,y)
            #for i in range(len(y)):
            #w[i]=cutoff_len+w[i]
            #print(x+y,"gives undetected")
            x=x+y
            not_detected[v]=x
        else:
            img_pths.append(list_of_cctv_vids[v])
            impl1=func1.remote(0,img_pths[0],1.33,-50,output_storage_path,queryPath,detected_op_path,0,0)
            x=ray.get([impl1])
            print(x)
            not_detected[v].extend(x)
        #print(x,"gives undetected")
        print("Frames in each video which are not detected :",not_detected)
        for i in range(len(list_of_cctv_vids)):
            if not_detected[i]==[[]]:
                not_detected[i]=[]
        
        print(output_storage_path, list_of_cctv_vids)

        op = []
        valid_images = [".jpg",".gif",".png",".tga"]
        
        for i in list_of_cctv_vids:
          for f in os.listdir("content/drive/MyDrive/person-modeltest/person-identify/Detected Frames/"+str(today)+"/"+i.split("/")[-1]+"/Face Recognition/"):
            ext = os.path.splitext(f)[1]
            if ext.lower() not in valid_images:
              op.append(Image.open("content/drive/MyDrive/imageNotFound.jpeg"))
              continue
            else:
              op.append(Image.open(os.path.join("content/drive/MyDrive/person-modeltest/person-identify/Detected Frames/"+str(today)+"/"+i.split("/")[-1]+"/Face Recognition/",f)))
              break
        while(len(op)<5):
          op.append(Image.open("imageNotFound.jpeg"))

        print(op)

        opText = []
        for i in list_of_cctv_vids:
          opText.append(i.split("/")[-1])
        while(len(opText)<5):
          opText.append("No file provided!")


    return opText[0], op[0], opText[1], op[1], opText[2], op[2], opText[3], op[3], opText[4], op[4]


# In[21]:


demo = gr.Interface(title = "Patrol Pal", description="Face recogntion application",fn=patrolPal, inputs=["file","file"], outputs=[gr.Textbox(label="file name", placeholder="Filename will be displayed after running.."), gr.Image(type='PIL.Image'), gr.Textbox(label="file name", placeholder="Filename will be displayed after running.."), gr.Image(type='PIL.Image'),gr.Textbox(label="file name", placeholder="Filename will be displayed after running.."), gr.Image(type='PIL.Image'), gr.Textbox(label="file name", placeholder="Filename will be displayed after running.."), gr.Image(type='PIL.Image'), gr.Textbox(label="file name", placeholder="Filename will be displayed after running.."), gr.Image(type='PIL.Image')], server_port=5000)


# In[ ]:


demo.launch(debug=True)


# In[6]:


demo.close()


# In[ ]:




