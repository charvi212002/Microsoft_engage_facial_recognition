# importing the necessary dependencies 

import cv2
import os
import FaceEmotionDetectionModel  
import torch
import torchvision
import numpy as np

def predict_expression(file_path):

    # Read the input image
    img = cv2.imread(file_path)

    # if we do not get anyfile path
    if img is None:                     
        print('Wrong path:', file_path)

    # if we get an image with valid filepath
    else:   
        print("Path:",file_path)                            
        #print("HI")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        #print(type(faces))
        #print(faces)

        #if there is no face being detected by haarcascade classifier
        if faces == ():  
            faces = gray

        #if there is a face detected then crop the rest of the part except the face
        else:            
            for (x, y, w, h) in faces:
                cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 0, 255), 2)
                faces = gray[y:y + h, x:x + w]

        test_img = cv2.resize(faces, (48,48))  

        #loading the model
        net = FaceEmotionDetectionModel.Model(7)       
        checkpoint = torch.load(os.path.join('model','test_model.t7'),
                                 map_location=torch.device('cpu'))    
        net.load_state_dict(checkpoint['net'])
        net.eval()

        #making predictions
        x1 = torchvision.transforms.functional.to_tensor(test_img.astype('uint8').reshape(48,48))

        # Changing the shape as required by the convolutional layer 
        prediction = net(x1.view(1,1,48,48))   

        # the classes of emotion
        classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']      
        val = np.argmax(prediction.detach().numpy())                                       

        #print(classes[val])       
        
        return classes[val]     # returning the predicted emotion over the face