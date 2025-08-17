import os
import cv2
import dataset
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.models import load_model
from keras.models import model_from_json
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,Dropout



##MEMBER VARIABLES
#model_path
#user_model_path



##MEMBER FUNCTIONS
#image_to_input_arrays
#train_test_save_cnn_model
#save_best_models
#predicting_live_images


 
#Inherits Parent class from dataset module
class CNN_model(dataset.Dataset):

    #Initialize the variables
    def __init__(self,username,gender):
        super().__init__(username,gender)
        models="Trained_cnn_models"
        self.models_path=os.path.join(self.project_folder,models)                        #(Web_authentication_1->Trained_cnn_model)
        self.user_model_path=os.path.join(self.models_path,f'{self.username}_cnn_model')  #(Trained_cnn_mdoel->username_cnn_model)
        if not os.path.exists(self.user_model_path):                                      #if path not exist it creats
            os.makedirs(self.user_model_path)
    

    #It converts jpg images to arrays
    def images_to_input_array(self,directory_path):
        #directory path can be train_dataset path or test_dataset path
        src_path=os.path.join(self.project_folder,directory_path)
        x=[]
        for file in os.listdir(src_path):
            if file.endswith(('jpg','jpeg','png')):
                img_path=os.path.join(src_path,file)
                #cv2.imread converts jpg img to array and divide by 255.0 is for scaling the pixels into (0 to 1)
                x.append(cv2.imread(img_path)/255)
            if  file.endswith(('csv')):
                label_path=os.path.join(src_path,file)
                #reads the csv file
                df=pd.read_csv(label_path)
                #print(df)
                #print(df.loc[:,'image_label'])
                #print(list(df.loc[:,'image_label']))


                #stores the only specific coloumn from csv file
                y_arr=np.array(df.loc[:,'image_label'])
        #It stack one more axis to the list It converts 3d to 4d
        x_arr=np.stack(x,axis=0)
        #print(x_train[0],type(x_train[0]))
        #print(x_train_arr.shape)
        return [x_arr,y_arr]

    #Contains model architecture and It trains and test the model
    def train_test_save_cnn_model(self,arch_type=1):
        #assigns trauin_path in src_train
        src_train=self.train_d_path
        #assigns trauin_path in src_test
        src_test=self.test_d_path
        #stores the x_train,y_train data by calling images_to_input_array function with src_train path
        [x_train,y_train]=self.images_to_input_array(src_train)
        #stores the x_test,y_test data by calling images_to_input_array function with src_test path
        [x_test,y_test]=self.images_to_input_array(src_test)
        #if arch_type is 1 enters 1 st model_architecture 
        
        if arch_type==1:
            #cnn architeture
            model = Sequential()
            model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(int(self.org_img_height),int(self.org_img_width),3)))
            model.add(MaxPooling2D((2, 2)))
            model.add(Conv2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D((2, 2)))
            model.add(Conv2D(128, (3, 3), activation='relu'))
            model.add(MaxPooling2D((2, 2)))
            model.add(Flatten())
            model.add(Dense(128, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(1, activation='sigmoid'))
        elif arch_type==2:
            model = Sequential()
            model.add(Conv2D(32, (3, 3), activation='relu', padding='same', name='conv_1', 
                             input_shape=(self.org_img_height,self.org_img_width, 3)))
            model.add(MaxPooling2D((2, 2), name='maxpool_1'))
            model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_2'))
            model.add(MaxPooling2D((2, 2), name='maxpool_2'))
            model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_3'))
            model.add(MaxPooling2D((2, 2), name='maxpool_3'))
            model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_4'))
            model.add(MaxPooling2D((2, 2), name='maxpool_4'))
            model.add(Flatten())
            model.add(Dropout(0.5))
            model.add(Dense(512, activation='relu', name='dense_1'))
            model.add(Dense(128, activation='relu', name='dense_2'))
            model.add(Dense(1, activation='sigmoid', name='output'))
            #This model architecture  reference link
            #https://towardsdatascience.com/applied-deep-learning-part-4-convolutional-neural-networks-584bc134c1e2
        
        #It initialize optimizer,loss_functin,metrics before fitting to train data.
        model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
        #Train_data fits to model
        model.fit(x_train, y_train, epochs=7)
       
        # Save the model architecture to a file
        model_json = model.to_json()
        with open(os.path.join(self.user_model_path,f"{self.username}_model_architecture.json"), "w") as json_file:
            json_file.write(model_json)
        
        #Saves the model weights in specific_folder
        model.save_weights(os.path.join(self.user_model_path,f"{self.username}_model_save_weights.h5"))

        model.save(os.path.join(self.user_model_path,f"{self.username}_model.h5"))
        
        

        #return the test_acc and test_loss by evaluting model
        test_loss, test_acc = model.evaluate(x_test,y_test,verbose=2)
        return [test_loss,test_acc]

    #Save the best model by using test_acc.
    def save_best_models(self):
        print(self.org_img_height,self.org_img_width)
        [test_loss1,test_acc1]=self.train_test_save_cnn_model(arch_type=2)
        #If test_acc is greater than 0.85 It stores the model.
        if test_acc1<=0.85:
            [test_loss2,test_acc2]=self.train_test_save_cnn_model(arch_type=1)
            
        
            

    
        
    #It initialize  all functions
    def model_initializer(self):

        self.images_to_input_array(self.train_d_path)
        self.images_to_input_array(self.test_d_path)
        self.set_img_shape()
        self.save_best_models()

class Login_fun(CNN_model):
    #Predicting the live images Whether the person or not
    def predicting_live_image(self):
        cnn_model_path=self.user_model_path

        #Save the live image by using cv2 library 
        cam=cv2.VideoCapture(0)
        face_cascade = cv2.CascadeClassifier(self.haarcascade_file_path)
        while True:
            # Read a frame from the camera
            ret, frame = cam.read()

            # Convert the frame to grayscale for face detection
            #BGR means a colour image reads as BGR 
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)

            # Draw rectangles around the detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x-20, y-20), (x + w+20, y + h+20), (0, 0, 255), 2)

            # Display the frame
            cv2.imshow('Press s key to capture', frame)

            # Check for the 's' key press
            if cv2.waitKey(1) == ord('s'):
                # Save the first detected face
                ##y=rows(height),x=coloumns(width)
                if len(faces) > 0:
                    (x, y, w, h) = faces[0]
                    roi = frame[(y):(y + h), (x):(x + w)]
                    image_path = os.path.join(self.user_model_path, f'live_{self.username}_.jpg' )
                    
                    cv2.imwrite(image_path, roi)
                    print(f'Saved: {image_path}')
                    # Pause for a moment to avoid multiple captures with a single press
                    break
            # Press q key to exit from video
            if cv2.waitKey(1)== ord('q'):
                break 
        self.set_img_shape()
        self.resize_and_replace(self.user_model_path,self.org_img_height,self.org_img_width)

     
        # json_path =os.path.join(cnn_model_path,f"{self.username}_model_architecture.json")
        # with open(json_path, 'r') as json_file:
        #     loaded_model_json = json_file.read()
        #     loaded_model = model_from_json(loaded_model_json)
        ##load weights of the best model stored in specific path
        new_model=load_model(os.path.join(cnn_model_path,f"{self.username}_model.h5"))
        #new_model.load_weights(os.path.join(cnn_model_path,"my_model_weights.h5"))
        
        #Convert the live image jpg to array
        live_img_array=(cv2.imread(image_path)/255.0)
        


        #Expands the axis of img_array from 3d to 4d
        new_live_img_array=np.expand_dims(live_img_array, axis=0)
       
        #Stores the img probability by predicting the live image
        img_prob=new_model.predict(new_live_img_array)
        print(img_prob)
        #If img probability is greater than 85% then it prints the Identification is successful
        if img_prob>0.85:

            print("Your identification is successful")
            return 1
        
        #If img probability is less than 85% then it prints the Identification is unsuccessful
        else:
            print("Your identification is unsuccessful")
            return 0
            






    









