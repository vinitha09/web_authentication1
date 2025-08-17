import cv2
import os
import shutil
import random
import csv
import pandas as pd
from PIL import Image

import Augmentor



##MEMBER_FUNCTION_NAMES
#creating_images
#dataset_builder
#rename_image
#renaming_images
#dataset_assembler
#split_train_test
#csv_file_creator
#adding_coloumn_to_existing_files
#dataset_labeler
#resizing_images

##MEMBER VARIABLES INITIALIZED
#project_d_path
#username
#user_d_path
#train_d_path
#test_d_path
#org_d_path
#cmn_high_d_path
#max_width and max_height 

##MEMBER VARIABLES INPUT
#no_usr_img
#random_img

class Dataset:
    #initializing all variables
    def __init__(self,username,gender):
        self.project_folder=os.getcwd()        #It gets the project path  (Web_authetication->)
        self.username=username                 #username 
        self.gender=gender                     #gender
        self.no_usr_img=100                    #no.of user images
        self.random_img=300                    #no.of random images 
        self.train_test_ratio=0.8              #which divides the training data and test data in particular ratio
        self.no_of_aug_imgs=200
        self.org_img_width=0
        self.org_img_height=0
        self.other_usr_images=50


        ## creating paths for folders
        data="Dataset"                      
        self.project_d_path=os.path.join(self.project_folder,data)    # Dataset path (webauthetication->Dataset)
        self.user_d_path=os.path.join(self.project_d_path,f'{self.username}_dataset')  #username_dataset folder(Dataset->username_dataset)
        self.org_d_path=os.path.join(self.user_d_path,f'original_dataset')     # (Dataset->username_dataset->original dataset) where we can store combination of real images and random images
        self.train_d_path=os.path.join(self.user_d_path,f'train_dataset')     #(Dataset->username_dataset->train_dataset)where we can store contains train images 
        self.test_d_path=os.path.join(self.user_d_path,f'test_dataset')       #(Dataset->username_dataset->test_dataset) where we can store  contains test images
        self.cmn_high_d_path=os.path.join(self.project_d_path,"common_high_dataset") #(Dataset->common_high_dataset) where random images contains.
        self.cmn_high_male_d_path=os.path.join(self.cmn_high_d_path,"Male_dataset")
        self.cmn_high_female_d_path=os.path.join(self.cmn_high_d_path,"Female_dataset")
        self.haarcascade_file_path=os.path.join(self.project_folder,"haarcascade_frontalface_default.xml")
        self.duplicate_folder_path="duplicate_dataset"


        ##creating folders using os.makedirs
        #if username_path not exists it creats and also creates org_path train_path test_path
        if not os.path.exists(self.user_d_path):
            os.makedirs(self.user_d_path)
            os.makedirs(self.org_d_path)
            os.makedirs(self.train_d_path)
            os.makedirs(self.test_d_path)
        #if username_path exist but org_path,train_path,test_path not exists it creats all paths simultaneously
        elif not os.path.exists(self.org_d_path):
            os.makedirs(self.org_d_path)
            os.makedirs(self.train_d_path)
            os.makedirs(self.test_d_path) 


        #access the camera in user pc
        cam=cv2.VideoCapture(0)
        #gets the width and height of camera picture
        self.max_width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.max_height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
        

    #creating new images for user to build dataset
    def creating_images(self,img_count):
        #set the max_width and max_height for the cam variable
        cam=cv2.VideoCapture(0)
        cam.set(3,self.max_width)
        cam.set(3,self.max_height)
        
        #if img_count is less than no.of user images then enters the loop
        ##no_usr_img+start_count because it saves the images in the format of 21 if already 20 images exist
        face_cascade = cv2.CascadeClassifier(self.haarcascade_file_path)
        start_count=img_count
        while img_count<self.no_usr_img+start_count:
            # Read a frame from the camera
            ret, frame = cam.read()

            # Convert the frame to grayscale for face detection
            #BGR means a colour image reads as BGR 
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the frame
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)

            # Draw rectangles around the detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Display the frame
            cv2.imshow('Press s key to capture', frame)

            # Check for the 's' key press
           
            # Save the first detected face
            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                roi = frame[(y):(y + h), (x):(x + w)]
                image_path = os.path.join(self.org_d_path, f'{self.username}_{img_count}.jpg' )
                img_count+=1
                
                cv2.imwrite(image_path, roi)
                print(f'Saved: {image_path}')
                # Pause for a moment to avoid multiple captures with a single press
            cv2.waitKey(200)
            # Press q key to exit from video
            if cv2.waitKey(1)== ord('q'):
                break 
        #releases the camera 
        cam.release()
        cv2.destroyAllWindows()

        

    ##Used to build Dataset for user
    def dataset_builder(self):
        img_count = 1
        while True:
            #joins the path of original_dataset and images in it.
            img_file = os.path.join(self.org_d_path , f'{self.username}_{img_count}.jpg' )
            #if the image exist in that path img_count increses by one
            if os.path.exists(img_file):
                img_count+=1
            else:
                print("no of images is", img_count)
                break
        # calling creating images function
        self.creating_images(img_count)
       

    #used to create dupicate folder for any folder
    def create_duplicate_folder(self,directory):
        source_folder=directory
        destination_path=os.path.join(self.user_d_path,self.duplicate_folder_path)
        shutil.copytree(source_folder,destination_path)


    #data augumentation
    def data_augmentation(self,source_data):
        dest_data=source_data
        # Initialize the Augmentor Pipeline
        p = Augmentor.Pipeline(source_data,dest_data)

        # Customize the filename format with "{count}" placeholder
    

        # Add augmentation operations to the pipeline
        p.flip_left_right(probability=0.4)
        p.flip_top_bottom(probability=0.8)
        p.rotate90(probability=0.1)

        # Number of samples to generate
        num_of_samples =self.no_of_aug_imgs

        # Generate augmented images
        p.sample(num_of_samples)


    ##adding images from other user_dataset
    def copy_other_user_images(self,spec_username_lst):
        for spec_username in spec_username_lst:
            ##joining the Dataset folder with other_username dataset
            src=os.path.join(self.project_d_path,f'{spec_username}_dataset')
            #joining the other_username dataset with their original_dataset
            image_files_path=os.path.join(src,"original_dataset")
            #storing image names in list if it startswith their name
            image_files = [file for file in os.listdir(image_files_path) if file.startswith(spec_username)]
            #randomly taking 10 images from other_user
            smp_img_files=random.sample(image_files,self.other_usr_images)
            #destination_path is adding other_username images into the user original dataset
            for img in smp_img_files:
                #adding the paths of org_path with img and dst_path with img which are described from above code
                org_img_path=os.path.join(image_files_path,img)
                dst_img_path = os.path.join(self.org_d_path,img)

                #used to paste the org_img_path images to destination_path
                shutil.copyfile(org_img_path,dst_img_path)



    

    
    #crop the images to face and  save in same file
    def face_detect_save(self,directory_path):
        model_path = self.haarcascade_file_path
        img_path =self.org_d_path
        face_cascade = cv2.CascadeClassifier(self.haarcascade_file_path)

        for i in os.listdir(img_path):

            if i.endswith(('jpg', 'jpeg', 'png')):
                img_path1 = os.path.join(img_path, i)
                img = cv2.imread(img_path1)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

                for (x, y, w, h) in faces:
                    roi = img[y:y + h, x:x + w]
                    save_path = os.path.join(directory_path, i)
                    cv2.imwrite(save_path, roi)


    

    #renaming the images in common_high_dataset
    
    def rename_image(self,src_path,old_name, new_prefix):
        for filename in os.listdir(src_path):
            if filename.startswith(old_name):
                split_lst = old_name.split('.')
                # Construct the new filename by replacing the old prefix with the new prefix
                #len(split_lst[0]):len(split_lst[0])+4 it means adding .jpg with new prefix
                new_filename = new_prefix + filename[len(split_lst[0]):len(split_lst[0])+4]
                # Construct the full paths for the old and new filenames
                old_filepath = os.path.join(src_path, filename)
                new_filepath = os.path.join(src_path, new_filename)
                # Rename the file
                os.rename(old_filepath, new_filepath)


    # renaming the images in  bulk  for common_high_dataset   
    # directory_path=directory_path wrt project_folder     
    def renaming_images(self,tag,directory):
        src_path=directory
        image_files = [file for file in os.listdir(src_path)]
       
        s=[]
        count=1
        for i in range(len(image_files)):
            s.append(tag+str(count))
            count+=1
            self.rename_image(src_path,image_files[i],s[i])




    # randomly picking n images from common high dataset and pasting into the original dataset
    def dataset_assembler(self):
        #adds the project_folder path with origin_path means common_high_dataset path and also adds destination_path means original_dataset.
        if self.gender=="Male":
            org_path=self.cmn_high_male_d_path
        elif self.gender=="Female":
            org_path=self.cmn_high_female_d_path
        else:
            print("Please choose Male or Female in this format")
        dst_path=self.org_d_path

        # storing image names in image_files list which ends with jpeg,png,jpg
        image_files = [file for file in os.listdir(org_path) if file.endswith(('jpeg', 'png', 'jpg'))]

        #removing the extra images in original_dataset so that we can control the no.of random images.
        for file in os.listdir(dst_path):
            if file.startswith('random'):
                os.remove(os.path.join(dst_path,file))

        #extract n images from image_files and store in another  variable
        smp_img_files=random.sample(image_files, self.random_img)


        for img in smp_img_files:
            #adding the paths of org_path with img and dst_path with img which are described from above code
            org_img_path=os.path.join(org_path,img)
            dst_img_path = os.path.join(dst_path,img)

            #used to paste the org_img_path images to destination_path
            shutil.copyfile(org_img_path,dst_img_path)


    
    #split the original data images into train and test
    def split_train_test(self):
    # Define paths for source and destination folders
        
        source_folder = self.org_d_path # Replace with your source folder path
        train_folder = self.train_d_path  # Replace with your train folder path
        test_folder = self.test_d_path     # Replace with your test folder path

        # Define the ratio for splitting (e.g., 80% train, 20% test)
        train_ratio = self.train_test_ratio

        # Create destination folders if they don't exist

        # List all image files in the source folder
        image_files = [file for file in os.listdir(source_folder) if file.endswith(('jpeg', 'png', 'jpg'))]

        # Shuffle the list to randomize the selection
        random.shuffle(image_files)

        # Calculate the number of files for training
        train_count = int(len(image_files) * train_ratio)

        # Split files into train and test sets
        train_images = image_files[:train_count]
        test_images = image_files[train_count:]

        # Move images to the respective folders
        for image in train_images:
            src_path = os.path.join(source_folder, image)
            dst_path = os.path.join(train_folder, image)
            shutil.copyfile(src_path, dst_path)
        
        for image in test_images:
            src_path = os.path.join(source_folder, image)
            dst_path = os.path.join(test_folder, image)
            shutil.copyfile(src_path, dst_path)


    # Create a new CSV file with multiple columns
    def create_csv_with_columns(self,file_path, column_names, column_data_lists):
        try:
            # Create a new CSV file
            with open(file_path, mode='w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)

                # Write the header row
                csv_writer.writerow(column_names)

                # Write the data rows
                for row_data in zip(*column_data_lists):
                    csv_writer.writerow(row_data)

            print(f"New CSV file '{file_path}' created successfully.")
        except Exception as e:
            print(f"Error creating CSV file: {e}")


    # add coloumns to existing csv file
    def add_columns_to_csv(self,existing_csv_path, new_column_names, new_column_data):
        try:
            # Read the existing CSV file
            with open(existing_csv_path, mode='r') as csv_file:
                csv_reader = csv.reader(csv_file)
                rows = list(csv_reader)


            # Add the new column headers
            for column_name in new_column_names:
                rows[0].append(column_name)

            # Add the new column data to each row
            for i in range(1, len(rows)):
                for column_data in new_column_data:
                    rows[i].append(column_data[i - 1])

            # Write the updated data back to the CSV file
            with open(existing_csv_path, mode='w', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerows(rows)

            print(f"New columns added to CSV file '{existing_csv_path}' successfully.")
        except Exception as e:
            print(f"Error adding new columns to CSV file: {e}")

    

    #label the dataset with 0 and 1
    def dataset_labelling(self,label_directory_path,file_name):
        src=label_directory_path
        img_files=[file for file in os.listdir(src) if file.endswith(('jpeg', 'png', 'jpg'))] 
        #final csv file  in which one coloumn is names fo file and other coloumn is 1 if it is my photo else 0
        img_label=[]
        for img in img_files:
            if img.startswith(self.username):
                new_variable=1
                img_label.append(new_variable)
            else:
                new_variable=0
                img_label.append(new_variable)
        #creating username_train/test.csv
        file=self.username+"_"+file_name+".csv"
        file_path=os.path.join(src,file)

        ##Giving coloumn names as image_names and image_labels
        coloumn_names=["image_names","image_label"]

        #add multiple coloumns in new csv file
        self.create_csv_with_columns(file_path,coloumn_names, [img_files,img_label])

        # existing_csv_path = "image.csv"
        #new_column_names = ["NewColumn1"]
        #new_column_data1 = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]


        # Add the new columns to the existing CSV file
        # Add the new columns to the existing CSV file
        #self.add_columns_to_csv(file_path, new_column_names, [new_column_data1])



    ##Finding the average shape of cropped face pictures
    def set_img_shape(self):
        #storing imges paths in list if image name starts with username
        image_files = [file for file in os.listdir(self.org_d_path) if file.startswith((self.username))]
        a=[]
        for i in image_files:
            #converting jpg to array
            img=cv2.imread(os.path.join(self.org_d_path,i))
            #adding each image shape in a list
            a.append(img.shape[0])
        #average all images sizes 
        org_img_shape=(sum(a)//len(a))
        #assigning the values to member variables
        self.org_img_height,self.org_img_width=org_img_shape,org_img_shape


    #resizing images in specific folder
    def resize_and_replace(self,resize_directory_path,img_width,img_height):
        src=resize_directory_path
        new_size=(img_width,img_height)
        # Loop through each file in the input folder
        for filename in os.listdir(src):
            input_path = os.path.join(src, filename)

            try:
                # Open the image file
                with Image.open(input_path) as img:
                    # Resize the image
                    resized_img = img.resize(new_size)

                    # Save the resized image, replacing the original file
                    resized_img.save(input_path)

                print(f"Resized and replaced {filename} successfully.")
            except Exception as e:
                print(f"Error resizing and replacing {filename}: {e}")

    
  



    
                

    #Initialize all the function 
    #oth_usr_lst=list of other dataset usernames
    def dataset_initializer(self):   
        self.dataset_builder()
        self.create_duplicate_folder(self.org_d_path)
        self.data_augmentation(self.org_d_path)
        self.renaming_images(self.username,self.org_d_path)
        #oth_usr_lst take input for function
        #self.copy_other_user_images(oth_usr_lst)
        #token=1 take input in main fun when required
        #if token==1:
        #     self.renaming_images("random_f",self.cmn_high_female_d_path)
        #     self.renaming_images("random_m",self.cmn_high_male_d_path)
        self.dataset_assembler()
        self.split_train_test()
        self.dataset_labelling(self.train_d_path,"train")
        self.dataset_labelling(self.test_d_path,"test")
        self.set_img_shape()
        self.resize_and_replace(self.train_d_path,self.org_img_width,self.org_img_height)
        self.resize_and_replace(self.test_d_path,self.org_img_width,self.org_img_height)
    

   
