import cv2
import numpy
import numpy as np
import csv
import os
import os.path
import pandas as pd
from os import system
import math
from math import acos
from math import sqrt
from math import degrees
import array as arr
from numpy import asarray
from numpy import savetxt
from numpy import loadtxt
import matplotlib as plt
import matplotlib.pyplot as plt 

'''
Michelle Liu
EACS Bounds E

Last Updated: Wednesday, December 13th, 2023

This code provides real time joint angle feedback to PT users and their doctors.
Doctors are able to view saved patient exercise data and create exercises for their patients to do.
Patients are able to preform exercises with auditory and visual real time feedback as well as view their own saved data.

On My Honor, I have neither given nor recieved any unauthorized aid on this work.
                                                                       - Michelle Liu'''
#Define the calc angles function
def calc_angles():
    global final_angle
    global frame
    global color_1_parts_of_image
    global color_2_parts_of_image
    global color_3_parts_of_image
    global color_4_parts_of_image
    global customized_image
    global out_of_view
    out_of_view = False #initialize as not said yet
    #convert to hsv
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #define color ranges (cyan, pink, green yellow) 
    lower_color1 = np.array([90, 127, 50]) #cyan
    upper_color1 = np.array([128, 255, 255])
    lower_color2 = np.array([142, 127, 50]) #pink
    upper_color2 = np.array([173, 255, 255])
    lower_color3 = np.array([32, 127, 50]) #green
    upper_color3 = np.array([68, 255, 255])
    lower_color4 = np.array([22, 127, 50]) #yellow
    upper_color4 = np.array([30, 255, 255])
    #create masks
    mask_color1 = cv2.inRange(hsv, lower_color1, upper_color1)
    mask_color2 = cv2.inRange(hsv, lower_color2, upper_color2)
    mask_color3 = cv2.inRange(hsv, lower_color3, upper_color3)
    mask_color4 = cv2.inRange(hsv, lower_color4, upper_color4)
    kernel = np.ones((5, 5), np.uint8) #make kernel 
    #use the masks
    color_1_parts_of_image = cv2.bitwise_and(frame, frame, mask=mask_color1)
    color_2_parts_of_image = cv2.bitwise_and(frame, frame, mask=mask_color2)
    color_3_parts_of_image = cv2.bitwise_and(frame, frame, mask=mask_color3)
    color_4_parts_of_image = cv2.bitwise_and(frame, frame, mask=mask_color4)
    #create erosion filter
    eroded_filtered_C1 = cv2.erode(color_1_parts_of_image, kernel, iterations=1)
    eroded_filtered_C2 = cv2.erode(color_2_parts_of_image, kernel, iterations=1)
    eroded_filtered_C3 = cv2.erode(color_3_parts_of_image, kernel, iterations=1)
    eroded_filtered_C4 = cv2.erode(color_4_parts_of_image, kernel, iterations=1)
    filtered_images = [eroded_filtered_C1, eroded_filtered_C2, eroded_filtered_C3, eroded_filtered_C4]
    #combine masks
    customized_image = cv2.bitwise_or(color_1_parts_of_image, color_2_parts_of_image)
    customized_image = cv2.bitwise_or(customized_image, color_3_parts_of_image)
    customized_image = cv2.bitwise_or(customized_image, color_4_parts_of_image)
    #convert video to grayscale
    grayscale_customized_video = cv2.cvtColor(customized_image, cv2.COLOR_BGR2GRAY)
    #convert color 1 to grayscale
    grayscale_color1 = cv2.cvtColor(color_1_parts_of_image, cv2.COLOR_BGR2GRAY)
    M1 = cv2.moments(grayscale_color1)
    #color 1: calculate x,y coordinate of center
    if M1["m00"] != 0:
        cX_1 = int(M1["m10"] / M1["m00"])
        cY_1 = int(M1["m01"] / M1["m00"])
    #if the object is not shown, then x and y coordinates are (0, 0)
    else:
        cX_1, cY_1 = 0, 0
    #calculate x, y coordinate of center
    if M1["m00"] != 0:
        cX_1 = int(M1["m10"] / M1["m00"])
        cY_1 = int(M1["m01"] / M1["m00"]) 
        cv2.circle(customized_image, (cX_1, cY_1), 5, (255, 255, 255), -1)
        cv2.putText(customized_image, "centroid", (cX_1 - 25, cY_1 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    else:
    #if the object is not shown, then x and y coordinates are (0, 0)
        cX_1, cY_1 = 0, 0
    grayscale_color2 = cv2.cvtColor(color_2_parts_of_image, cv2.COLOR_BGR2GRAY)
    M2 = cv2.moments(grayscale_color2)
    #color 2: calculate x,y coordinate of center
    if M2["m00"] != 0:
        cX_2 = int(M2["m10"] / M2["m00"])
        cY_2 = int(M2["m01"] / M2["m00"])
    #if the object is not shown, x and y coordinates are (0, 0)
    else:
        cX_2, cY_2 = 0, 0
    if M2["m00"] != 0:
    #Calculate the center, and put circle and text that represents the center
        cX_2 = int(M2["m10"] / M2["m00"])
        cY_2 = int(M2["m01"] / M2["m00"])
        cv2.circle(customized_image, (cX_2, cY_2), 5, (255, 255, 255), -1)
        #put text on center
        cv2.putText(customized_image, "centroid", (cX_2 - 25, cY_2 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    else:
    #if the object is not shown, x and y coordinates are (0, 0)
        cX_2, cY_2 = 0, 0 
    grayscale_color3 = cv2.cvtColor(color_3_parts_of_image, cv2.COLOR_BGR2GRAY)
    M3 = cv2.moments(grayscale_color3)
    #color 3: calculate x,y coordinate of center
    if M3["m00"] != 0:
        cX_3 = int(M3["m10"] / M3["m00"])
        cY_3 = int(M3["m01"] / M3["m00"])
    #if the color is not shown, x and y coordinates are (0, 0)
    else:
        cX_3, cY_3 = 0, 0
    if M3["m00"] != 0:
    #Calculate the center, and put circle and text that represents center
        cX_3 = int(M3["m10"] / M3["m00"])
        cY_3 = int(M3["m01"] / M3["m00"])
        cv2.circle(customized_image, (cX_3, cY_3), 5, (255, 255, 255), -1)
        cv2.putText(customized_image, "centroid", (cX_3 - 25, cY_3 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    else:
    #if the object is not shown, x and y coordinates are (0, 0)
        cX_3, cY_3 = 0, 0 
    grayscale_color4 = cv2.cvtColor(color_4_parts_of_image, cv2.COLOR_BGR2GRAY)
    M4 = cv2.moments(grayscale_color4)
    #color 4: calculate x,y coordinate of center
    if M4["m00"] != 0:
        cX_4 = int(M4["m10"] / M4["m00"])
        cY_4 = int(M4["m01"] / M4["m00"])
    else:
    #if the object is not shown, x and y coordinates are (0, 0)
        cX_4, cY_4 = 0, 0
    if M4["m00"] != 0:
    #calculate center, and put circle and text that represents the center
        cX_4 = int(M4["m10"] / M4["m00"])
        cY_4 = int(M4["m01"] / M4["m00"])
        cv2.circle(customized_image, (cX_4, cY_4), 5, (255, 255, 255), -1)
        cv2.putText(customized_image, "centroid", (cX_4 - 25, cY_4 - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    else:
        cX_4, cY_4 = 0, 0
    #draw the lines of the stick figures
    cv2.line(customized_image, (cX_1, cY_1), (cX_2, cY_2), (255, 0, 0), 2)
    cv2.line(customized_image, (cX_3, cY_3), (cX_4, cY_4), (0, 0, 255), 2)
    #start slopes with random values to avoid div 0
    M_34 = 2
    M_12 = 1
    line_OC4 = 2
    line_OC1 = 1
    #check for div 0
    final_angle = 0
    if cX_2 - cX_1 !=0 and cX_4 - cX_3 != 0 and ((M_12 - M_34)) != 0 and (2 * line_OC4 * line_OC1) !=0: 
        #recalculate intersection and other values
        #slope of C3 and C4
        M_34 = ((cY_4 - cY_3) / (cX_4 - cX_3))
        #slope of C2 and C1
        M_12 = ((cY_2 - cY_1) / (cX_2 - cX_1))
        #"b" value of C3 and C4
        b_34 = (cY_3 - (M_34 * cX_3))
        #"b" value of C2 and C1
        b_12 = (cY_1 - (M_12 * cX_1))
        Ox = ((b_34 - b_12) / (M_12 - M_34))
        #Ox is the x intersection
        Oy = M_12 * ((b_34 - b_12) / (M_12 - M_34)) + b_12
        #Oy is the y intersection
        #define intersection
        intersection_O = (Ox, Oy)
        #calculate distance of lines
        line_OC1 = math.sqrt(((cX_1 - Ox) ** 2) + ((cY_1 - Oy) ** 2))
        line_OC4 = math.sqrt(((cX_4 - Ox) ** 2) + ((cY_4 - Oy) ** 2))
        line_C1C4 = math.sqrt(((cX_4 - cX_1) ** 2) + ((cY_4 - cY_1) ** 2))
        #if no div 0
        if (2 * line_OC4 * line_OC1) != 0:
            cos_B = (((line_OC1) ** 2) + (line_OC4 ** 2) - (line_C1C4 ** 2)) / (2 * line_OC4 * line_OC1)
            #check for range of cosine
            if -1 <= cos_B and cos_B <= 1:
                angle_before_subtract = acos(cos_B)
                angle_degrees = (angle_before_subtract) * 57.2958
                final_angle = angle_degrees
                #if move to the right
                if cX_1 > Ox:
                    final_angle = 180 - angle_degrees
                    if final_angle > 90:
                        #if greater than 90 degrees, final angle = 0
                        final_angle = 0
                #else if move to the left
                elif cX_1 < Ox:
                    final_angle = -1 * (180 - angle_degrees)
                    #if less than -90 degrees, final angle = 0
                    if final_angle < -90:
                        final_angle = 0
                        #place text on the screen: angle = angle measure
        cv2.putText(customized_image, "Angle: {:.2f}".format(final_angle), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        print(final_angle)
        #return the final_angle for later use
        return final_angle
        """else:
            #say that "dots are out of view" if the markers are out of view (div 0)
                if out_of_view == False: 
                    print(" in view of camera")
                    system("say put dots in view of camera")
                    out_of_view = True
        else:
            if out_of_view == False:
            #say that "dots are out of view" if the markers are out of view (div 0)
                    print("error 182")
                    system("say put dots in view of camera")
                    out_of_view = True
    else:
        if out_of_view == False:
            #say that "dots are out of view" if the markers are out of view (div 0)
                    print("error 187")
                    system("say put dots in view of camera")
                    out_of_view = True"""

#BEGIN PROGRAM
print("welcome to our physical therapy program, which can be used by doctors and patients in and outside of a clinical setting.")
print("if you are a doctor you can create a new exercise for your patients.")
print("with this program you can specify the angle measure, direction of movement, and repitition count of your patient's exercise.")
print("you can also view patient data which has been saved in the same folder as this program.")
print("as a patient you can preform exercises created by your doctor, or preset exercises with various difficulties.")
print("during exercises, you will be provided with real time feedback on your movements and can watch your movemnts through a live 2D stick figure model") 
print("you can also view your own saved exercise data")
print("\n")
#ask if user is doctor or patient
user_begin = False #define valid variable as false
while user_begin == False: #while variable remains false...
    user = input("are you a doctor or a patient? type 'doctor' or 'patient': ")
    if user == 'doctor' or user == 'patient':
        user_begin = True #redefine valid variable as true 
    
        print("please type 'doctor' or 'patient'")
       
#DOCTOR INTERFACE 
if user == 'doctor':
    doctor_choice = input("would you like to view patient exercise data or create a new exercise? type 'v' to view data or type 'c' to create a new exercise: ")
    while doctor_choice != 'v' and doctor_choice != 'c':
        print("please select an option")
    if doctor_choice == 'v': #if doctor chooses to view past data...
        filename_valid = False #define filename_valid variable as false
        while filename_valid == False: #while the variable remains false...
            file_open = input("what is the name of your patient's data file?: ")
            if os.path.isfile(file_open) == True: #if the file is valid...
                filename_valid = True #redefine variable as true
                #display file data for patient
                print("data from your patient's exercise: ") 
                doctorread_userdata = open(file_open, 'r') #open file
                print(doctorread_userdata .read()) #read & printdata from file
                doctorread_userdata .close() #close file
                print("the graph of this data is saved in the same location as this program.")
                print("thank you for using this program!") 
            else:
                print ("something was wrong with that filename. please try again.") #while variable remains false, inform user to try again
    if doctor_choice == 'c': #if doctor chooses to create new exercise...
        #initialize input variables
        desired_direction = 'i'
        movement_angle = 0
        movement_repeat = 0
        #ask user to define variables
        while desired_direction == 'i':
            desired_direction = input("provide exercise direction. type 'right' for movement to the right, 'left' for movement to the left, or 'both' for movement in both directions: ")
            if desired_direction == 'right' or desired_direction == 'left' or desired_direction == 'both':
                while movement_angle == 0:
                    movement_angle = input("please provide angle of wrist movement (in degrees): ")
                if int(movement_angle) > 0:
                    while movement_repeat == 0:
                        movement_repeat = input("how many times would you like the patient to repeat this exercise?: ")
                        if movement_repeat == int and movement_repeat > 0:
                            print("all variables selected!")
            else:
                print("please choose 'r', 'l', or 'b'")
        exercise_name = input("what would you like to name this exercise?: ")
        #save exercise data onto csv file
        fields = [['Movement Direction', 'Movement Angle','Movement Repetition']]
        rows = [[desired_direction, movement_angle, movement_repeat]]
        exercise_file = exercise_name + ".csv"
        #write exercise data onto csv file  
        with open(exercise_file, 'w') as csvfile:  
            #create a csv writer object  
            csvwriter = csv.writer(csvfile)
            #write the data rows  
            csvwriter.writerows(fields)
            csvwriter.writerows(rows)
            print("exercise has been sucessfully saved!")

#PATIENT INTERFACE
if user == 'patient':
    patient_choice = input("would you like to view past exercise data or do a new exercise? type 'v' to view data or type 'e' to do a new exercise: ")
    if patient_choice != 'v' and patient_choice != 'e': #patient selects to view or exercise
        print("please select an option")
    #PATIENT VIEW DATA 
    if patient_choice == 'v': #if patient selects view...
        file_open = input("what is the name of the exercise file you would like to view (please include the extension)?: ") #patient selects file
        #display file data for patient
        patient_dataviewing = open(file_open, 'r') #open file
        print("\njoint angle data from this exercise:")
        print(patient_dataviewing.read()) #read & printdata from file
        patient_dataviewing.close() #close file
    if patient_choice == 'e': #if patient picks exercise... 
        exercise_pick = input("would you like to do a preset exercise or an exercise given by your doctor? please type 'preset' or 'doctor': ")
        if exercise_pick == 'preset': # patient selects preset exercise  
            difficulty_choice = input("please type your desired difficulty: 'b' for beginner, 'i' for intermediate, or 'a' for advanced: ")
            #set variables per difficluty choice: max angle, direction, and repeat number
            if difficulty_choice == 'b':
                desired_max_angle = 10
                desired_direction = 'left'
                desired_direction_name = 'left'
                desired_repeat = 2
                int_desired_max_angle = int(desired_max_angle)
                int_desired_repeat = int(desired_repeat)
            if difficulty_choice == 'i':
                desired_max_angle = 20
                desired_direction = 'left'
                desired_direction_name = 'left'
                desired_repeat = 5
                int_desired_max_angle = int(desired_max_angle)
                int_desired_repeat = int(desired_repeat)
            if difficulty_choice == 'a':
                desired_max_angle = 30
                desired_direction = 'left'
                desired_direction_name = 'left'
                desired_repeat = 10
                int_desired_max_angle = int(desired_max_angle)
                int_desired_repeat = int(desired_repeat)
        if exercise_pick == 'doctor': #read in exercise data from doctor file (also check if file entered is valid)
            filename_valid = False #define valid variable as false
            while filename_valid == False: #while variable remains false...
                doctor_exercise_import = input("please type the name of the exercise file given by your doctor: ")
                if os.path.isfile(doctor_exercise_import) == True: #if the file is valid...
                    filename_valid = True #redefine valid variable as true
                    #read in exercise data 
                    #movement direction
                    doctorexercise_readin = pd.read_csv(doctor_exercise_import, usecols=[0]) #read in CSV file, specify column
                    desired_direction = doctorexercise_readin.loc[doctorexercise_readin.index[0]] #define variable by row
                    desired_direction = desired_direction.tolist() #convert number to list (removes name, dtype)
                    desired_direction = str(desired_direction) #convert list to string
                    desired_direction = desired_direction.replace('Movement Direction', '') #format: remove label
                    desired_direction = desired_direction.replace('[', '') #format: remove bracket
                    desired_direction = desired_direction.replace(']', '') #format: remove bracket
                    desired_direction = desired_direction.replace("'", '') #format: remove '
                    desired_direction = desired_direction.strip() #format: strip whitespace
                    print(desired_direction) #for testing
                    #set name for desired direction 
                    if desired_direction == 'both':
                        desired_direction_name = 'both right and left'
                    if desired_direction != 'both': 
                        desired_direction_name = desired_direction
                    #angle 
                    doctorexercise_readin = pd.read_csv(doctor_exercise_import, usecols=[1]) #read in CSV file, specify column
                    desired_max_angle = doctorexercise_readin.loc[doctorexercise_readin.index[0]] #define variable by row
                    desired_max_angle = desired_max_angle.tolist() #convert number to list (removes name, dtype)
                    desired_max_angle = str(desired_max_angle) #convert list to string
                    desired_max_angle = desired_max_angle.replace('Movement Angle','') #format: remove label
                    desired_max_angle = desired_max_angle.replace('[', '') #format: remove bracket
                    desired_max_angle = desired_max_angle.replace(']', '') #format: remove bracket
                    desired_max_angle = desired_max_angle.replace("'", '') #format: remove '
                    desired_max_angle = desired_max_angle.strip() #format: strip whitespace
                    print(desired_max_angle) #for testing
                    int_desired_max_angle = int(desired_max_angle)
                    #repetition number
                    doctorexercise_readin = pd.read_csv(doctor_exercise_import, usecols=[2]) #read in CSV file, specify column
                    desired_repeat = doctorexercise_readin.loc[doctorexercise_readin.index[0]] #define variable by row
                    desired_repeat = desired_repeat.tolist() #convert number to list (removes name, dtype)
                    desired_repeat = str(desired_repeat) #convert list to string
                    desired_repeat = desired_repeat.replace('Movement Repetition', '') #format: remove label
                    desired_repeat = desired_repeat.replace('[', '') #format: remove bracket
                    desired_repeat = desired_repeat.replace(']', '') #format: remove bracket
                    desired_repeat = desired_repeat.replace("'", '') #format: remove '
                    desired_repeat = desired_repeat.strip() #format: strip whitespace
                    print(desired_repeat) #for testing 
                    int_desired_repeat = int(desired_repeat)
                else:
                    print ("Something was wrong with that filename. Please try again.") #while variable remains false, inform user to try again
        #START EXERCISE 
        #tell user to start program (auditory & visual)
        '''system('say hello! i am Charlie your personal physical therapist. i see you have selected your exercise.')
        system('say please hold your hand in front of your camera with the shapes facing it. ')
        system('say please press s to start.')
        print("hello! i am Charlie your personal physical therapist. i see you have selected your exercise.\nplease hold your hand in front of your camera with the shapes facing it.")
        ''' 
        #inform user of their exercise 
        exercise_info = "your selected exercise is to move your wrist " + str(desired_max_angle) + " degrees " + desired_direction_name + " " + str(desired_repeat) + " times"
        start = input('please press s to start: ')
        if start == 's': 
            #welcome user
            system('say you can exit this program by finishing the exercise sucessfully or pressing the escape key.')
            print('you can exit this program by finishing the exercise sucessfully or pressing the escape key.')
            system('say lets begin!')
            print('lets begin!')
            #initialize graph array
            x_array = []
            angle_array = []
            x = 0
            #initialize other exercise variables              
            #start camera 
            cap = cv2.VideoCapture(1) #open the cam
            keypressed = 1 #esc key = open
            #READING JOINT ANGLE DATA
            while keypressed != 27:
                ret, frame = cap.read() #initialize camera
                #run angle calculation function
                found_angle = calc_angles()
                #display windows
                cv2.imshow('frame', frame)
                cv2.imshow('color 1 mask', color_1_parts_of_image)
                cv2.imshow('color 2 mask', color_2_parts_of_image)
                cv2.imshow('color 3 mask', color_3_parts_of_image)
                cv2.imshow('color 4 mask', color_4_parts_of_image)
                cv2.imshow('all color masks', customized_image)
                keypressed = cv2.waitKey(1) #wait for a key event (1 ms delay)
                
                #START EXERCISE (once user has produced a joint angle) 
                if found_angle == None:
                    found_angle = 0
                    print("None, 392")
                print("True, 393")
                x = x + 1
                x_array.append(float(x)) 
                angle_array.append(float(found_angle))
                #BOTH LEFT AND RIGHT EXERCISES 
                if desired_direction == 'both':
                    print("2") #for testing
                    #exercise done 
                    if int_desired_repeat == angle_met: 
                        end_exercise = True
                        print("3") #for testing
                    #exercise in process 
                    if int_desired_repeat > angle_met:
                        #IN ANY CASE, CAUTION ANGLE
                        print("4") #for testing
                        if found_angle > int_desired_max_angle: #hand above desired range
                            print("5") #for testing
                            #if user has not already been informed that they are in caution range 
                            if please_zonecaution == False:
                                print("6") #for testing
                                print("you have moved your hand outside of the desired angle range. please move it back")
                                system("say you have moved your hand outside of the desired angle range. please move it back") 
                                #change other aid variable back
                                please_zonecaution = True
                                #initialize aid commands
                                please_posdirection = False
                                please_negdirection = False
                                please_zero = False
                        #USER NEEDS TO ZERO 
                        if need_zero == True:
                            print("7") #for testing
                            print("please move your hand to the starting position")
                            please_zero = True #said
                            if found_angle == 0: #starting at 0
                                print("8") #for testing
                                print("hand in starting position!")
                                system("say hand in starting position!") 
                                need_zero = False
                                please_zero = True
                                #initialize aid commands
                                please_posdirection = False
                                please_negdirection = False
                                please_zonecaution = False
                        #USER NEEDS TO REP (HAS ZEROED)
                        if need_zero == True:
                            print("7")  # for testing
                            print("please move your hand to the starting position")
                            system("say please move your hand to the starting position")
                            please_zero = True  # said

                            if found_angle == 0:  # starting at 0
                                print("8")  # for testing
                                print("hand in starting position!")
                                system("say hand in starting position!")
                                need_zero = False

                            # initialize aid commands
                            please_posdirection = False
                            please_negdirection = False
                            please_zonecaution = False

# USER NEEDS TO REP (HAS ZEROED)
                        if need_zero == False:
                            print("9")  # for testing
                            if pos_halfrep == 0 or pos_halfrep == neg_halfrep:
                                # not hit angle and not told yet
                                if found_angle != int_desired_max_angle and please_posdirection == False:
                                    print("10")  # for testing
                                    print("please move your arm in the desired positive direction to desired angle")
                                    system("say please move your arm in the desired positive direction to desired angle")
                                    please_posdirection = True
                                    # initialize aid commands
                                    please_negdirection = False
                                    please_zero = False
                                    please_zonecaution = False

                                # hit angle
                                if found_angle == int_desired_max_angle:
                                    print("correct angle met!")
                                    print("please move your hand to the desired angle on the other side to complete the repetition.")
                                    system("say please move your hand to the desired angle on the other side to complete the repetition.")
                                    pos_halfrep = pos_halfrep + 1  # add count
                                    need_zero = True  # user needs to zero

                                    # initialize aid commands
                                    please_posdirection = False
                                    please_negdirection = False
                                    please_zero = False
                                    please_zonecaution = False
                        """if need_zero == False:
                            print("9") #for testing
                            if pos_halfrep == 0 or pos_halfrep == neg_halfrep: 
                                #not hit angle and not told yet 
                                if final_angle != int_desired_max_angle and please_posdirection == False:
                                    print("10") #for testing
                                    print("please move your arm in the desired positive direction to desired angle")
                                    system("say please move your arm in the desired positive direction to desired angle")
                                    please_posdirection = True
                                    #initialize aid commands
                                    please_negdirection = False
                                    please_zero = False
                                    please_zonecaution = False
                                #hit angle 
                                if final_angle == int_desired_max_angle:
                                    print("correct angle met!")
                                    print("please move your hand to desired angle on the other side to complete the repetition.")
                                    system("say please move your hand to desired angle on the other side to complete the repetition.")
                                    pos_halfrep = pos_halfrep + 1 #add count
                                    need_zero = True #user needs to zero
                                    please_posdirection = True
                                    #initialize aid commands
                                    please_negdirection = False
                                    please_zero = False
                                    please_zonecaution = False"""
                        #USER HAS DONE HALF A REP -- NEEDS TO DO NEG SIDE 
                        if pos_halfrep > neg_halfrep and need_zero == False:
                            print("12") #for testing
                            if please_negdirection == False:
                                print("13") #for testing
                                print("please move your arm in the desired negative direction")
                                system("say please move your arm in the desired negative direction") 
                                please_negdirection = True
                                #initialize aid commands
                                please_posdirection = False
                                please_zero = False
                                please_zonecaution = False
                            if found_angle == neg_int_desired_angle:
                                print("14") #for testing
                                neg_halfrep = neg_halfrep + 1
                                angle_met = angle_met + 1
                                need_zero = True
                    #ONLY LEFT OR ONLY RIGHT EXERCISES
                    if desired_direction == 'left': 
                        desired_direction = desired_direction * (-1) #left desired angle is in opposite direction
                    if desired_direction == 'left' or desired_direction == 'right':
                        #exercise done 
                        if int_desired_repeat == angle_met: 
                            end_exercise = True 
                        #exercise in process 
                        if int_desired_repeat > angle_met:
                            #USER NEEDS TO REP (HAS ZEROED) 
                            if need_zero == False: 
                                if angle_met == 0: #first time doing rep
                                    #not hit angle and not told yet 
                                    if found_angle != int_desired_max_angle and please_direction == False: 
                                        print("please move your arm in the desired direction to desired angle")
                                        system("say please move your arm in the desired direction to desired angle")
                                        please_direction = True
                                        #initialize aid commands
                                        please_zonecaution = False
                                        please_zero = False 
                                    #hit angle 
                                    if int_desired_max_angle + 2 > found_angle > int_desired_max_angle - 2:
                                        print("correct angle met!")
                                        system("say correct angle met!")
                                        angle_met = angle_met + 1 
                                        need_zero = True #user needs to zero
                                        #initialize aid commands
                                        please_zonecaution = False
                                        please_zero = False
                                        please_direction = False
                            #IN ANY CASE, CAUTION ANGLE
                            if found_angle > int_desired_max_angle: #hand above desired range 
                                #if user has not already been informed that they are in caution range 
                                if please_zonecaution == False: 
                                    print("you have moved your hand outside of the desired angle range. please move it back")
                                    system("say you have moved your hand outside of the desired angle range. please move it back")
                                    #change other aid variable back
                                    please_zonecaution = True
                                    #initialize aid commands
                                    please_direction = False
                                    please_zero = False
                            #USER NEEDS TO ZERO 
                            if need_zero == True: 
                                print("please move your hand to the starting position")
                                system("say please move your hand to the starting position") 
                                please_zero = True #said
                                #initialize aid commands
                                please_zonecaution = False
                                please_direction = False 
                                if found_angle == 0: #starting at 0
                                    print("hand in starting position!")
                                    system("say please move your hand to the starting position") 
                                    need_zero = False 
            #POST EXERCISE 
            if keypressed == 27 or end_exercise == True:
                print("15") #for testing
                cap.release() #release webcam 
                cv2.destroyAllWindows() #close windows 
                #GRAPHING ARRAYS 
                plt.scatter(x_array, angle_array)
                plt.title("Program Angle Measurements")
                plt.ylabel("Angle Measurement") 
                plt.show()
                #SAVING USER DATA 
                save_userdata = input("would you like to save the data from this exercise? type 's' to save or 'n' to not save: ")
                if save_userdata == 's': 
                    #save angle data array into fields and rows for csv file
                    fields = [['program intervals','joint angle']]
                    rows = [[x_array, angle_array]]
                    #name save data file 
                    userdata_filename_noending = input("please type a name for this data file: ") #ask user for filename
                    userdata_filename = userdata_filename_noending + ".csv"
                    #save data as csv file 
                    with open(color_filename, 'w') as csvfile:  
                        #creating a csv writer object  
                        csvwriter = csv.writer(csvfile)
                        #writing the data rows  
                        csvwriter.writerows(fields)
                        csvwriter.writerows(rows)
                    print("colors have been sucessfully saved!")
                    #save angle data graph
                    plt.savefig("Angle Measurement ")
                    print("your joint angle data has been saved") 
                    print("thank you for using this physical therapy program!")
                if save_userdata == 'n': 
                    print("your joint angle data has not been saved")
                    print("thank you for using this physical therapy program!") 
