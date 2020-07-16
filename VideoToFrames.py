# Importing all necessary libraries 
import cv2 
import os 

# Read the video from specified path 
for Num in range(1,26):
    if Num < 10:
        video_path = "/store/travail/CATARACTS/Videos/train0" +str(Num) + ".mp4"
    else:
        video_path = "/store/travail/CATARACTS/Videos/train" +str(Num) + ".mp4"        
   
    cam = cv2.VideoCapture(video_path) 
    
    #directory = "C:\\Users\lucas\OneDrive\Documents\Professionnel\StageMontreal2020\Python\RetinalSurgeryRecognition\src\dataset"
    #os.chdir(directory)  
    
    VideosTest = [12, 19, 22, 24, 25]
    if Num in VideosTest:
        Folder = '/store/travail/CATARACTS/Testing_Data/'
    else:
        Folder = '/store/travail/CATARACTS/Training_Data/'    
    
    try: 
    	
    	# creating a folder named data        
        if not os.path.exists(Folder + 'data' + str(Num)): 
            os.makedirs(Folder + 'data' + str(Num)) 
    
    # if not created then raise error 
    except OSError: 
    	print ('Error: Creating directory of data') 
    
    # frame 
    currentframe = 0
    
    while(True): 
    	
    	# reading from frame 
    	ret,frame = cam.read() 
    
    	if ret: 
    		# if video is still left continue creating images 
    		name = Folder + 'data' + str(Num) + '/' + str(currentframe) + '.jpg'
    		print ('Creating ' + name) 
    
    		# writing the extracted images 
    		cv2.imwrite(name, frame) 
    
    		# increasing counter so that it will 
    		# show how many frames are created 
    		currentframe += 1
    	else: 
    		break
    
    # Release all space and windows once done 
    cam.release() 
    cv2.destroyAllWindows()
    


