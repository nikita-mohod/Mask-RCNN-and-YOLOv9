# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 15:52:33 2024

@author: ASUS
"""



# Read all the extracted frames from the folder and detect the relevant frames while discarding others
try:
      
    # creating a folder named data
    if not os.path.exists('FRAMESNEW_yolo'):
        os.makedirs('FRAMESNEW_yolo5')
  
# if not created then raise error
except OSError:
    print ('Error: Creating directory for Compressed Video -  yolo5')

FLAG = np.zeros((newframe,)) # This array will tell which frame is detected as relevant frame

# path
path = 'FRAMES/'

pathnew = 'FRAMESNEW_yolo/'

count = 0

start = time.time()

for xx in range(0, newframe):
    
    # Clear Console
    print("\033[H\033[J") # Just to clear all console outputs
    
    print('Scanning with Yolo5 frame - %i' %xx)
    
    framename = 'frame' + str(xx) + '.jpg'
     
    # Using cv2.imread() method
    img = cv2.imread('FRAMES/' + framename)
  

    results = model([img])
    
    labels, cord = results.xyxyn[0][:, -1].to('cpu').numpy(), results.xyxyn[0][:, :-1].to('cpu').numpy()
    
    # results.print()
    
    # results.xyxy[0]
    
    # print(results.pandas().xyxy[0])

    n = len(labels)
    
    # x_shape, y_shape = img.shape[1], img.shape[0]
    
    if n>=1:
        pathname = pathnew + framename
        
        # writing the extracted images
        cv2.imwrite(pathname, img)
        
        FLAG[xx] = 1
    else:
        count = count + 1
        FLAG[xx] = 0
        continue
    
end = time.time()
# _____________________________________________________________________________
height, width, layers = img.shape
size = (width, height)
    
    
# Creating Video files for both
out = cv2.VideoWriter('Compressed_yolo.avi',cv2.VideoWriter_fourcc(*'DIVX'),fps,size)

frame_written = 0

for m in range(0, newframe):
    
    # Clear Console
    print("\033[H\033[J") # Just to clear all console outputs
    
    print('Creating Video of RELEVANT frames for Yolo...')
    
    pp = pathnew + 'frame' + str(m) + '.jpg'
    
    if os.path.isfile(pp):
        
        print('Writing frame - %i' %m)
        # read image
        one_image = cv2.imread(pp)
        out.write(one_image)
        
        frame_written = frame_written + 1
        
out.release()
print('Number of frames written to video in Yolo5 case - %i' %frame_written)


print('Total number of frames in given Video - ', str(newframe*fps))
print('Number of frames considered - ', str(newframe))

print('Number of REVALENT frames with Yolo Network- ', str(newframe-count))
print('Number of IRREVALENT frames with Yolo Network- ', str(count))

print('Time required for deteting all frames by yolo Network - %i' %(end-start))

# Delete the Relevant frames folders

# import sys
# import shutil

# # Get directory name - Yolo5
# mydir = pathnew

# # Try to remove the tree; if it fails, throw an error using try...except.
# try:
#     shutil.rmtree(mydir)
# except OSError as e:
#     print("Error: %s - %s." % (e.filename, e.strerror))




