import cv2
import numpy as np
import time
import math
import optparse
import pickle
import glob
import sys

fr=open('frames.data','wb')
fl=open('flow.data','wb')

class OpticalFlowCalculator:
    '''
    A class for optical flow calculations using OpenCV
    '''
    
    def __init__(self, frame_width, frame_height, scaledown=1,
            		move_step=16, window_name=None, flow_color_rgb=(0,255,0)):
        '''
        Creates an OpticalFlow object for images with specified width and height.

        Optional inputs are:

          move_step           - step size in pixels for sampling the flow image
          window_name       - window name for display
          flow_color_rgb    - color for displaying flow
        '''

        self.move_step = move_step
        self.mv_color_bgr = (flow_color_rgb[2], flow_color_rgb[1], flow_color_rgb[0])
        self.window_name = window_name
        self.size = (int(frame_width/scaledown), int(frame_height/scaledown))
        self.prev_gray = None
        self.prev_time = None

    def processFrame(self, frame, frameno=0, timestep=1):
        '''
        Processes one image frame, returning summed X,Y flow and frame.

        Optional inputs are:
          timestep - time step in seconds for returning flow in meters per second
        '''
	frame2 = cv2.resize(frame, self.size)
        gray = cv2.cvtColor(frame2, cv2.cv.CV_BGR2GRAY)

        xsum, ysum = 0,0

        xvel, yvel = 0,0
        
        if self.prev_gray != None:

            flow = cv2.calcOpticalFlowFarneback(self.prev_gray, gray, pyr_scale=0.5, levels=5, winsize=13, iterations=10, poly_n=5, poly_sigma=1.1, flags=0) 
	    #resize and dump frame data
	    gray_rsz=cv2.resize(gray,(32,32))
	    pickle.dump(gray_rsz,fr)

	    #resize and dump flow data
	    flow_rsz=cv2.resize(flow,(32,32))
	    pickle.dump(flow_rsz.flatten(),fl)
	    
	    for y in range(0, flow.shape[0], self.move_step):
		flowrow=[]
                for x in range(0, flow.shape[1], self.move_step):
                    fx, fy = flow[y, x]
                    xsum += fx
                    ysum += fy
		    cv2.line(frame2, (x,y), (int(x+fx),int(y+fy)), self.mv_color_bgr)
                    cv2.circle(frame2, (x,y), 1, self.mv_color_bgr, -1)
	

            # Default to system time if no timestep
            curr_time = time.time()
            if not timestep:
                timestep = (curr_time - self.prev_time) if self.prev_time else 1
            self.prev_time = curr_time

        self.prev_gray = gray

        if self.window_name:
            cv2.imshow(self.window_name, frame2)
            if cv2.waitKey(1) & 0x000000FF== 27: # ESC
                return None
        
        return  frame2

if __name__=="__main__":
	
    ''' Takes in path as an cmdline argument, must end with a slash'''

    videos=glob.glob(sys.argv[1]+"*.avi")
    for video in videos:
	    cap = cv2.VideoCapture(video)

	    width    = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
	    height   = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))

	    scaledown = 1

	    movestep = 16

	    flow = OpticalFlowCalculator(width, height, window_name='Optical Flow', scaledown=scaledown, move_step=movestep) 

	    start_sec = time.time()
	    count = 0
	    while True:

		success, frame = cap.read()

		count += 1
		    
		if not success:
		    break

		result = flow.processFrame(frame,count)

		if result==None:
		    break

	    elapsed_sec = time.time() - start_sec

	    print('%dx%d image: %d frames in %3.3f sec = %3.3f frames / sec' % 
		     (width/scaledown, height/scaledown, count, elapsed_sec, count/elapsed_sec))
    fr.close()
    fl.close()
