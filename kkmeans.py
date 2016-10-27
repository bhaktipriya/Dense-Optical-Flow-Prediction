from sklearn.cluster import KMeans
import numpy as np
import os
import sys	
import glob
import random
import cv2

optflSet=0
kmeans=0

flows = []

def computeMeans(dirname):
	videos = pickRandomVideos(dirname)
	for video in videos:
		pickRandomFramesFromVideo(dirname+"/"+ video)
	
	X = np.array(flows)
	kmeans = KMeans(n_clusters=40, random_state=0).fit(X)
	print kmeans.cluster_centers_
	print "Computed K means successfully!! :D"
	return kmeans

def pickRandomOpticalFlowVectorsFromFrame(frame, next_frame):
	flow = 0
	flow = cv2.calcOpticalFlowFarneback(frame, next_frame, flow, pyr_scale=0.5, levels=5, winsize=13, iterations=10, poly_n=5, poly_sigma=1.1, flags=0) 
	no_flows = len(flow)
	num_to_select = int(no_flows*(10/100)) + 1;
#	flows.append( random.sample(flow, num_to_select) )
	l = random.sample(flow, num_to_select)
	for k in l:
		for item in k:
			flows.append(item)
	pass
	#append to optflSet

def pickRandomFramesFromVideo(video):
	cap = cv2.VideoCapture(video)

	width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

	no_of_frames = 0

	while True:
		ret, frame = cap.read()
		if ret == False:
			break
		no_of_frames+=1

	num_to_select = int(no_of_frames*(10/100)) + 1
	
	l = random.sample(range(no_of_frames-1),num_to_select)
	for frame_no in l:
		cap.set(1,frame_no)
		ret, frame = cap.read()
		ret, next_frame = cap.read()
		gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray2 = cv2.cvtColor(next_frame, cv2.COLOR_BGR2GRAY)
		pickRandomOpticalFlowVectorsFromFrame(gray1, gray2)



	pass

def pickRandomVideos(datasetDir):
	l = os.listdir(datasetDir)
	num_to_select = int(len(l)*(10/100))+1
	k = random.sample(l,num_to_select)
	return k



#computeMeans(X = np.array([[1, 2], [1, 4], [1, 0],[4, 2], [4, 4], [4, 0]]), num=2)


	
