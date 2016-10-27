from sklearn.cluster import KMeans
import numpy as np
import os
import sys	
import glob
import random
import cv2

optflSet=0

flows = []

def computeMeans(X=optflSet, num=40):
	km = KMeans(n_clusters=num, random_state=0).fit(X)
	c=km.cluster_centers_
	return km
	
def predict(km,X):
	return km.predict(X)



def pickRandomOpticalFlowVectorsFromFrame(frame, next_frame):
	flow = 0
	flow = cv2.calcOpticalFlowFarneback(frame, next_frame, flow, pyr_scale=0.5, levels=5, winsize=13, iterations=10, poly_n=5, poly_sigma=1.1, flags=0) 
	no_flows = len(flow)
	num_to_select = int(no_flows*(0.1)) + 1;
	l = random.sample(flow, num_to_select)
	for k in l:
		for item in k:
			flows.append(item)
	pass

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

	num_to_select = int(no_of_frames*(0.1)) + 1
	
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
	num_to_select = int(len(l)*(0.1))+1
	k = random.sample(l,num_to_select)
	return k

# path must end with /
def run_kmeans(path_to_videos):
	videos = pickRandomVideos(path_to_videos)
	for video in videos:
		pickRandomFramesFromVideo(path_to_videos+ video)
	X = np.array(flows)
	km = computeMeans(X)
	return km 
