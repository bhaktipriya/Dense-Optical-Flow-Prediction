from sklearn.cluster import KMeans
import numpy as np

optflSet=0
kmeans=0

def computeMeans(X=optflSet, num=40):
	kmeans = KMeans(n_clusters=num, random_state=0).fit(X)
	print "======================================"
	print "Cluster centres are"
	c=kmeans.cluster_centers_
	for i in xrange(num):
		print "(", c[i][0], ",",c[i][1],")" 

	
def predict(X):
	return kmeans.predict(X)



def pickRandomOpticalFlowVectorsFromFrame(frame):
	pass
	#append to optflSet

def pickRandomFramesFromVideo(video):
	pass

def pickRandomVideos(datasetDir):
	pass


computeMeans(X = np.array([[1, 2], [1, 4], [1, 0],[4, 2], [4, 4], [4, 0]]), num=2)


	
