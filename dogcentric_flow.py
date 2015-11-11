import cv2
import numpy as np



feature_vec = []
train_ave = []
train_label = []

def calc_flow(f, samples):
	cap = cv2.VideoCapture(f)
	
	length = int(cap.get(7))
	step = int(length/samples)

	ave = 0
    	label = 0

	if step != 0:
		index = range(0,step*samples,step)
		
		ret, frame1 = cap.read()
		prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
	
		while True:
			for i in index:
				cap.set(1, i)
				ret, frame2 = cap.read()
		
				if not ret: break
		
		    		nextt = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
		
		    		flow = cv2.calcOpticalFlowFarneback(prvs,nextt, None, 0.5, 3, 15, 3, 5, 1.2, 0)
				
				feature_vec.append(np.average(flow))
				#feature_vec.append(low)
		
		    		prvs = nextt
		    	break
	
		if 'Car' in f: label = 9
		if 'Pet' in f: label = 1
    		if 'LookRight' in f: label = 2
    		if 'LookLeft' in f: label = 3
    		if 'Sniff' in f: label = 4
    		if 'Walk' in f: label = 5
    		if 'Feed' in f: label = 6
    		if 'Shake' in f: label = 7
    		if 'Drink' in f: label = 8
	
    		# average of all the opt flows of a video
    		ave = reduce(lambda x, y: x + y, feature_vec) / len(feature_vec)
    		print ave



    	return ave, label

