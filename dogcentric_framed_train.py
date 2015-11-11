import cv2
import glob
import numpy as np
from sklearn import svm
from sklearn.externals import joblib
import dogcentric_flow


# feature_vec = []
train_ave = []
train_label = []


# path = "/Users/soledad/Box Sync/Fall 15/Research/Data Sets/dogcentric/Ringo"
path = "/Users/soledad/Box Sync/Fall 15/Research/Data Sets/dogcentric"

dogs = glob.glob(path+ "/*")


for d in dogs:
	# print d

	folder = glob.glob(d + "/*")

	for fol in folder:

		files = glob.glob(fol+ "/*.avi")
		end = int(len(files)/2)
		
		for f in files:
			print f
		
		    	ave,label = dogcentric_flow.flow(f, 40)
			if (label) != 0 :
		    		train_ave.append(ave)
		    		train_label.append(label)
		
		    	feature_vec = []
		    	label = []
	
# prepare data for svm
t = np.asarray(train_ave)
t = t.reshape(len(train_ave),1)

clf = svm.SVC(kernel='rbf', gamma=0.001, C=1000.0)
clf.fit(t, train_label)

# save svm class
joblib.dump(clf, path + '/' + 'dogcentric.pkl') 