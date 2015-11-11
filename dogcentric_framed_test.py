import cv2
import glob
import numpy as np
from sklearn import svm
from sklearn.externals import joblib	
import dogcentric_flow


path = "/Users/soledad/Box Sync/Fall 15/Research/Data Sets/dogcentric"

# reload svm classif to predict other video
clf = joblib.load(path + '/' + 'dogcentric.pkl')
results = []

dogs = glob.glob(path+ "/*")


for d in dogs:

	folder = glob.glob(d + "/*")

	for fol in folder:

		files = glob.glob(fol+ "/*.avi")
		beg = int(len(files)/2)+1
		end = int(len(files))
		
		for f in files[beg:end]:
			
			print f
			test_ave, test_label = dogcentric_flow.calc_flow(f, 40)
			
			t = np.array(test_ave)


			predicted = clf.predict(t) 
			print test_label, predicted
			vec = [ test_label, predicted]
			results.append(vec)

# print results