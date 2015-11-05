from sklearn import svm
import numpy as np

gram=[]
clf = svm.SVC(kernel=precomputed)
clf.fit(gram,Y)
