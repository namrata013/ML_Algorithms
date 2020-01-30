import numpy as np
import matplotlib.pyplot as plt 
import statistics_func as sf
from sklearn.svm import SVC
import os

def most_common (lst):
    return max(((item, lst.count(item)) for item in set(lst)), key=lambda a: a[1])[0]
# paths=["Data/Data2/"]
# Train=[]
# Test=[]
# for i in paths:
# 	arr=os.listdir(i)
# 	for j in arr:
# 		a,b=sf.get_data(i+j)
# 		Train.append(a)
# 		Test.append(b)

paths=["Data/Train/BOVW/bovw_CandyStore/","Data/Train/BOVW/bovw_FootBallStadium/","Data/Train/BOVW/bovw_ForestBroadLeaf/"]
Train=[]
for i in paths:
	arr=os.listdir(i)
	class_d=[]
	for j in arr:
		temp=np.load(i+j)
		class_d.append(temp)
	Train.append(class_d)	



paths=["Data/Test/BOVW/bovw_CandyStore/","Data/Test/BOVW/bovw_FootBallStadium/","Data/Test/BOVW/bovw_ForestBroadLeaf/"]
Test=[]
for i in paths:
	arr=os.listdir(i)
	class_d=[]
	for j in arr:
		temp=np.load(i+j)
		class_d.append(temp)
	Test.append(class_d)


kernel='linear'
X1=[]
X2=[]
X3=[]
Y1=[]
Y2=[]
Y3=[]
for i in range(len(Train)):
	for j in range(len(Train[i])):
		if i==0:
			X1.append(Train[i][j])
			Y1.append(i)
		elif i==1:
			X2.append(Train[i][j])
			Y2.append(i)
		else:
			X3.append(Train[i][j])
			Y3.append(i)
X12=[]
for i in range(len(X1)):
	X12.append(X1[i])
for i in range(len(X2)):
	X12.append(X2[i])
Y12=[]
for i in range(len(Y1)):
	Y12.append(Y1[i])
for i in range(len(Y2)):
	Y12.append(Y2[i])
X23=[]
for i in range(len(X2)):
	X23.append(X2[i])
for i in range(len(X3)):
	X23.append(X3[i])
Y23=[]
for i in range(len(Y2)):
	Y23.append(Y2[i])
for i in range(len(Y3)):
	Y23.append(Y3[i])
X13=[]
for i in range(len(X1)):
	X13.append(X1[i])
for i in range(len(X3)):
	X13.append(X3[i])
Y13=[]
for i in range(len(Y1)):
	Y13.append(Y1[i])
for i in range(len(Y3)):
	Y13.append(Y3[i])
print ("<Asd></Asd>")
classifier12=SVC(kernel='rbf')
classifier12.fit(X12,Y12)
classifier23=SVC(kernel='rbf')
classifier23.fit(X23,Y23)
classifier13=SVC(kernel='rbf')
classifier13.fit(X13,Y13)
conf=[[0,0,0],[0,0,0],[0,0,0]]
for i in range(len(Test)):
	for j in range(len(Test[i])):
		a=[]
		a.append(classifier12.predict([Test[i][j]])[0])
		a.append(classifier23.predict([Test[i][j]])[0])
		a.append(classifier13.predict([Test[i][j]])[0])	
		conf[i][most_common(a)]+=1
print ("matrix {",conf[0][0],"#",conf[0][1],'#',conf[0][2],'##',conf[1][0],"#",conf[1][1],'#',conf[1][2],'##',conf[2][0],"#",conf[2][1],'#',conf[2][2],"}")
sf.get_Score(conf)
# ans=[[],[],[]]
# x=-3.0
# while(x<3):
# 	y=-3.0
# 	while(y<3):
# 		a=[]
# 		a.append(classifier12.predict([[x,y]])[0])
# 		a.append(classifier23.predict([[x,y]])[0])
# 		a.append(classifier13.predict([[x,y]])[0])
# 		ans[most_common(a)].append([x,y])
# 		y=y+0.03
# 	x=x+0.03
# sf.plot(ans[0],'c.',"Class1")
# sf.plot(ans[1],'y.',"Class2")
# sf.plot(ans[2],'g.',"Class3")
# sf.plot(Train[0],'r.')
# sf.plot(Train[1],'b.')
# sf.plot(Train[2],'m.')
# plt.show()