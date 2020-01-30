import numpy as np
import statistics_func as sf
import os
import matplotlib.pyplot as plt
from pathlib import Path

def get_y(W,x):
	return (-1*W[0]-x*W[1])/(W[2]*1.0)
def most_common (lst):
    return max(((item, lst.count(item)) for item in set(lst)), key=lambda a: a[1])[0]
paths=["Data/Data1/"]
Train=[]
Test=[]
plot_Train=[]
for i in paths:
	arr=os.listdir(i)
	for j in arr:
		a,b=sf.get_data(i+j)
		for k in range(len(a)):
			a[k]=[1]+a[k]
		for k in range(len(b)):
			b[k]=[1]+b[k]
		Train.append(a)
		Test.append(b)

plot_Train=[]
for i in paths:
	arr=os.listdir(i)
	for j in arr:
		a,b=sf.get_data(i+j)
		plot_Train.append(a)
alpha=0.005
W12=np.array([1,-1,1])
# 1 and 2
Dm=[[],[]]
for i in Train[0]: 
	if np.array(i).dot(W12)<=0:
		Dm[0].append(i)
for i in Train[1]:
	if np.array(i).dot(W12)>=0:
		Dm[1].append(i)
for i in Dm[0]:
	W12=np.add(alpha*np.array(i),W12)
for i in Dm[1]:
	W12=np.add(-1*alpha*np.array(i),W12)
# exit()
while len(Dm[0])>0 or len(Dm[1])>0:
	Dm=[[],[]]
	for i in Train[0]:
		if np.array(i).dot(W12)<=0:
			Dm[0].append(i)
	for i in Train[1]:
		if np.array(i).dot(W12)>=0:
			Dm[1].append(i)
	for i in Dm[0]:
		W12+=alpha*np.array(i)
	for i in Dm[1]:
		W12-=alpha*np.array(i)
a=[]
b=[]
x=-15.0
while(x<20):
	y=-20.0
	while(y<20):
		if (W12[0])+(W12[1]*x)+(W12[2]*y)>=0:
			a.append([x,y])
		else:
			b.append([x,y])
		y=y+0.1
	x=x+0.1
sf.plot(a,'c.',"Class1")
sf.plot(b,'y.',"Class2")
sf.plot(plot_Train[0],"r.")
sf.plot(plot_Train[1],"b.")
folder = Path("Results/")
plt.savefig(folder / 'class12_Plot.png', dpi=300, bbox_inches='tight')
# plt.plot([-20,-10,10,20],[get_y(W12,-20),get_y(W12,-10),get_y(W12,10),get_y(W12,20)])
plt.show()



W23=np.array([1,-1,1])
# 1 and 2
Dm=[[],[]]
for i in Train[1]: 
	if np.array(i).dot(W23)<=0:
		Dm[0].append(i)
for i in Train[2]:
	if np.array(i).dot(W23)>=0:
		Dm[1].append(i)
for i in Dm[0]:
	W23=np.add(alpha*np.array(i),W23)
for i in Dm[1]:
	W23=np.add(-1*alpha*np.array(i),W23)
# exit()
while len(Dm[0])>0 or len(Dm[1])>0:
	Dm=[[],[]]
	for i in Train[1]:
		if np.array(i).dot(W23)<=0:
			Dm[0].append(i)
	for i in Train[2]:
		if np.array(i).dot(W23)>=0:
			Dm[1].append(i)
	for i in Dm[0]:
		W23+=alpha*np.array(i)
	for i in Dm[1]:
		W23-=alpha*np.array(i)

a=[]
b=[]
x=-15.0
while(x<30):
	y=-20.0
	while(y<20):
		if (W23[0])+(W23[1]*x)+(W23[2]*y)>=0:
			a.append([x,y])
		else:
			b.append([x,y])
		y=y+0.1
	x=x+0.1
sf.plot(a,'c.',"Class2")
sf.plot(b,'y.',"Class3")
sf.plot(plot_Train[1],"r.")
sf.plot(plot_Train[2],"b.")
folder = Path("Results/")
plt.savefig(folder / 'class23_Plot.png', dpi=300, bbox_inches='tight')
# plt.plot([-20,-10,10,20],[get_y(W23,-20),get_y(W23,-10),get_y(W23,10),get_y(W23,20)])
plt.show()


W13=np.array([1,-1,1])
# 1 and 2
Dm=[[],[]]
for i in Train[0]: 
	if np.array(i).dot(W13)<=0:
		Dm[0].append(i)
for i in Train[2]:
	if np.array(i).dot(W13)>=0:
		Dm[1].append(i)
for i in Dm[0]:
	W13=np.add(alpha*np.array(i),W13)
for i in Dm[1]:
	W13=np.add(-1*alpha*np.array(i),W13)
# exit()
while len(Dm[0])>0 or len(Dm[1])>0:
	Dm=[[],[]]
	for i in Train[0]:
		if np.array(i).dot(W13)<=0:
			Dm[0].append(i)
	for i in Train[2]:
		if np.array(i).dot(W13)>=0:
			Dm[1].append(i)
	for i in Dm[0]:
		W13+=alpha*np.array(i)
	for i in Dm[1]:
		W13-=alpha*np.array(i)
		
a=[]
b=[]
x=-15.0
while(x<30):
	y=-20.0
	while(y<20):
		if (W13[0])+(W13[1]*x)+(W13[2]*y)>=0:
			a.append([x,y])
		else:
			b.append([x,y])
		y=y+0.1
	x=x+0.1
sf.plot(a,'c.',"Class1")
sf.plot(b,'y.',"Class3")
sf.plot(plot_Train[0],"r.")
sf.plot(plot_Train[2],"b.")
folder = Path("Results/")
plt.savefig(folder / 'class13_Plot.png', dpi=300, bbox_inches='tight')
# plt.plot([-20,-10,10,20],[get_y(W13,-20),get_y(W13,-10),get_y(W13,10),get_y(W13,20)])
plt.show()
ans=[[],[],[]]
x=-15.0
while(x<30):
	y=-20.0
	while(y<20):
		a=[]
		if(np.array([1,x,y]).dot(W12)>0):
			a.append(0)
		else:
			a.append(1)
		if(np.array([1,x,y]).dot(W23)>0):
			a.append(1)
		else:
			a.append(2)
		if(np.array([1,x,y]).dot(W13)>0):
			a.append(0)
		else:
			a.append(2)
		ans[most_common(a)].append([x,y])
		y=y+0.1
	x=x+0.1
sf.plot(ans[0],'c.',"Class1")
sf.plot(ans[1],'y.',"Class2")
sf.plot(ans[2],'g.',"Class3")
sf.plot(plot_Train[0],'r.')
sf.plot(plot_Train[1],'b.')
sf.plot(plot_Train[2],'m.')
folder = Path("Results/")
plt.savefig(folder / 'allclasses_Plot.png', dpi=300, bbox_inches='tight')
plt.show()
conf=[[0,0,0],[0,0,0],[0,0,0]]
for i in range(len(Test)):
	for j in range(len(Test[i])):
		a=[]
		if(np.array(Test[i][j]).dot(W12)>0):
			a.append(0)
		else:
			a.append(1)
		if(np.array(Test[i][j]).dot(W23)>0):
			a.append(1)
		else:
			a.append(2)
		if(np.array(Test[i][j]).dot(W13)>0):
			a.append(0)
		else:
			a.append(2)
		conf[i][most_common(a)]+=1
print (conf)
sf.get_Score(conf)