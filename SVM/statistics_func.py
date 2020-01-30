

def get_Score(Conf_Matrix):
	total=0.0
	True_val=0.0
	for i in range(len(Conf_Matrix)):
		for j in range(len(Conf_Matrix)):
			if(i==j):
				True_val=True_val+Conf_Matrix[i][j]
			total=total+Conf_Matrix[i][j]
	Accuracy=True_val/total
	Recall=[]
	Precision=[]
	for i in range(len(Conf_Matrix)):
		Sum=0.0
		for j in range(len(Conf_Matrix)):
			Sum=Sum+Conf_Matrix[i][j]
		Recall.append(Conf_Matrix[i][i]/Sum)
	for i in range(len(Conf_Matrix)):
		Sum=0.0
		for j in range(len(Conf_Matrix)):
			Sum=Sum+Conf_Matrix[j][i]
		if(Sum==0):
			Precision.append(0)
		else:
			Precision.append(Conf_Matrix[i][i]/Sum)
	print ("Accuracy of Classifier:- ",Accuracy)
	for i in range(len(Conf_Matrix)):
		print("Precision of Class",(i+1),":-",Precision[i])
	for i in range(len(Conf_Matrix)):
		print("Recall of Class",(i+1),":-",Recall[i])
	Sum=0.0
	for i in range(len(Conf_Matrix)):
		if ((Recall[i]+Precision[i]) == 0):
			print("F Measure of Class",(i+1),":- 0")
		else:
			print("F Measure of Class",(i+1),":-",(2*Recall[i]*Precision[i])/(Recall[i]+Precision[i]))
			Sum=Sum+(Recall[i]*Precision[i])/(Recall[i]+Precision[i])
	print("Mean Precision :-",(sum(Precision)/len(Conf_Matrix)))	
	print("Mean Recall :-",(sum(Recall)/len(Conf_Matrix)))
	print("Mean F Measure :-",2*(Sum)/len(Conf_Matrix))