from config.settings import *
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_recall_fscore_support
from itertools import combinations
import csv 
import re 
import numpy as np
import argparse
import os 
import numpy as np

def meu(reporter_node, reporter_node_value,data):
	#without the cpt
	no_records =data.shape[0] 
	meu =[] #node, desired_state, reward
	
	for i in range(data.shape[1]):
		mask =[reporter_node, i]
		sub_data =data[:, mask]
		prob_0 = float(len(np.where(data[:, i] ==0)[0]))/no_records#where ith node is 0
		prob_1 = float(len(np.where(data[:, i]==1)[0]))/no_records
		#the definition provided is different than conventions meu as the output is fixed
		#but the path is to be determined
		reward_0 =float(len(np.where(sub_data ==[reporter_node_value, 0])[0]))/no_records		
		reward_1 =float(len(np.where(sub_data ==[reporter_node_value, 1])[0]))/no_records		
		best  =prob_0 * reward_0
		if prob_1 *reward_1 > best :
			best =prob_1 *reward_1
			meu.append([i, 1, best])
		else:
			meu.append([i, 0, best])	
	return meu
def otsu_thresholding(col):
	min_value =int(round(np.amin(col)))
	max_value =int(round(np.amax(col)))
	hist =np.histogram(col, np.arange(min_value, max_value+1, dtype='int'))#all integers b/w min and max are possible thresholds
	total_no =col.size
	sum_total, sum_0, sum_1, w_0, w_1, mean_0, mean_1 =0, 0, 0, 0, 0, 0,0
	current_max, threshold = 0, 0
	for i in range(len(hist[1])-1):
	
		sum_total += i * hist[0][i]
	
	for i in range(len(hist[1])-1):
		w_0 +=hist[0][i]
		w_1 =total_no -w_0
		
		sum_0 +=i * hist[0][i]
		sum_1 =sum_total -sum_0
		
		mean_0 =float(sum_0)/w_0
		if w_1 ==0:
			break
		mean_1 =float(sum_1)/w_1
		var_b =(w_0*w_1)*(mean_0-mean_1)**2
	
		if var_b > current_max:
			current_max =var_b
			threshold_idx =i
	threshold =hist[1][threshold_idx]
	print "threshold is " +str(threshold)
	col[col < threshold] =0
	col[col >= threshold] =1
	return col
def array_element_gene_name():
	"""dict for the gene and array element used in the venkat paper"""
	#use https://www.arabidopsis.org/tools/bulk/microarray/index.jsp to fully automate hardcoded at the moment
	gene_name_dict ={'255568_at': 'WRKY22', '267246_at': 'WRKY25', '266385_at': 'PR1',
						'255624_at':'MPK4', '249351_at': 'MKK3', '253646_at': 'MKK2',
						'256183_at': 'MKK4', '253993_at': 'MEK1', '248895_at': 'FLS2'}
	return gene_name_dict

def extract_gene_rows(ifile):
	"""parse out the rows for the genes are we are considering, also gets the names of the experiments"""
	gene_dict =array_element_gene_name()
	
	header =[]
	data =[]
	with open(ifile, 'r') as f:
		reader =csv.reader(f)
		count =0
		for row in reader:
			if count ==0:
				header.extend(row)
			else:
				for name in gene_dict.keys():#.keys() always returns items in the same order
					if row[0]  ==name:
						
						data.append(row)
			count +=1
		
		return header, [list(i) for i in zip(*data)]

def extract_series_matrix(ifile, new_name):
	"""from the txt file get the data matrix"""
	parse =False
	matrix =[]
	with open(ifile , 'r') as f :
		reader =csv.reader(f, delimiter ='\t')
		for row in reader:
			if row and row[0] =='!series_matrix_table_end':
				parse =False
			if parse ==False:
				pass
			else:
				matrix.append(row)
			if row and row[0] =='!series_matrix_table_begin':
				parse =True
			
	with open(new_name, 'w')as f:
		writer =csv.writer(f)
		for row in matrix:
			writer.writerow(row)
def test_train_split(x, y):
	skf = StratifiedKFold(n_splits=2, shuffle=True)
	for train_index, test_index in skf.split(x, y):
		X_train, X_test = x[train_index], x[test_index]
		Y_train, Y_test = y[train_index], y[test_index]
	return X_train, X_test, Y_train, Y_test 

def categorical(x):
	enc = OneHotEncoder()
	return enc.fit_transform(x)

def classify(x_train,y_train, x_test, labels):
	classifiers = [   KNeighborsClassifier(),    SVC(),    SVC(kernel='linear'),    DecisionTreeClassifier(),    RandomForestClassifier(),    AdaBoostClassifier(),        LogisticRegression()	]
	

	names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Decision Tree",
         "Random Forest", "AdaBoost",  "LogisticRegression" ]
	
	f1_scores=[]
	for name, clf in zip(names, classifiers):
		#print name
		
		clf.fit(X_train, Y_train)
		y_pred =clf.predict(X_test)
		#score=clf.score(X_test, labels)
		#print "score "+str(score)
		f1_scores.append(precision_recall_fscore_support(labels, y_pred, average=None, labels =[0,1])[2][1])#only printing f1
	return f1_scores, names
	#return scores, names
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--create_csv_files', action='store_true')
	parser.add_argument('--exp_files_folder')
	parser.add_argument('--combined_csv', action='store_true')
	parser.add_argument('--create_binary_thresholded', action='store_true')
	parser.add_argument('--ml', action='store_true')
	parser.add_argument('--utilities', action='store_true')
	parser.add_argument('--categorical', action='store_true')
	parser.add_argument('--reporter_node')
	parser.add_argument('--input_file')
	parser.add_argument('--output_file')
	args = parser.parse_args()
	if args.create_csv_files:
		#convert the txt files from ncbi to csv files
		for f  in os.listdir(args.exp_files_folder):
			if re.match(r'.*\.txt', f):
				ifile =args.exp_files_folder+ '/'+f
				name  = args.exp_files_folder+'/' + f.replace('txt', "csv")
				extract_series_matrix(ifile, name)
	if args.combined_csv:
		#create a combined csv from individually created csv files, for only the gene we want
		col_header, data_matrix =[], []
		for f  in os.listdir(args.exp_files_folder):
			if re.match(r'.*\.csv', f):
				ifile =args.exp_files_folder+ '/'+f
				
				head, data =extract_gene_rows(ifile)	
				del head[0]#del ref_ident now has exp names
				del data[0]#del gene names
				
				col_header.extend(head)
				data_matrix.extend(data)
		
		col_header.insert(0, 'name')
		row_header =[i for i in array_element_gene_name().values()]
		data_matrix.insert(0, row_header)
		for i, j in zip(data_matrix, col_header):
			i.insert(0, j)
		with open(args.output_file, 'w') as f:
			writer =csv.writer(f)
			for row in data_matrix:
				writer.writerow([i.encode('utf-8') for i in row])
	
	if args.create_binary_thresholded:
		col_names =np.genfromtxt(args.input_file, delimiter=',' ,dtype=str)[0]
		row_names =np.genfromtxt(args.input_file, delimiter=',', usecols=0, dtype=str)
		data =np.genfromtxt(args.input_file, delimiter=',', skip_header=1)[:, 1:]
		no_cols =data.shape[1]
		data =np.round(data)
		for i in range(no_cols):
			data [:, i] =otsu_thresholding(data[:, i])
		with open(args.output_file, 'w') as f:
			writer =csv.writer(f)
			writer.writerow(col_names[1:])
			for row in data:
				writer.writerow(row)

	if args.utilities:
		data =np.genfromtxt(args.input_file, delimiter=',', skip_header=1)
		print meu(5,1, data)

	if args.ml:
		if args.categorical:
			data =np.genfromtxt(args.input_file, delimiter=',', skip_header=1)
			reporter =int(args.reporter_node)
			y =data[:, reporter]
			comb =[i for i in range(data.shape[1]) if i !=reporter]
			#print comb
			output=[]
			for i in range(1, len(comb)+1):
				for j in combinations(comb, i):
					mask =j
					print "--------------Current feature set is ----------------------------------------"	
					print mask
					print "----------------------------------------------------------------------------------------"		
					x =data[:, mask]
					#print x.shape
					x =categorical(x)#categorical one hot applied
					#y =categorical(y)
					X_train, X_test, Y_train, Y_test =test_train_split(x, y)
					scores, names =classify(X_train,Y_train, X_test, Y_test)
					scores.insert(0, mask)
					output.append(scores)
			headers =['features']
			headers.extend(names)
		with open(args.output_file, 'w') as f:
			writer =csv.writer(f)
			writer.writerow(headers)
			for row in output:
				writer.writerow(row)
			
			#else:
			#	data =np.genfromtxt(args.input_file, delimiter=',', skip_header=1)[:, 0:]
