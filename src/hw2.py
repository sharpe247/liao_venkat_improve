from src.graph import Graph
from config.settings import *
from copy import deepcopy
from itertools import combinations
from itertools import product
from collections import Counter
from collections import defaultdict
from Queue import Queue
from Queue import PriorityQueue
from Queue import LifoQueue
from src.main import otsu_thresholding
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.sampling import BayesianModelSampling
from pgmpy.estimators import MaximumLikelihoodEstimator
from scipy import stats
import csv
import heapq
import argparse
import numpy as np
import src.figures as figures
import pandas as pd
import networkx as nx

from memory_profiler import profile

precision = 10

fp = open('memory_profiler_basic_mean.log', 'w+')
@profile(precision=precision, stream=fp)

def meu(reporter_node, cpt, graph, r_node_value=1 ):
	reporter_node =reporter_node -1
	#for each node find its parents--graph has it 
	#for all possible combinations of parent values find the most likely path 
	#use that cpt
	#iterative solution first to savetime
	no_nodes =len(graph.graph.keys())
	possible_paths =list(product([0, 1], repeat=no_nodes))
	v_paths =[]
	for path in possible_paths:
		v =1
		for idx, value in enumerate(path):
			parents =graph.graph[idx +1] #parents from the graph
			p_values =[ path[i-1] for i in parents]#map the parents in this path
			p_values.append(value)#store the current values of the parents
			for i in cpt:
				if idx ==i[0] and list(i[2]) ==p_values:
					v *=i[3]#multiply the cpt value for this node to v
		v_paths.append(v)
	cared_set =[[i,j] for i, j in zip(possible_paths, v_paths) if i[reporter_node] ==r_node_value]	
	cared_set.sort(key=lambda x: x[1])
	#print cared_set
	print "path", "score"
	for path, score in cared_set:
		print path, score

	#print "best_path is "+ str(possible_paths [v_paths.index(max(v_paths))])					i+1
	#print "the score is "+ str(max(v_paths))

def possible_results(graph, next):
	#next =(node, state(0/1))
	# idx ==i[0] and list(i[2]) ==p_values
	node =next[0] #the node
	current_state =next[1] #the state of current node
	parents =sorted(list(graph.graph[node]))#sets arent sorted
	results =[]
	for parent in parents:
		for i in [0,1]:
			results.append([parent, i])
	return results
	
def total_prob(nodes, X):
	#compute the mle total for all the nodes
	prob ={}
	total =len(X)
	for  i in nodes:
		ones =np.sum(X[:, i-1])/len(X)
		zeros =1 -ones
		prob[i] =[zeros, ones]
	return prob

def prob_from_data(node1, node1_value, node2, node2_value, X):
	#for 2 variables given their state
	#p(A=1|C=1) =P(A=1, C=1)/ P(C=1)
	#total =len(X)
	den =np.where(X[:, node2-1]==node2_value)[0].shape[0]
	mask =[node1-1, node2-1]
	num =len(np.where((X[:, mask] ==[node1_value, node2_value]).all(axis=1))[0])
	if den !=0:
		prob =float(num)/den
	else:
		prob =0.0
	#print ('prob path from %d = %d to %d =%d is %f' %(node1, node1_value, node2, node2_value, prob))
	#print (num, den)
	return prob
def prob_cost(new_state, prev_state, cpt, total_prob, graph, X):
	new_node =new_state[0]
	old_node =prev_state[0]
	new_node_value =new_state[1]
	old_node_value =prev_state[1]
	'''
	#marginalize all the other parents for this value of the new node(parent)
	parents = sorted(list(graph.graph[old_node]))
	parent_idx =parents.index(new_node)
	values, scores = [], []

	for i in cpt:
		if i[0] ==old_node and i[2][parent_idx] ==new_node_value and i[2][-1] ==old_node_value:
			
			#p(A = 1|B = 1) =p(A = 1|E = 0, B = 1)p(E = 0) + p(A = 1|E = 1, B = 1)p(E = 1)
			#so have to find the total probabilities for all the nodes
			values.append(i[2])#old and new node have values assigned
			scores.append(i[3])
	#print values
	#print scores
	remaining_idx =[k for k , j in enumerate(values[0]) if k !=parent_idx and k !=len(values[0])-1]#other parents

	cost =0
	if remaining_idx:
		for idx, value in enumerate(values):
			sub_value =np.array(value)[np.array(remaining_idx)]
			pdt =1
			for i, j in enumerate(sub_value):
				pdt *= total_prob[parents[remaining_idx[i]]][j]
			pdt *=scores[idx]
			cost +=pdt
		return -np.log(cost)	#to make costs additive and negative to sort in descending with heap queue
	else: #ie node had only one parent no marginalization required
		return -np.log(scores[0])
	'''
	#computing using the data for now not using the graph
	prob =prob_from_data(old_node, old_node_value, new_node, new_node_value, X)
	if prob !=0:
		return -np.log(prob)
	else:
		return np.inf
def find_path(path, k):
	if path[k]:
		print path[k],
		find_path(path, tuple(path[k]))
	else:
		print '\n'
		return
		
def paths(path, cost):
	for k in path.keys():
		print 'key is', k, 'cost is', np.exp(-1*cost[k])
		find_path(path, tuple(k))
			
			
def best_path(graph, cpt, reporter_node, reporter_state, total_prob, X):
	#state is the node and its value 0/1 
	#(node, value 0/1)
	
	#frontier =PriorityQueue()
	#frontier_set =set()
	#explored =set()
	source =reporter_node 
	state =reporter_state
	nodes =graph.nodes
	#path =defaultdict(list)
	cost =defaultdict(int)
	cost[(source, state)] =0
	
	#frontier_set.add((source, state))#node and values for all states
	#frontier.put((cost[source], (source, state)))
	#path[(source, state)] =[]
	previous =defaultdict(int)
	vertex_set =[]
	heapq.heappush(vertex_set, (cost[source, state], (source, state)))
	for i in nodes:
		if i !=source:
			cost[(i, 0)] =np.inf
			cost[(i, 1)] =np.inf
			previous[(i, 0)] =0#undefined
			previous[(i, 1)] =0#undefined
			heapq.heappush(vertex_set, (cost[i, 0], (i, 0)))
			heapq.heappush(vertex_set, (cost[i, 1], (i, 1)))
	while  vertex_set:
		#print vertex_set
		next =heapq.heappop(vertex_set)[1]	
		moves =possible_results(graph, next)
		for move in moves:
			new_state =tuple(move)
			#alt =cost[next] +prob_cost(new_state, next, cpt, total_prob, graph)
			alt =cost[next] +prob_cost(new_state, next, cpt, total_prob, graph, X)
			if alt < cost[new_state]:
				cost[new_state] =alt
				previous[new_state] =next
				heapq.heappush(vertex_set,(cost[new_state], new_state))

	for i, j in  sorted(cost.items(), key=lambda x:x[1]):
		print i, np.exp(-1 *j)
	return paths(previous, cost)

def check_max_parents(graph, max_no_parents):
	if sum([1 for i in graph.values()]) >0:
		return False
	else:
		True
#G is V list of nodes and E connections between the nodes 
def initialize_graph(nodes, max_no_parents):
	connections =set()
	#consider all the nodes but generate the edges randomly
	for node in nodes:
		parents =set()
		others =deepcopy(nodes)
		others.remove(node)
		#for each node decide the no of parents it should have
		
		no_parents =np.random.randint(0,high =max_no_parents+1)
		#randomly pick those parents
		if no_parents >0:
			parents =set([others[i]  for i in np.random.randint(len(others), size=(1, no_parents))[0]])
		#if there is a parent implies that there is  a connection	
		try: 
			for i in parents:
				connections.add((i, node))
		except UnboundLocalError:
			pass
		
	return connections

def true_inferene(true_cpt, graph, connections, nodes):
	pass	


def random_binary_cpt(graph, nodes):
	'''	
	for a given graph generate random cpt with binary values
	cpt =[[node, parents, row, prob],...]#node, set() of parents, values of all nodes(0/1) last one being this node's value,probability of this event
	Although there may be patterns to cpt values, we are right now just assuming that the values for a node add up to one
	'''
	cpt =[]
	for i in nodes:
		parents =graph.graph[i]
		table =list(product([0, 1], repeat=len(parents)))#binary combinations of the parent values
		for row in table:
			prob_0 =round(np.random.random(),2) #these sum to one
			prob_1 =round(1-prob_0, 2)
			cpt.append([i, parents, tuple(list(row)+[0]), prob_0]) #last value in the tuple is the value of the node
			cpt.append([i, parents, tuple(list(row)+[1]), prob_1])
	cpt =np.array(cpt)
	return cpt	

def convert_to_pgmy_cpt(cpt, nodes):
	'''
	evidence variables are parents, cardinality of all the variables is 2 in our case
	The cardinality of the variable in question is the number of rows in the array
	the number of columns is the possible combinations of the evidence or the parents
	in sorted order
	'''
	pgmy_cpt=[]	
	for i in nodes:
		cpt_i =cpt[cpt[:, 0]==i]
		variable =str(cpt_i[0][0])
		parents = sorted([i for i in list(cpt_i[0][1])])
		parents = [str(i) for i in parents]
		variable_card =2
		evidence_card =[2 for  i in parents]
		#values =[[] for i in range(variable_card)	]
		value_0, value_1 =[],[]
		for row in cpt_i:
			if row[-2][-1]==0:
				value_0.append(row[-1])
			else:
				value_1.append(row[-1])
		pgmy_cpt.append(TabularCPD(variable=variable, variable_card=variable_card, values =[value_0, value_1], evidence=parents, evidence_card=evidence_card))
	return pgmy_cpt
		
def calculate_cpt(graph, nodes, X):
	cpt =[]
	#print graph.graph
	for i in nodes:
		parents =graph.graph[i]
		table =list(product([0, 1], repeat=len(parents)+1))#assume last is the node we are considering
		mask =sorted(list(deepcopy(parents)))
		mask.append(i)#child
		mask =[i-1 for i in mask]#change nodes to indices to match
		sub_x =X[:, mask]
		for row in table:
			num =len(np.where((sub_x==row).all(axis =1))[0])
			
			if parents !=set():
				den =len(np.where((sub_x[:, 0:-1]==row[0:-1]).all(axis =1))[0])
			else:#if there are no parents, this calculation could be made easier
				den =len(X)
			
			
			
			if num ==0 and den==0:
				den=1 
			cpt.append([i, parents, row, round(float(num)/den, 4)])
	cpt =np.array(cpt)
	return cpt

def score(graph,nodes, X):
	cpt =calculate_cpt(graph,nodes, X)
	#print cpt
	value =0
	for idx, row in enumerate(X):
		row =np.array(row)
		for i in nodes:
			parents =graph.graph[i]
			mask =list(deepcopy(parents))
			mask.append(i)
			mask =[i-1 for i in mask]#change nodes to indices to match
			sub_x =tuple(row[mask])
			value += np.log(cpt[np.where((cpt[:, 0:-1]==np.array([i, parents, sub_x])).all(axis =1))[0][0]][-1])
			
	return value
def write_cpt(cpt, ifile):
	with open(ifile, 'w') as f:
		writer =csv.writer(f)
		for row in cpt:
			node =row[0]
			parents =tuple(row[1])
			value =tuple(row[2])
			cpt_vale =row[3]
			writer.writerow(row)

def accuracy(y, Y):
	tp =0
	fp =0
	for i, j in zip(y, Y):
		if i==1 and j[4]==1:
			tp +=1
		elif i ==1 and j[4]!=1:
			fp +=1
	return float(tp)*100/(tp + fp)

def given_algo(graph, connections, nodes, data, max_no_parents):
	#loop until there is no change in score 
	#i'll regard that as negligible score
	for i in range(200):#since no of combinations is so few 100 iterations is sufficient
		#print i
		graph_best =deepcopy(graph)
		score_best =score(graph_best,nodes, data)
		new_connections =deepcopy(connections)
		comb =[i for  i in combinations(nodes, 2)]#all possible pairs, this is without replacement
		pair =comb[np.random.randint(0,high =len(comb))]
		#print pair
		
		#addition
		if pair not in connections:#expensive call
			#print "add"
			new_connections.add(pair)
			new_graph =Graph(nodes, new_connections)
			
			if not new_graph.cyclic and check_max_parents(new_graph.graph, max_no_parents):
				new_score =score(new_graph,nodes, data)
				if new_score >score_best:
					graph_best =new_graph
					connections =new_connections
					score_best =new_score
			else:
				pass
		else:	
			#deletion
			pick =np.random.randint(0, high=2)
			if pick ==1:
				#print "del"
				new_connections.remove(pair)
				new_graph =Graph(nodes, new_connections)
				if not new_graph.cyclic:
					new_score =score(new_graph,nodes, data)
					if new_score >score_best:
						graph_best =new_graph
						connections =new_connections
						score_best =new_score
				else:
					#print "but pass"
					pass
			else:
				#reversal
				#print "reversal"
				new_connections =deepcopy(connections)
				new_connections.remove(pair)
				x, y =pair
				new_connections.add(tuple([y, x]))
				new_graph =Graph(nodes, new_connections)
				if not new_graph.cyclic:
					new_score =score(new_graph,nodes, data)
					if new_score >score_best:
						graph_best =new_graph
						connections =new_connections
						score_best =new_score
				else:
					#print "but pass"
					pass
	return graph_best, score_best
def create_data(ifile):
	X =np.genfromtxt(ifile, delimiter=',', skip_header=1)
	no_nodes =X.shape[1]	
	return X, no_nodes

def predict(Y, cpt):
	#prediction for g5
	y=[]
	sub_cpt =[]
	for i in cpt:
		if i[0] ==5:
			parents =i[1]
			sub_cpt.append([i[2],i[-1]])
	indices =[i-1 for i in parents]
	sub_cpt =np.array(sub_cpt)
	if parents !=set():
		for i  in Y:
			for  j in sub_cpt	:
				if (i[indices] ==j[0][0:-1]).all and j[0][-1] ==1:
					if j[1] >0.5:
						y.append(1)
					else:
						y.append(0)
	else:
		if sub_cpt[1,-1]>0.5:
			y=[1 for i in Y]
		else:
			y=[0 for i in Y]
	return y

def parse_connections(ip):
	'''ip =(1,2) (3,4) (4,5), 1 is parent 2 is child'''
	connections =set([eval(f) for f in ip.strip(' \t').split(' ')])
	return connections


#def find_path(depth, paths, graph,node):
	#parents =graph[node]
	#for p in parents:
	#	paths[depth].append([node, p])
def k_length_paths(graph, reporter_node):
	#staring at the reporter traverse to find the paths of increasing lengths
	#each node has 2 states 0,1 
	paths =defaultdict(list)
	path_length =1
	paths[0] =[[reporter_node]]
	depth =0
	node =reporter_node
	exit =False
	while 1:
		for path in paths[depth]:
			print path
			node =path[-1]
			parents =graph[node]
			if parents:
				for p in parents:
					paths[depth +1].append(path +[p] )
			else:
				exit =True
		if not exit:
			depth +=1
		else:
			break
	return paths

def convert_path_with_state(paths, case='activation'):
	#start(reportter) and end are both 1 for activation
	#start(reporter) is 1 and end is 0 for inhibition
	state_paths =defaultdict(list)
	if case =='activation':
		for path_length, values in paths.items():
			tmp2 =[]
			for value in values:
				#tmp1 =[]
				if len(value) ==1:
					tmp2.append([value[0],1])
				elif len(value) ==2:
					tmp2.append([(value[0],1) ,(value[1],1)])
				else:
					for i in [0,1]:
						tmp =[]
						for idx, v in enumerate(value):
							if idx ==0 or idx ==len(value) -1:
								tmp.append((v,1))
							else:
								tmp.append((v, i))
						tmp2.append(tmp)
			state_paths[path_length] =tmp2
		return state_paths
def generate_records(dist_nodes, value, count):
	fix_node =[i[0] for i in value]
	fix_value =[i[1] for i in value]
	no_nodes =len(dist_nodes)
	data =[]
	for c in range(count):
		ini =[0 for i in range(no_nodes)]
		for n in range(no_nodes):
			if n+1 in fix_node:
				ini[n] = fix_value[fix_node.index(n+1)]
			else:
				z= np.random.random_sample()

				if z > dist_nodes[n+1]:
					ini[n] =1
				else:
					ini[n] =0
		data.append(ini)
	return data	
	

def multitree_random(no_nodes, avg_indegree, sd_indegree, no_cycles, max_cycle_nodes, separate=False): 
	'''
	Generate a random DAG with cycles in it : Multitree
	1. Constraints no self loops, no directed loops (since a DAG)
	2. Min no of nodes for a cycle to be undirected but not directed is 3
	Input:
		Number of nodes :- no_nodes
		Average in-degree of a node(proxy for no of parents):- avg_indegree
		Sd in-degree of nodes(spread of the difference in no of parents):- sd_indegree
		Number of undirected cycles: no_cycles
		Max nodes in a cycle: max_cycle_nodes
	https://stackoverflow.com/questions/35275877/generate-a-directed-graph-with-n-cycles
	'''
	if max_cycle_nodes >= no_nodes or max_cycle_nodes <3:
		print 'max_cycle has inapt no of nodes'
		return
	else:
		while 1:
			nodes =np.arange(no_nodes) #the nodes starting at 0
			adj_matrix =np.zeros((no_nodes, no_nodes))#initialize the adj matrix to be all zeros
			#pick nodes in the biggest cycle 
			big_cycle_nodes =np.random.choice(nodes, max_cycle_nodes, replace=False)
			cycle_set =set()
			#make an undirected cycle b/w them 
			tmp =np.roll(big_cycle_nodes,-1)#shift the nodes by one
			for i, j in zip(big_cycle_nodes, tmp):
				adj_matrix[i][j] =1
				adj_matrix[j][i] =1
				cycle_set.add(i)
				cycle_set.add(j)
			no_cycles -=1
			if separate ==False:
				while no_cycles >0:
					done =False
					try:
						len_cycle =np.random.randint(3, max_cycle_nodes)#no of nodes in this cycle
					except ValueError:
						len_cycle =3
					#cycle_nodes =np.random.choice(big_cycle_nodes, len_cycle, replace=False)
					cycle_nodes =big_cycle_nodes
					#there are max_cycle_nodes -len_cycle +1 cycles of length len_cycle directed cycles
					#this method creates additional cycles 
					tmp =np.roll(cycle_nodes, -1*(len_cycle-1))
					for i, j in zip(tmp[0:len_cycle -1], cycle_nodes[0:len_cycle-1]):
						if adj_matrix[i][j] !=1:
							adj_matrix[i][j] =1
							adj_matrix[j][i] =1
							no_cycles -=1
							done =True
							break
						else:
							pass
					#cant incorporate this small cycle into the biggest one
					#find new nodes
					if done ==False:
						print 'no graph found'
						break
						pass
						#deal with this later
			else:#assuming there are enough nodes and other corner cases
				while no_cycles >0:
					try:
						len_cycle =np.random.randint(3, max_cycle_nodes)
					except ValueError:
						len_cycle =3
					cycle_nodes =np.random.choice(nodes, len_cycle, replace=False)
					if cycle_set:
						while 1:
							if sum([1 for i in cycle_nodes if i in cycle_set]) >0:
								cycle_nodes =np.random.choice(nodes, len_cycle, replace=False)
							else:
								break
					tmp =np.roll(cycle_nodes,-1*(len_cycle-1))			
					for i, j in zip(cycle_nodes, tmp):
						if adj_matrix[i][j] !=1:
							adj_matrix[i][j] =1
							adj_matrix[j][i] =1
							cycle_set.add(i)
							cycle_set.add(j)
					no_cycles -=1
			print "cycles done"
			#add other connections
			for n in nodes:
				#other_nodes =np.delete(nodes, np.where(nodes ==n))
				#other_nodes =deepcopy(list(nodes))
				#other_nodes.remove(n)
				other_nodes =nodes[np.where(nodes !=0)]
				#del other_nodes[other_nodes.index(n)]
				in_degree =int(np.random.normal(avg_indegree, sd_indegree))
				if not np.sum(adj_matrix[n]) >=in_degree:
					parents =np.random.choice(other_nodes, int(in_degree-np.sum(adj_matrix[n])), replace=False)
					for p in parents:
						adj_matrix[n][p] =1
						adj_matrix[p][n] =1
			print "graph done checking if connected"
			#convert the adj matrix to adj list with total ordering ie a DAG
			if is_connected(adj_matrix, nodes):
				break
			else:
				break
		adj_list, reporter_node =total_ordering(adj_matrix, nodes)
		graph =Graph(nodes, adj_list)
		return graph, reporter_node

def is_connected(adj_matrix, nodes):
	#start at one nodes reach all the nodes
	start =np.random.randint(0, len(nodes))
	nodes_reached =set()
	q =LifoQueue()
	connected =False
	while 1:
		nodes_reached.add(start)
		current_connected =np.where(adj_matrix[start] !=0)[0]
		for i in current_connected:
			nodes_reached.add(i)
			if i not in nodes_reached:
				q.put(i)
		if len(nodes_reached) ==len(nodes):
			connected =True
			break
		else:
			if q.empty():
				break
			else:
				start =q.get()
	print "graph is connected", connected
	return connected	

def total_ordering(adj_matrix, nodes):
	print "runnning total ordering"
	#ip adj_matrix output adj list
	frontier =LifoQueue()# stack
	#frontier_set =set()
	explored =set()#states are  tuples and hashable
	ini_state =np.random.randint(0, len(nodes))
	#frontier_set.add(ini_state)
	frontier.put(ini_state) 
	if frontier.empty():
		#print ("no solution"),
		return
	
	while not frontier.empty():
		next =frontier.get()
		print ('node selected is ',next)
		#frontier_set.remove(next)
		moves =np.where(adj_matrix[next] ==1)[0]
		print ('outgoing edges are ',moves)
		for move in moves:
			print ('changing edges for ', next, move)
			new_state =move
			if new_state not in explored:
				adj_matrix[next][move] =1
				adj_matrix[move][next] =-1
				print ('adjancency mat', adj_matrix)
				#frontier_set.add(new_state)
				frontier.put(new_state)
				#print 'frontier set', frontier_set
				print 'frontier q ', frontier.queue
		explored.add(next)
		print 'explored', explored
		if len(explored) == len(nodes):
			print ("all edges assigned in total ordering")
			break

	adj_list =[]
	for idx, row in enumerate(adj_matrix):
		children =np.where(row ==1)[0]
		for i in children:
			adj_list.append((idx, i))
	#each dag has atleast one leaf node picking the first one
	for idx, row in enumerate(adj_matrix):
		row_len =len(row)
		satisfying =sum([1 for i in row if i ==0 or i==-1])
		if row_len ==satisfying:
			reporter_node =idx
			break
	#reporter_node =np.where(adj_matrix.sum(axis=1) <0)[0][0]
	return adj_list, reporter_node

def generate_data(graph, reporter, influencing_prop, uninfluencing_prop):
	dist ={}
	correlation ={}
	sd ={}
	theta ={}
	sd_seed1 =round(np.random.uniform(0,5),2)
	sd_seed2 =round(np.random.uniform(0,5),2)
	nodes =[i for i in graph.keys() if i !=reporter]
	influencing =[]
	non_influencing =[]
	random_nodes =[]
	#find the parents and put in inlfuencing
	reporter_parents  =list(graph[reporter])
	#random pick 
	while 1:
		selected =np.random.choice(nodes, 1)[0]
		if selected not in reporter_parents:
			break
	if np.random.random_sample() >0.5:
		correlation[selected] =0.8
	else:
		correlation[selected] =-0.8
	theta[selected] =0.5 * np.arcsin(correlation[selected])
	for i in reporter_parents:
		if np.random.random_sample() >0.5:
			correlation[i] =round(np.random.uniform(0.8, 1),2)
		else:
			correlation[i] =round(np.random.uniform(-0.8, -1),2)
		theta[i] =0.5 * np.arcsin(correlation[i])
		#sd[i] =round(np.random.uniform(0,5),2)
	count_influencing =int( influencing_prop *len(nodes))
	total_influencing =count_influencing + 1+ len(reporter_parents)
	#if count_influencing <=0:
	#	pass
	#else:
	influencing.extend(np.random.choice([i for i in nodes if i not in reporter_parents and i !=selected], count_influencing, replace=False))
	for i in influencing:
		if np.random.random_sample() >0.5:
			correlation[i] =round(np.random.uniform(0.5, 0.8),2)
		else:
			correlation[i] =round(np.random.uniform(-0.5, -0.8),2)
		theta[i] =0.5 * np.arcsin(correlation[i])
	count_non_influencing =int(uninfluencing_prop * len(nodes))
	#if count_non_influencing <=0:
	#	pass
	#else:
	try:
		non_influencing.extend(np.random.choice([i for i in nodes if i not in influencing and i !=selected and i not in reporter_parents], count_non_influencing, replace=False))
	except ValueError:
		pass
	for i in non_influencing:
		if np.random.random_sample() >0.5:
			correlation[i] =round(np.random.uniform(0, 0.5),2)
		else:
			correlation[i] =round(np.random.uniform(0, -0.5),2)
		theta[i] =0.5 * np.arcsin(correlation[i])
		#sd[i] =round(np.random.uniform(0,5),2)
	count_random =len(nodes) -total_influencing -count_non_influencing
	try:
		random_nodes.extend(np.random.choice([i for i in nodes if i not in influencing and i not in non_influencing and i!=selected and i not in reporter_parents], count_random, replace=False))
	except ValueError:
		pass
	for i in random_nodes:
		sd[i] =round(np.random.uniform(0,5),2)
	print "random nodes ", random_nodes
	print "selected ", selected
	print "influencing ", influencing
	print "non_influencing ", non_influencing
	print "reporter_parents", reporter_parents
	'''X1 = aY1 + bY2, X2 = cY1 + dY2.
	a = cos(theta), b = sin(theta)
	c = sin(theta), d = cos(theta)
	theta =0.5 inverse_sin(correlation_coeff_x1_x2)
	'''
	samples =[]
	for i in range(100): #generate 100 samples
		sample =[]
		for i in nodes:
			report_x =0
			if i in random_nodes:
				sample.append(50*np.random.normal(loc=0,scale=sd[i]))
			else: 
				y1 =50*normal(sd_seed1)
				y2 =50*normal(sd_seed2)
				x1 =(np.sin(theta[i] )* y1) + (np.cos(theta[i])* y2 )#assume this to be for reporter node
				x2 =(np.sin(theta[i]) * y2) + (np.cos(theta[i])* y1 )#assume this to be for the related node
				report_x +=x1
				sample.append(x2)
		sample.append(report_x)
		samples.append(sample)
	return samples
def normal(sd):
	return np.random.normal(loc=0 , scale=sd)			

def variable_elimination(model, reporter_node, no_evidence, nodes):
	infer =VariableElimination(model)
	other_nodes =[str(i) for i in nodes if i!= reporter_node]
	evidences =[]
	probs =[]
	for tup in combinations(other_nodes, no_evidence):
		done =0
	i	for val in product([0,1], repeat=no_evidence):
			print 'single prob infered %d' %done
			evidence ={i:j for i, j in zip(tup, val)}
			evidences.append(evidence)
			print 'evidence is ', evidence
			inf_prob =infer.query([str(reporter_node)], evidence=evidence)[str(reporter_node)].values
			probs.append(inf_prob)
			#print 'reporter state 0 has probability', inf_prob[0]
			#print 'reporter state 1 has probability', inf_prob[1]
			done +=1
	return probs, evidences

def variable_elimination_filtered_set(model, reporter_node, no_intervention, nodes, filtered_evidences):
	'''
	filtered_evidences is a list of tuples of evidences
	'''
	infer =VariableElimination(model)
	other_nodes =[str(i) for i in nodes if i!= reporter_node]
	evidences =[]
	probs =[]
	for tups in filtered_evidences:
		evidence ={}
		for tup in tups:
			evidence[tup[0]] =tup[1]
		evidences.append(evidence)
		inf_prob =infer.query([str(reporter_node)], evidence=evidence)[str(reporter_node)].values
		probs.append(inf_prob)
	return probs, filtered_evidences


def venkat(model, reporter_node, no_evidence, nodes):
	#this is only for one variable
	infer =VariableElimination(model)
	other_nodes =[str(i) for i in nodes if i!= reporter_node]
	evidences =[]
	probs =[]
	for tup in combinations(other_nodes, no_evidence):
		for val in product([0,1], repeat=no_evidence):
			evidence ={i:j for i, j in zip(tup, val)}
			evidences.append(evidence)
			#print 'evidence is ', evidence
			inf_prob_cond =infer.query([str(reporter_node)], evidence=evidence)[str(reporter_node)].values
			inf_prob_node_marginal =infer.query([str(evidence.keys()[0])])[str(evidence.keys()[0])].values[evidence.values()[0]]
			probs.append(inf_prob_cond * inf_prob_node_marginal)
			#print 'reporter state 0 has probability', inf_prob_cond[0]
			#print 'reporter state 1 has probability', inf_prob_cond[1]
	return probs, evidences

def random_graph_adj_mat(s):
	'''
	generate random strictly upper triangular matrix
	this has total order hence a dag, it is possible that there are unconnected 
	components in it. That is however under the definition of bayesian nets
	This is biased at creating graphs with large no of connections
	'''
	#generate random square matrix
	mat =np.random.randint(2, size=(s, s))
	#convert to upper triangular
	mat =np.triu(mat)
	#convert to strictly upper
	np.fill_diagonal(mat, 0)#0 implies main diagonal
	return mat
def random_graph_networkx(s):
	#using networkx random graph creater
        #no of nodes is s, assuming max no of parents as 4
        #and min as 1 so sampling for 10 nodes a no b/w [10, 40] as the no of edges on 
        #average we expect 25 edges
	#count =0
	while 1:
		lowest_no_edges =s*1
		highest_no_edges =s*4 +1
	        edge_count =np.random.randint(lowest_no_edges, high=highest_no_edges)
        	G=nx.gnm_random_graph(s, edge_count, seed=None)
		#count +=1
		if nx.is_connected(G):
			break
	#G =nx.fast_gnp_random_graph(n, p,seed=None directed=True)
        #there is still a possibility that some nodes would be unconnected
	#print count
        connections =[(i+1, j+1) for (i, j) in G.edges() if i <j] #increment the indices to match i<j ensures no cycles
	return connections
def adj_mat_to_connections(mat):
	connections= [(r_idx+1, idx+1)for r_idx, row in enumerate(mat) for idx, i in enumerate(row) if i==1 ]
	return connections

def utility1_values(probs, evidences):
	'''
	P(R=1|X=x), use this probability computation for utility--single intervention
	evidences are list of dicts
	'''
	utility1 ={}
	for prob, evidence in zip(probs, evidences):
		evidence_tup =evidence.items()[0]#there is only one evidence
		utility1[evidence_tup] =prob
	return utility1

def utility_2_3_values(probs, evidences, model):
	'''
	utility2 P(R=1|X=x). P(X=x), use this probability computation for utility--single intervention
	utiltiy3 P(R=1|X=x).(1-P(X=x)), use this probability computation for utility--single intervention
	'''
	infer =VariableElimination(model)
	utility2, utility3 ={}, {}
	for prob, evidence in zip(probs, evidences):
		evidence_tup =evidence.items()[0]#there is only one evidence
		marginal_prob =infer.query([str(evidence.keys()[0])])[str(evidence.keys()[0])].values[evidence.values()[0]]
		utility2[evidence_tup] =prob * marginal_prob
		utility3[evidence_tup] =prob * (1-marginal_prob)
	return utility2, utility3

def compute_corr_ordering(probs, evidences, no_intervention, heu_dict):
	'''
	heu_dict: heuristic intervention utility values for sets(combination of evidences) computed using single intervention utilities
	; dict key is evidence value is prob value
	no_intervention- no of intervention points
	evidences - list of dicts, dict contain the evidences
	probs- probabilites for these evidences exact for set of nodes
	'''
	#print heu_dict
	heu_probs =[]	
	for prob, evidence in zip(probs, evidences):
		heu_probs.append(heu_dict[evidence])
	spearman =compute_spearman(probs, heu_probs)
	kendall =compute_kendall(probs, heu_probs)
	if np.isnan(spearman[0]) or np.isnan(kendall[0]):
		print  'heu_probs' ,heu_probs
		print 'probs', probs
		print 'evidences', evidences
	return spearman, kendall


def compute_spearman(order1, order2):
	'''
	compute spearman rank order coeff for the 'top' elements in the orders
	spearman can't handle ties
	np.argsort by default returns higer index in case of ties 
	'''
	rand_array =np.random.random(len(order1))
	order1_ties =np.lexsort((rand_array, order1))#break ties randomly sort of 
	#print order1_ties
	order1_arg_sorted =np.argsort(order1_ties)
	rand_array =np.random.random(len(order2))
	order2_ties =np.lexsort((rand_array, order2))
	order2_arg_sorted =np.argsort(order2_ties)
	spear =stats.spearmanr(order1_arg_sorted, order2_arg_sorted)
	return spear

def compute_kendall(order1, order2):
	'''
	Compute the kendall tau for the 2 orderings
	This can handle ties: so sort the uniq array convert it to dict with elements as
	keys and index as value
	Then use this to argsort original array
	'''
	order1_dict ={i:idx for idx, i in enumerate(np.sort(np.unique(order1)))}
	order1_arg_sorted =[order1_dict[i] for i in order1]
	order2_dict ={i:idx for idx, i in enumerate(np.sort(np.unique(order2)))}
	order2_arg_sorted =[order2_dict[i] for i in order2]
	kendall =stats.kendalltau(order1_arg_sorted, order2_arg_sorted)
	return kendall
	
def find_top_utility(utility, no_intervention, top=20, computation='add'):
	'''
	filter top items: make combinations sum them sort them then filter the top ones
	'''
	if computation =='add':
		combined =[]
		for comb in combinations(utility.items(), no_intervention):
			tmp_sum =0
			tmp_element =[]
			for i in comb:
				tmp_sum +=i[1]
				tmp_element.append(i[0])
			combined.append((tuple(tmp_element), tmp_sum))
	elif computation =='multiply':
		combined =[]
		for comb in combinations(utility.items(), no_intervention):
			tmp_pdt =1
			tmp_element =[]
			for i in comb:
				tmp_pdt *=i[1]
				tmp_element.append(i[0])
			combined.append((tuple(tmp_element), tmp_pdt))
	combined.sort(key=lambda x: x[1])
	filter_sorted_comb =combined[-top:][::-1]
	return {i[0]:i[1] for i in filter_sorted_comb}



if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--ga_graph', action ='store_true')
	parser.add_argument('--fixed_graph', action ='store_true')
	parser.add_argument('--connections')
	parser.add_argument('--max_no_parents')
	parser.add_argument('--train_file')
	parser.add_argument('--test_file')
	parser.add_argument('--random_iterations')
	parser.add_argument('--no_nodes')
	parser.add_argument('--simulate_data', action='store_true')
	parser.add_argument('--check_path_length')
	parser.add_argument('--new_simulation', action='store_true')
	parser.add_argument('--test_method', action='store_true')
	parser.add_argument('--reporter_node')
	parser.add_argument('--reporter_state')
	parser.add_argument('--infer_graph', action='store_true')
	parser.add_argument('--multi_hypothesis', action='store_true')
	args = parser.parse_args()



	if args.ga_graph:
		X, no_nodes =create_data(args.train_file) #training data
		nodes =[i+1 for i in range(no_nodes)]
		exc_score =np.NINF# negative infinity
		for i in range(int(args.random_iterations)):
			print "Experimental Iteration no "+str(i)+"---------------------------------------"
			while 1:
				connections =initialize_graph(nodes, int(args.max_no_parents))
				#print connections
				graph =Graph(nodes, connections)
				#print graph.graph
				if not graph.cyclic:
					break
			#break
			graph_best, score_best =given_algo(graph, connections, nodes, X, int(args.max_no_parents))
			if score_best >exc_score:
				exc_score =score_best
				exc_graph =deepcopy(graph_best)
			print "best score-----------"
			print exc_score
			print exc_graph.graph
		print "best graph is--------"
		print exc_graph.graph
		cpt =calculate_cpt(exc_graph, nodes, X)
		print "best cpt------"
		print cpt
		total_prob =total_prob(nodes, X)
		best_path(graph, cpt, 5, 1, total_prob)

	if args.fixed_graph:
		X, no_nodes =create_data(args.train_file) #training data                                     		
		nodes =[i+1 for i in range(no_nodes)] 
		exc_score =np.NINF# negative infinity#connections for venkat  (1,5) (9,5) (8,1) (2,9) (7,8) (4,8) (4,2) (6,2) (6,7) (6,4) (3,6)
		#to compare with venkat et all, fixing the graph
		connections =parse_connections(args.connections)
		graph =Graph(nodes, connections)# assuming that this fixed graph is not cyclic
	
		print "Graph"
		print graph.graph, '\n'
		print 'Score: ',score(graph,nodes, X)
		cpt =calculate_cpt(graph, nodes, X)
		print "CPT values: "
		print cpt, '\n'
		total_prob =total_prob(nodes, X)
		#print cpt
		print 'best path'
		best_path(graph, cpt, 7, 0, total_prob)
	
	if args.simulate_data:
		nodes =[i+1 for i in range(int(args.no_nodes))]
		while 1:
			connections =initialize_graph(nodes, int(args.max_no_parents))
			print connections
			#connections =parse_connections(args.connectiongenerate_records(dist_nodes, values[idx], count )s)
			#print connections
			graph =Graph(nodes, connections)
			#print graph.graph
			if not graph.cyclic:
				break
		print graph.graph
		#no_nodes =int(args.no_nodes)
		while 1:
			node =np.random.randint(1, int(args.no_nodes) +1)
			if graph.graph[node]:
				reporter_node =node
				break
		
		paths =k_length_paths(graph.graph, reporter_node)
		state_paths =convert_path_with_state(paths, case='activation')
		path_length =int(args.check_path_length)
		values =state_paths[path_length]
		#randomly pick one to be best
		pick =np.random.randint(1, len(values))
		best_path =values[pick]
		print 'best path ', best_path
		#generate a distribution for the values with highest for best_path
		dist_paths =np.sort(np.random.random_sample(len(values)))[::-1]
		count_positive = [int(100* i) for  i in dist_paths]
		dist_nodes ={}
		for i in range(int(args.no_nodes)):
			dist_nodes[i+1] =np.random.random_sample()
		data_set =[]
		#best_path_count =int(dist_paths[0]*100)
		for idx , count in enumerate(count_positive):
			data_set.extend(generate_records(dist_nodes, values[idx], count ))
		print data_set
	if args.new_simulation:
		no_nodes =100
		avg_indegree =3
		sd_indegree =1
		no_cycles =4
		max_cycle_nodes =5
		separate =True
		influencing_prop =0.1
		uninfluencing_prop =0.4
		graph, reporter_node =multitree_random(no_nodes, avg_indegree, sd_indegree, no_cycles, max_cycle_nodes, separate=True)
		print graph.graph
		print reporter_node
		figures.make_graph(graph.graph, '', 'graph/true_graph')
		
		samples =generate_data(graph.graph, reporter_node, influencing_prop, uninfluencing_prop)
		'''
		samples =np.array(samples)
		no_cols =samples.shape[1]
                data =np.round(samples)
                for i in range(no_cols):
                        samples [:, i] =otsu_thresholding(samples[:, i])
		'''
		samples =np.array(samples)
		print samples[0]
		samples[samples>0] =1
		samples[samples<0] =0
		print samples	
		#inference setting max parents to 3	
		nodes =[i+1 for i in range(no_nodes)]
		exc_score =np.NINF# negative infinity
		for i in range(5):
                        print "Experimental Iteration no "+str(i)+"---------------------------------------"
                        while 1:
                                connections =initialize_graph(nodes, int(3))
                                #print connections
                                graph =Graph(nodes, connections)
                                #print graph.graph
                                if not graph.cyclic:
                                        break
                        #break
                        graph_best, score_best =given_algo(graph, connections, nodes, samples, 3)
                        if score_best >exc_score:
                                exc_score =score_best
                                exc_graph =deepcopy(graph_best)
                        #print "best score-----------"
                        #print exc_score
                        #print exc_graph.graph
		print "best graph is--------"
		print exc_graph.graph
		cpt =calculate_cpt(exc_graph, nodes, samples)
		print "best cpt------"
		print cpt
		figures.make_graph(graph.graph, '', 'graph/infered_graph')
		total_prob =total_prob(nodes, samples)
		best_path(graph, cpt, reporter_node, 1, total_prob)


	if args.test_method:
		#python src/hw2.py  --test_method --connections "(7,8) (7,9) (8,9) (9,10) (9,1) (2,1) (3,1) (4,3) (2,1) (6,4) (6,5)" --reporter_node 1 --reporter_state 1 --infer_graph >>runs/infer/single/1
		"(7,8) (7,9) (8,9) (9,10) (9,1) (2,1) (3,1) (4,3) (2,1) (6,4) (6,5)"
		connections =parse_connections(args.connections)
		no_nodes =len(set([j for i in connections for j in i]))
		reporter_node =int(args.reporter_node)
		reporter_state =int(args.reporter_state)
		nodes =[i+1 for i in range(no_nodes)]
		graph =Graph(nodes, connections)
		#print ('Input Graph is ')
		#print (graph.graph)
		figures.make_graph(graph.graph, '', 'figures/true_graph')
		true_cpt =random_binary_cpt(graph, nodes)
		#print "True Cpt ", true_cpt
		#conditional probability query P(reporter_node=reporter_state|X=x) for all other nodes
		node_state_truth =true_inferene(true_cpt, graph, connections, nodes)
		#convert connection values to string
		connections_str =[(str(i), str(j)) for i,j in connections]
		pgmpy_graph =BayesianModel(connections_str)
		pgmpy_graph_copy =BayesianModel(connections_str)
		pgmpy_true_cpt =convert_to_pgmy_cpt(true_cpt, nodes)
		#print "pgmpy true cpt-- " 
		#for i in pgmpy_true_cpt:
		#	print i
		for row in pgmpy_true_cpt:
			pgmpy_graph.add_cpds(row)
		
		#print "Cpds correct according to the graph? ",pgmpy_graph.check_model()
		#print 'Using variable elimination to run inference'
		no_evidence =1
		#print "No of evidence variables %d" %no_evidence
		probs, evidences =variable_elimination(pgmpy_graph, reporter_node, no_evidence, nodes)
		probs =np.array(probs)
		print 'True utilities given 1 Evidence variable in decreasing order of probability are '
		for i in np.argsort(probs[:, reporter_state])[::-1]:
			print evidences[i], probs[i][reporter_state]
		no_evidence =2
		#print "No of evidence variables %d" %no_evidence
		probs, evidences =variable_elimination(pgmpy_graph, reporter_node, no_evidence, nodes)
		probs =np.array(probs)
		print 'True utilities given 2 Evidence variables in decreasing order of probability are '

		for i in np.argsort(probs[:, reporter_state])[::-1]:
			print evidences[i], probs[i][reporter_state]
		print 'Generating data'
		sampler =BayesianModelSampling(pgmpy_graph)
		data =sampler.forward_sample(size=1000, return_type='recarray')
		print '1000 data points generated'
		X =data.view(np.int).reshape(data.shape + (-1,))#convert to np array
		total_prob =total_prob(nodes, X)
		if args.infer_graph:
			exc_score =np.NINF# negative infinity
			for i in range(10):
				print "Experimental Iteration no "+str(i)+"---------------------------------------"
				while 1:
					new_connections =initialize_graph(nodes, int(3))
					inf_graph =Graph(nodes, connections)
					if not inf_graph.cyclic:
						 break
				graph_best, score_best =given_algo(inf_graph, new_connections, nodes, X, 3)
				if score_best >exc_score:
					exc_score =score_best
					exc_graph =deepcopy(graph_best)
			print "best graph is--------"
			print exc_graph.graph
			inf_cpt =calculate_cpt(exc_graph, nodes, X)
			print "best cpt------"
			print inf_cpt
			figures.make_graph(graph.graph, '', 'figures/infered_graph')

		else:
			inf_cpt =calculate_cpt(graph, nodes, X)
			#print 'infered Cpt is ', inf_cpt
			print 'best path'
			best_path(graph, inf_cpt, reporter_node, reporter_state, total_prob, X)
			data_frame_dict =defaultdict(list)
			for row in X:
				for idx, i in enumerate(row):
					data_frame_dict[str(idx +1)].append(i)
			data_frame =pd.DataFrame(data_frame_dict)
			pgmpy_graph_copy.fit(data_frame)
			#print "pgmpy inferred probs"
			#for i in pgmpy_graph_copy.get_cpds():
			#	print i
			#probs, evidences =variable_elimination(pgmpy_graph_copy, reporter_node, no_evidence, nodes)
			#probs =np.array(probs)
			#print 'PGMPY inference in decreasing order of probability are '
			#for i in np.argsort(probs[:, reporter_state])[::-1]:
			#	print evidences[i], probs[i][reporter_state]
			no_evidence =1
			venkat_probs, venkat_evidences =venkat(pgmpy_graph_copy, reporter_node, no_evidence, nodes)
			venkat_probs =np.array(venkat_probs)
			print 'Venkat utilities given Evidence in decreasing order of probability are '
			for i in np.argsort(venkat_probs[:, reporter_state])[::-1]:
				print venkat_evidences[i], venkat_probs[i][reporter_state]
	if args.multi_hypothesis:
		no_nodes =[i for i in range(0,51,10)]
		#no_nodes =[50]
		graphs_per_node_count =5
		cpts_per_graph =5
		max_intervention_pts =4
		#reporter node shift per graph sqrt no_nodes
		observations =defaultdict(list)
		for node_count in no_nodes:#iterationg over differnt node sizes
			#no_reporter_node_shifts =int(np.sqrt(node_count)) #no of reporter node shifts
			no_reporter_node_shifts =3
			print 'Current node count ', node_count
			for g in range(graphs_per_node_count):#iterating for number of graphs per node size
				print 'Current graph iteration %d out of %d' %(g, graphs_per_node_count)
				#mat =random_graph_adj_mat(node_count)#totally ordered adj matrix 
				#print mat
				#connections =adj_mat_to_connections(mat) #adj list for that mat parent to child
				connections =random_graph_networkx(node_count)
				#connections =[(7,8), (7,9), (8,9), (9,10), (9,1), (2,1), (3,1), (4,3), (2,1), (6,4), (6,5)]
				nodes =[i+1 for i in range(node_count)]#node indices incremented by 1 
				nodes_str =[str(i) for i in nodes]
				graph =Graph(nodes, connections)#generate graph
				print 'Current graph structure'
				print graph.graph
				print connections
				connections_str =[(str(i), str(j)) for i,j in connections]#convert connections to str
				pgmpy_graph =BayesianModel()#initialize bn 
				pgmpy_graph.add_nodes_from(nodes_str)
				pgmpy_graph.add_edges_from(connections_str)
				for c in range(cpts_per_graph):#iterating for number of cpts per graph structure
					print 'Current cpt iteration %d out %d' %(c, cpts_per_graph)
					true_cpt =random_binary_cpt(graph, nodes)#generate random cpt
					pgmpy_true_cpt =convert_to_pgmy_cpt(true_cpt, nodes)#incorporate this cpt into pgmpy graph
					for row in pgmpy_true_cpt:#add cpd in pgmpy graph
						pgmpy_graph.add_cpds(row)
					reporter_set =set()
					for  r in range(no_reporter_node_shifts):
						print 'Current reporter shift iteration %d out of %d' %(r, no_reporter_node_shifts)
						#pick a reporter node
						if reporter_set:#if a node has been picked before for this graph and cpt, select another
							while 1:
								reporter_node =np.random.choice(nodes)
								if reporter_node not in reporter_set:
									reporter_set.add(reporter_node)
									break
						else:
							reporter_node =np.random.choice(nodes)
							reporter_set.add(reporter_node)
						reporter_state =np.random.randint(2)#pick the reporter state
						print 'Current reporter node is %d in state %d' %(reporter_node, reporter_state)
						for no_intervention in range(1, max_intervention_pts+1):#compute inference values for each count of intervention pts
							if no_intervention ==1:#compute the values these form are ground truth
								print 'Computing Utility values when 1 intervention point is selected'
								probs, evidences =variable_elimination(pgmpy_graph, reporter_node, no_intervention, nodes)
								probs =np.array(probs)
								probs =probs[:, reporter_state]#only pick the probs for the chosen reporter state
								probs =np.round(probs, 4)
								utility_1 =utility1_values(probs, evidences)
								utility_2, utility_3 =utility_2_3_values(probs, evidences, pgmpy_graph)
							else:
								for u, u_values in {'utility_1':utility_1, 'utility_2':utility_2, 'utility_3':utility_3}.items():
									for comp in ['add', 'multiply']:
										print 'Finding top 20 sets of size --%d-- using single intervention for utility type --%s-- using --%s--'%(no_intervention, u, comp)
										filtered_comb_utility =find_top_utility(u_values, no_intervention, top=20, computation=comp)
										filtered_evidences =filtered_comb_utility.keys()
										probs, evidences =variable_elimination_filtered_set(pgmpy_graph, reporter_node, no_intervention, nodes, filtered_evidences)
										probs =np.array(probs)
										probs =probs[:, reporter_state]#extract only the values for the reporter state
										probs =np.round(probs, 4)
										spearman_coeff, kendall_coeff= compute_corr_ordering(probs, evidences, no_intervention,filtered_comb_utility)
										observations[(node_count, no_intervention, u, comp, 'spearman')].append(spearman_coeff)
										observations[(node_count, no_intervention, u, comp, 'kendall')].append(kendall_coeff)
		output =[]
		header =['node_count', 'no_intervention_pts','utility_type', 'calculation_method', 'metric', 'average_value', 'single_no']
		for key, observation in observations.items():
			tmp =[]
			tmp.extend(key)
			avg =np.mean([0 if np.isnan(o[0]) else o[0]for o in observation])
			one_no =np.tan(np.mean([0 if np.isnan(o[0]) else np.tanh(o[0]) for o in observation]))
			tmp.extend([avg,one_no])
			output.append(tmp)
		with open('output.csv', 'w') as f:
			writer =csv.writer(f)
			writer.writerow(header)
			writer.writerows(output)
			
