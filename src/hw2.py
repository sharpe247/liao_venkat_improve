from src.graph import Graph
from config.settings import *
from copy import deepcopy
from itertools import combinations
from itertools import product
from collections import Counter
from collections import defaultdict
from Queue import Queue
from Queue import PriorityQueue
import csv
import heapq
import argparse
import numpy as np


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

def prob_cost(new_state, prev_state, cpt, total_prob, graph):
	new_node =new_state[0]
	old_node =prev_state[0]
	new_node_value =new_state[1]
	old_node_value =prev_state[1]
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
			
			
def best_path(graph, cpt, reporter_node, reporter_state, total_prob):
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
			alt =cost[next] +prob_cost(new_state, next, cpt, total_prob, graph)
			if alt < cost[new_state]:
				cost[new_state] =alt
				previous[new_state] =next
				heapq.heappush(vertex_set,(cost[new_state], new_state))

	for i, j in  cost.items():
		print i, np.exp(-1 *j)
	return paths(previous, cost)
	""""
	while not frontier.empty():
		#print frontier.queue
		next =frontier.get()[1]
		frontier_set.remove(next)
		moves =possible_results(graph, next)
		for move in moves:
			new_state =tuple(move)
			
			cost[new_state] =cost[next] + prob_cost(new_state, next, cpt, total_prob, graph)
			#print new_state, cost[new_state]
			if new_state not in frontier_set and new_state not in explored:
				
				#print new_state, next
				frontier_set.add(new_state)
				frontier.put((cost[new_state], new_state))
				
				#print new_state
				path[new_state].extend(next)#in the if clause to avoid loops
		explored.add(next)
		#print explored
	return paths(path, cost)
	#for i, j in  cost.items():
	#	print i, np.exp(-1 *j)
	#print path
	"""

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


def calculate_cpt(graph, nodes, X):
	cpt =[]
	#print graph.graph
	for i in nodes:
		parents =graph.graph[i]
		table =list(product([0, 1], repeat=len(parents)+1))#assume last is the node we are considering
		mask =list(deepcopy(parents))
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
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--ga_graph', action ='store_true')
	parser.add_argument('--fixed_graph', action ='store_true')
	parser.add_argument('--connections')
	parser.add_argument('--max_no_parents')
	parser.add_argument('--train_file')
	parser.add_argument('--test_file')
	parser.add_argument('--random_iterations')
	args = parser.parse_args()
	X, no_nodes =create_data(args.train_file) #training data
	nodes =[i+1 for i in range(no_nodes)]

	if args.ga_graph:
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
		meu(5, cpt, graph)

	if args.fixed_graph:
		#connections for venkat  (1,5) (9,5) (8,1) (2,9) (7,8) (4,8) (4,2) (6,2) (6,7) (6,4) (3,6)
		#to compare with venkat et all, fixing the graph
		connections =parse_connections(args.connections)
		graph =Graph(nodes, connections)# assuming that this fixed graph is not cyclic
		#print graph.graph
		cpt =calculate_cpt(graph, nodes, X)
		total_prob =total_prob(nodes, X)
		#print cpt
		best_path(graph, cpt, 5, 1, total_prob)
	#Y, _ =create_data(args.test_file)#testing data
	#y =predict(Y, cpt)
	#print "accuracy-----------"
	#print accuracy(y, Y)
