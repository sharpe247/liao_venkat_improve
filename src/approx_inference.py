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
from pgmpy.inference.EliminationOrder import WeightedMinFill
from pgmpy.factors.discrete import State
from scipy import stats
import csv
import heapq
import argparse
import numpy as np
import src.figures as figures
import pandas as pd
import networkx as nx
import cPickle as pickle

#cpt functions
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
			prob_0 =round(np.random.random(),4) #these sum to one
			if prob_0 ==0:
				prob_0 =0.0001
			prob_1 =round(1-prob_0, 4)
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

def normal(sd):
	return np.random.normal(loc=0 , scale=sd)			
#inference methods
def variable_elimination(model, reporter_node, no_evidence, nodes):
	'''
	exact inference using variable elimination
	elimination order is fetched using the pgmpy library
	'''
	elimination_order =WeightedMinFill(model).get_elimination_order(model.nodes())
	#print elimination_order
	#print min_fill_order(model)
	infer =VariableElimination(model)
	other_nodes =[str(i) for i in nodes if i!= reporter_node]
	evidences =[]
	probs =[]
	for tup in combinations(other_nodes, no_evidence):
		done =0
		for val in product([0,1], repeat=no_evidence):
			#print 'single prob infered %d' %done
			evidence ={i:j for i, j in zip(tup, val)}
			evidences.append(evidence)
			#print 'evidence is ', evidence
			#print [i for i in elimination_order if i not in evidence and i !=str(reporter_node)]
			inf_prob =infer.query([str(reporter_node)], evidence=evidence,elimination_order =[i for i in elimination_order if i not in evidence and i !=str(reporter_node)])[str(reporter_node)].values
			probs.append(inf_prob)
			#print 'reporter state 0 has probability', inf_prob[0]
			#print 'reporter state 1 has probability', inf_prob[1]
			done +=1
	return probs, evidences

def likelihood_sampling(model, reporter_node, no_evidence, nodes, likelihood_samples=1000):
	'''
	inference using likelihood sampling
	'''
	infer =BayesianModelSampling(model)
	other_nodes =[str(i) for i in nodes if i!= reporter_node]
	evidences =[]
	probs =[]
	for tup in combinations(other_nodes, no_evidence):
		#done =0
		for val in product([0,1], repeat=no_evidence):
			#print 'single prob infered %d' %done
			evidence ={i:j for i, j in zip(tup, val)}
			#print evidence
			state_evidence =[State(i,j) for i, j in zip(tup, val)]
			evidences.append(evidence)
			inf_samples =infer.likelihood_weighted_sample(evidence=state_evidence, size=likelihood_samples, return_type='recarray')
			#print inf_samples
			var, vals =[], []
			for ev in state_evidence:
				var.append(ev[0])
				vals.append(ev[1])
			prob_0_num, prob_1_num, den =0,0,0
			done =0
			for sample in inf_samples:
				#print sample
				check =0
				check =sum([1 for idx, v in enumerate(var) if sample[v] ==vals[idx]])
				if check ==len(var):
					done +=1
					den +=sample[-1] #the weight
					if sample[str(reporter_node)] ==1:
						prob_1_num +=sample[-1]
					else:
						prob_0_num +=sample[-1]
				#break
			if den ==0:
				#print 'check'
				#print done, den, prob_0_num, prob_1_num, prob_0, prob_1
				#print sample
				prob_0, prob_1 =0,0
			else:
				prob_0, prob_1 =float(prob_0_num)/den, float(prob_1_num)/den
				#print done, den, prob_0_num, prob_1_num, prob_0, prob_1
			probs.append([prob_0, prob_1])
			#break
	return probs, evidences




def variable_elimination_filtered_set(model, reporter_node, no_intervention, nodes, filtered_evidences):
	'''
	filtered_evidences is a list of tuples of evidences
	'''
	elimination_order =WeightedMinFill(model).get_elimination_order(model.nodes())
	infer =VariableElimination(model)
	other_nodes =[str(i) for i in nodes if i!= reporter_node]
	evidences =[]
	probs =[]
	for tups in filtered_evidences:
		evidence ={}
		for tup in tups:
			evidence[tup[0]] =tup[1]
		evidences.append(evidence)
		inf_prob =infer.query([str(reporter_node)], evidence=evidence,elimination_order=[i for i in elimination_order if i not in evidence and i !=str(reporter_node)])[str(reporter_node)].values
		probs.append(inf_prob)
	return probs, filtered_evidences
def likelihood_sampling_filtered_set(model, reporter_node, no_intervention, nodes, filtered_evidences, likelihood_samples=1000):
	'''
	filtered_evidences is a list of tuples of evidences
	'''
	infer =BayesianModelSampling(model)
	other_nodes =[str(i) for i in nodes if i!= reporter_node]
	evidences =[]
	probs =[]
	for tups in filtered_evidences:
		state_evidence =[State(tup[0], tup[1]) for tup in tups]
		evidence ={}
		for tup in tups:
			evidence[tup[0]] =tup[1]
		evidences.append(evidence)
		inf_samples =infer.likelihood_weighted_sample(evidence=state_evidence, size=likelihood_samples, return_type='recarray')
		var, vals =[], []
		for ev in state_evidence:
			var.append(ev[0])
			vals.append(ev[1])
		prob_0_num, prob_1_num, den =0,0,0
		for sample in inf_samples:
			check =0
			check =sum([1 for idx, v in enumerate(var) if sample[v] ==vals[idx]])
			if check ==len(var):
				den +=sample[-1]
				if sample[str(reporter_node)] ==1:
					prob_1_num +=sample[-1]
				else:
					prob_0_num +=sample[-1]
		if den ==0:
			prob_0, prob_1 =0,0
		else:
			prob_0, prob_1 =float(prob_0_num)/den, float(prob_1_num)/den	
		probs.append([prob_0, prob_1])
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
#random graph generation
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
	'''
	using networkx random graph creater
        no of nodes is s, assuming max no of parents as 4
        and min as 1 so sampling for 10 nodes a no b/w [10, 40] as the no of edges on 
        average we expect 25 edges
	'''
	count =0
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
	'''
	convert the adjacency matrix to connections or adj list
	'''
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

def utility_2_3_values(probs, evidences, model, likelihood=False, likelihood_samples=1000):
	'''
	utility2 P(R=1|X=x). P(X=x), use this probability computation for utility--single intervention
	utiltiy3 P(R=1|X=x).(1-P(X=x)), use this probability computation for utility--single intervention
	'''
	if likelihood:
		infer =BayesianModelSampling(model)
	else:
		infer =VariableElimination(model)
	utility2, utility3 ={}, {}
	for prob, evidence in zip(probs, evidences):
		evidence_tup =evidence.items()[0]#there is only one evidence
		if likelihood:
			#since in this case we have to compute the marginal, there is no evidence while sampling
			inf_samples =infer.likelihood_weighted_sample(evidence=[], size=likelihood_samples, return_type='recarray')
			num, den =0,0
			for sample in inf_samples:
				den +=sample[-1]
				if sample[int(evidence_tup[0]) -1] ==evidence_tup[1]:
					num +=sample[-1]
                	if den ==0:
                        	marginal_prob =0
              		else:
                        	marginal_prob =float(num)/den
		else:
			marginal_prob =infer.query([str(evidence.keys()[0])])[str(evidence.keys()[0])].values[evidence.values()[0]]
		utility2[evidence_tup] =prob * marginal_prob
		utility3[evidence_tup] =prob * (1-marginal_prob)
	return utility2, utility3


#ordering comparison functions
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

def min_fill_order(model):
	'''
	min fill order heuristic
	'''
	nodes =model.nodes()
	edges =model.edges()
	node_order ={}
	for node in nodes:
		prospective_nodes =[n for tup in edges if node in tup for n in tup if n !=node]
		node_order[node] =sum([1 for i in combinations(prospective_nodes,2) if i not in edges])
	order = [node for node, count in sorted(node_order.items(), key=lambda x: x[1])]
	return order


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--max_no_parents')
	parser.add_argument('--random_iterations')
	parser.add_argument('--no_nodes')
	parser.add_argument('--multi_hypothesis', action='store_true')
	parser.add_argument('--output_file')
	parser.add_argument('--max_nodes')
	parser.add_argument('--likelihood_sampling', action ='store_true')
	parser.add_argument('--likelihood_samples')
	parser.add_argument('--test_computation', action='store_true')
	parser.add_argument('--output_folder')
	args = parser.parse_args()


	if args.multi_hypothesis:
		no_nodes = [int(args.max_nodes)]
		#no_nodes =[i for i in range(5,51,5)]
		#no_nodes =[50]
		graphs_per_node_count =10
		cpts_per_graph =2
		max_intervention_pts =4
		no_reporter_node_shifts =3
		#reporter node shift per graph sqrt no_nodes
		observations =defaultdict(list)
		iteration =0
		iteration_output ={}
		for node_count in no_nodes:#iterationg over differnt node sizes
			#no_reporter_node_shifts =int(np.sqrt(node_count)) #no of reporter node shifts
			#no_reporter_node_shifts =3
			print 'Current node count ', node_count
			for g in range(graphs_per_node_count):#iterating for number of graphs per node size
				print 'Iteration %d out of %d'%(iteration, graphs_per_node_count*cpts_per_graph* no_reporter_node_shifts)
				print 'Current graph iteration %d out of %d' %(g, graphs_per_node_count)
				#mat =random_graph_adj_mat(node_count)#totally ordered adj matrix 
				#print mat
				#connections =adj_mat_to_connections(mat) #adj list for that mat parent to child
				connections =random_graph_networkx(node_count)
				#connections =[(7,8), (7,9), (8,9), (9,10), (9,1), (2,1), (3,1), (4,3), (2,1), (6,4), (6,5)]
				nodes =[i+1 for i in range(node_count)]#node indices incremented by 1 
				nodes_str =[str(i) for i in nodes]
				graph =Graph(nodes, connections)#generate graph
				#print 'Current graph structure'
				#print graph.graph
				#print connections
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
						iteration +=1 #this the id of the experiment
						iteration_output[(iteration, 'graph')] =graph
						iteration_output[(iteration, 'pgmpy_graph')] =pgmpy_graph
						iteration_output[(iteration, 'connections')] =connections
						iteration_output[(iteration, 'connections_str')] =connections_str
						iteration_output[(iteration, 'true_cpt')] =true_cpt
						iteration_output[(iteration, 'pgmpy_true_cpt')] =pgmpy_true_cpt
						iteration_output[(iteration, 'reporter_node')] =reporter_node
						iteration_output[(iteration, 'reporter_state')] =reporter_state
						for no_intervention in range(1, max_intervention_pts+1):#compute inference values for each count of intervention pts
							if no_intervention ==1:#compute the values these form are ground truth
								print 'Computing Utility values when 1 intervention point is selected'
								if args.likelihood_sampling:
									likelihood =True
									likelihood_samples =int(args.likelihood_samples)
									probs, evidences =likelihood_sampling(pgmpy_graph, reporter_node, no_intervention, nodes, likelihood_samples)
									iteration_output[(iteration, no_intervention, 'probs_sampling')] =probs
									iteration_output[(iteration, no_intervention, 'evidences_sampling')] =evidences
								else:
									likelihood =False
									probs, evidences =variable_elimination(pgmpy_graph, reporter_node, no_intervention, nodes)
									iteration_output[(iteration, no_intervention, 'probs_exact')] =probs
									iteration_output[(iteration, no_intervention, 'evidences_exact')] =evidences							
								probs =np.array(probs)
								probs =probs[:, reporter_state]#only pick the probs for the chosen reporter state
								probs =np.round(probs, 4)
								utility_1 =utility1_values(probs, evidences)
								utility_2, utility_3 =utility_2_3_values(probs, evidences, pgmpy_graph, likelihood)
							else:
								for u, u_values in {'utility_1':utility_1, 'utility_2':utility_2, 'utility_3':utility_3}.items():
									for comp in ['add', 'multiply']:
										print 'Finding top 20 sets of size --%d-- using single intervention for utility type --%s-- using --%s--'%(no_intervention, u, comp)
										filtered_comb_utility =find_top_utility(u_values, no_intervention, top=20, computation=comp)
										filtered_evidences =filtered_comb_utility.keys()
										if args.likelihood_sampling:
											likelihood =True
											likelihood_samples =int(args.likelihood_samples)
											probs, evidences =likelihood_sampling_filtered_set(pgmpy_graph, reporter_node, no_intervention, nodes, filtered_evidences, likelihood_samples)
											iteration_output[(iteration, no_intervention, 'evidences_sampling')] =evidences
											iteration_output[(iteration, no_intervention, 'probs_sampling')] =probs
										else:
											likelihood =False
											probs, evidences =variable_elimination_filtered_set(pgmpy_graph, reporter_node, no_intervention, nodes, filtered_evidences)
											iteration_output[(iteration, no_intervention, 'evidences_exact')] =evidences
											iteration_output[(iteration, no_intervention, 'probs_exact')] =probs
										probs =np.array(probs)
										probs =probs[:, reporter_state]#extract only the values for the reporter state
										probs =np.round(probs, 4)
										spearman_coeff, kendall_coeff= compute_corr_ordering(probs, evidences, no_intervention,filtered_comb_utility)
										observations[(iteration, node_count, no_intervention, u, comp, 'spearman')].append(spearman_coeff)
										observations[(iteration, node_count, no_intervention, u, comp, 'kendall')].append(kendall_coeff)
		output =[]
		header =['iteration','node_count', 'no_intervention_pts','utility_type', 'calculation_method', 'metric', 'average_value', 'single_no']
		for key, observation in observations.items():
			tmp =[]
			tmp.extend(key)
			avg =np.mean([0 if np.isnan(o[0]) else o[0]for o in observation])
			one_no =np.tan(np.mean([0 if np.isnan(o[0]) else np.tanh(o[0]) for o in observation]))
			tmp.extend([avg,one_no])
			output.append(tmp)
		with open(args.output_folder+'correlations.csv', 'w') as f:
			writer =csv.writer(f)
			writer.writerow(header)
			writer.writerows(output)
		with open(args.output_folder+'datastructures', 'wb') as f:
			pickle.dump(iteration_output, f)


	if args.test_computation:
		'''generate a few random graphs and test the inference computation using sampling and variable elimination
		Although this can be handled in the multihypothesis experiment, I think it more cleaner this way
		'''
		no_nodes = [int(args.max_nodes)]
		graphs_per_node_count =5
		cpts_per_graph =5
		max_intervention_pts =3
		no_reporter_node_shifts =3
		print('Experimental setup-------------------')
		print('No nodes in graph %d'%int(args.max_nodes))
		print('No randomly generated graphs %d'%graphs_per_node_count)
		print('No cpts per graphs %d'%cpts_per_graph)
		print('No of reporter node shifts %d'%no_reporter_node_shifts)
		print('Max no of intervention points %d'%max_intervention_pts)
		error_observations =defaultdict(list)
		iterations =0
		for node_count in no_nodes:#iterationg over differnt node sizes
			for g in range(graphs_per_node_count):#iterating for number of graphs per node size
				#print 'Current graph iteration %d out of %d' %(g, graphs_per_node_count)
				print 'Iteration %d'%iterations
				connections =random_graph_networkx(node_count)
				#connections =[(1, 2), (1, 3), (1, 5), (2, 5), (3, 4), (4, 5)]
				#connections =[(1, 2), (1, 3), (1, 4), (1, 5), (2, 3), (2, 4), (2, 5), (3, 4), (3, 5), (4, 5)]
				nodes =[i+1 for i in range(node_count)]#node indices incremented by 1 
				nodes_str =[str(i) for i in nodes]
				graph =Graph(nodes, connections)#generate graph
				#print 'Current graph structure'
				#print graph.graph
				#print connections
				connections_str =[(str(i), str(j)) for i,j in connections]#convert connections to str
				pgmpy_graph =BayesianModel()#initialize bn 
				pgmpy_graph.add_nodes_from(nodes_str)
				pgmpy_graph.add_edges_from(connections_str)
				for c in range(cpts_per_graph):#iterating for number of cpts per graph structure
					#print 'Current cpt iteration %d out %d' %(c, cpts_per_graph)
					true_cpt =random_binary_cpt(graph, nodes)#generate random cpt
					#print('true_cpt')
					#print (true_cpt)
					pgmpy_true_cpt =convert_to_pgmy_cpt(true_cpt, nodes)#incorporate this cpt into pgmpy graph
					#print('pgmpy_true_cpt')
					#print(pgmpy_true_cpt)
					for row in pgmpy_true_cpt:#add cpd in pgmpy graph
						pgmpy_graph.add_cpds(row)
					reporter_set =set()
					for  r in range(no_reporter_node_shifts):
						#print 'Current reporter shift iteration %d out of %d' %(r, no_reporter_node_shifts)
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
						#print 'Current reporter node is %d in state %d' %(reporter_node, reporter_state)
						iterations +=1
						for no_intervention in range(1, max_intervention_pts+1):#compute inference values for each count of intervention pts
							#print('running intervention pts =%d'%no_intervention)
							likelihood_samples =int(args.likelihood_samples)
							probs_sampling, evidences_sampling =likelihood_sampling(pgmpy_graph, reporter_node, no_intervention, nodes, likelihood_samples)
							probs_exact, evidences_exact =variable_elimination(pgmpy_graph, reporter_node, no_intervention, nodes)
							probs_sampling, probs_exact =np.array(probs_sampling), np.array(probs_exact)
							probs_sampling, probs_exact =probs_sampling[:, reporter_state], probs_exact[:, reporter_state]#only pick the probs for the chosen reporter state
							probs_sampling, probs_exact =np.round(probs_sampling, 4), np.round(probs_exact, 4)
							#for i, j,k in zip(evidences_exact, probs_exact, probs_sampling):
							#	print i,j,k
							sampling_evidence_dict ={tuple(i.items()):j for i, j in zip(evidences_sampling, probs_sampling)} 
							exact_evidence_dict ={tuple(i.items()):j for i, j in zip(evidences_exact, probs_exact)} 
							errors =[exact_evidence_dict[k] - v for k,v in sampling_evidence_dict.items()]#difference in the inference values
							error_observations[no_intervention].extend(errors)
		for no_pts, errs in error_observations.items():
			print('For %d intervention pt'%no_pts)
			print('Mean difference %f'%np.mean(errs))
			print('Max absolute difference %f'%np.amax(np.absolute(errs)))
			print('Std deviation  %f'%np.std(errs))
