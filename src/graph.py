from collections import defaultdict
from collections import Counter
from copy import deepcopy
class Graph:
#take in the vertices and the edges and build the graph
	def __init__(self, nodes, connections, directed=True):
		self.graph =defaultdict(set)#initialize the graph
		self.directed =directed
		self.nodes =nodes
		self.build_graph(connections)
		self.cyclic =self.is_cyclic()
	def build_graph(self, connections):
		
		#initialize the nodes in the graph
		for node in self.nodes:
			self.graph[node]
		
		#each node knows its parents
		for connection in connections:
			node1, node2 =connection
			
			if self.directed:
				self.graph[node2].add(node1) #node1(parent) points to node2(child) 

	def is_cyclic(self):
		#topological sort wikipedia
		#S set of all nodes with no incoming edges
		#L Empty list containing sorted elements
		graph_copy =deepcopy(self.graph)
		S=set([i for i, j in graph_copy.items() if j ==set()])
		L=[]
		while S:
			out =S.pop()
			L.append(out)
			for k, l in graph_copy.items():
				if out in l:
					l.remove(out)
					if l ==set():
						S.add(k)
		if sum([1 for i, j in graph_copy.items() if j !=set()]) >0:#edges left
			return True
		else:#no edges left
			#print L
			return False

