from config.settings import *
from graphviz import Digraph

def make_graph(graph, comment, name='MAPK.jpg'):
	if name =='MAPK.jpg':
		node_name_dict ={1:'WRKY22', 2:'MPK4', 3:'FLS2', 4:'MKK2', 5:'PR1', 6:'MEK1', 7:'MKK4', 8:'MKK3', 9:'WRKY25'}
		dot =Digraph(comment)
		for i, j in  node_name_dict.items():
			dot.node(str(i), str(j))
	else:
		dot =Digraph(comment)
		for i in  graph.keys():
			dot.node(str(i), str(i))
		edges =[]
		for i, j in graph.items():
			for k in j:
				edges.append([str(k), str(i)])
		#edges =[str(k)+ str(i) for i, j in graph.items() for k in j]
		print edges
		dot.edges(edges)
	dot.render(name, view=False)

if __name__ == '__main__':
	#the graphs are hard coded at the moment
	graph_1 ={1: set([]), 2: set([9]), 3: set([2]), 4: set([6]), 5: set([2]), 6: set([1]), 7: set([1]), 8: set([]), 9: set([])}
	comment_1='Inferred BN with max number of parents =1'
	graph_2 ={1: set([]), 2: set([8, 7]), 3: set([6]), 4: set([]), 5: set([4]), 6: set([2, 5]), 7: set([1, 5]), 8: set([]), 9: set([8, 7])}
	comment_2='Inferred BN with max number of parents =2'
	graph_3 ={1: set([3]), 2: set([1, 7, 9]), 3: set([]), 4: set([6]), 5: set([6]), 6: set([3]), 7: set([3, 5]), 8: set([4, 5]), 9: set([7])}
	comment_3='Inferred BN with max number of parents =3'
	graph_4 ={1: set([8]), 2: set([4, 6]), 3: set([]), 4: set([6]), 5: set([1, 9]), 6: set([3]), 7: set([6]), 8: set([4, 7]), 9: set([2])}
	graph_5 ={0: set([]), 1: set([8, 10]), 2: set([17]), 3: set([0, 16, 14, 6, 17]), 4: set([10, 5]), 5: set([16]), 6: set([8, 9]), 7: set([18, 13, 15]), 8: set([15]), 9: set([12, 7]), 10: set([11, 14]), 11: set([6]), 12: set([0]), 13: set([8, 19]), 14: set([18]), 15: set([12]), 16: set([10]), 17: set([16]), 18: set([8]), 19: set([14])}

	comment_4='MAPK kegg'
	#make_graph(graph_1, comment_1)
	#make_graph(graph_2, comment_2)
	make_graph(graph_5, 'comment_4', '12.jpg')

