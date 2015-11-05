import math
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
import numpy as np

# 
lambda1= 0.5
class node(object):
	def __init__(self, Type, Text = "",Role=None,child=None,parent=None):
		self.Type = Type.upper()
		self.Text = Text.upper()
		self.size=0
		if Role:
			self.Role = Role.upper()
		else:
			self.Role=None
		if child:
			self.child=child
			self.size=len(child)
		else:
		   	self.child=None
		if parent:
			self.parent=parent
		else:
			self.parent=None
	def __str__(self):
		return self.Text


class relation(object):
	def __init__(self, first, child, isRoot = False):
		self.first = first
		self.child = child
		self.isRoot = isRoot
		self.parent = None

	def insertChild(self, child):
		self.child = (self.child).append(child)

	def size(self):
		return len(self.child)

	def setMyParent(self, parent):
		self.parent = parent

	def setChildParent(self):
		for it in range(len(self.child)):
			(self.child)[it].setMyParent(self)

def t(R1,R2):
	if R1.Type==R2.Type and R1.Role == R2.Role:
		return 1
	else:
		return 0

def k(R1,R2):
	if R1.Text == R2.Text:
		return 1
	else:  
		return 0

def K(R1,R2):
	if t(R1,R2)==0:
		print "Type and Role dont match"
		return 0
	else:
		return k(R1,R2)+Kc(R1,R2)

def L(R1,R2):
	R1_size=R1.size
	R2_size=R2.size
	L_Matrix=np.zeros((R1_size+1,R2_size+1))
	row=R1_size
	col=1 
	while col<=R2_size:
		temp_row=row-1
		temp_col=col-1
		while temp_col >= 0 and temp_row >=0 :
			if t(R1.child[temp_row],R2.child[temp_col])==0:
				L_Matrix[temp_row][temp_col]=0
			else:
			 	L_Matrix[temp_row][temp_col]=L_Matrix[temp_row+1][temp_col+1]+1
			temp_col=temp_col-1
			temp_row=temp_row-1
		col=col+1
	row=R1_size-1
	col=R2_size
	while row >0 :                                 
		temp_row=row-1
		temp_col=col-1
		while temp_col >= 0 and temp_row >=0 :
			if t(R1.child[temp_row],R2.child[temp_col])==0:
				L_Matrix[temp_row][temp_col]=0
			else:
				L_Matrix[temp_row][temp_col]=L_Matrix[temp_row+1][temp_col+1]+1
			temp_col=temp_col-1
			temp_row=temp_row-1
		row=row-1
	print L_Matrix
	return L_Matrix

def C(R1,R2,L_Matrix):
	R1_size=R1.size
	R2_size=R2.size
	#L_Matrix=L(R1,R2)
	C_Matrix=np.zeros((R1_size+1,R2_size+1))
	row=R1_size
	col=1
	while col<=R2_size:
		temp_row=row-1
		temp_col=col-1
		while temp_col >= 0 and temp_row >=0 :
			if t(R1.child[temp_row],R2.child[temp_col])==0:
				C_Matrix[temp_row][temp_col]=0        
			else:
				C_Matrix[temp_row][temp_col]=(lambda1*((1-pow(lambda1,L_Matrix[temp_row][temp_col]))/(float)(1-lambda1))*K(R1.child[temp_row],R2.child[temp_col])+lambda1*C_Matrix[temp_row+1][temp_col+1])
			temp_col=temp_col-1
			temp_row=temp_row-1
		col=col+1
	row=R1_size-1
	col=R2_size
	while row >0 :
		temp_row=row-1
		temp_col=col-1
		while temp_col >= 0 and temp_row >=0 :
			if t(R1.child[temp_row],R2.child[temp_col])==0:
				C_Matrix[temp_row][temp_col]=0
			else:
				C_Matrix[temp_row][temp_col]=lambda1*((1-pow(lambda1,L_Matrix[temp_row][temp_col]))/(1-lambda1))*K(R1.child[temp_row],R2.child[temp_col])+lambda1*C_Matrix[temp_row+1][temp_col+1]
			temp_col=temp_col-1
			temp_row=temp_row-1
		row=row-1
	return C_Matrix

def Kc(R1,R2):
	L_Matrix=L(R1,R2)
	C_Matrix=C(R1,R2,L_Matrix)
	R1_size=R1.size
	R2_size=R2.size
	print "R1,",(R1_size)
	print "R2,",(R2_size)
	Kcc=0
	for i in range(0,R1_size+1):
		for j in range(0,R2_size+1):
			Kcc=Kcc+C_Matrix[i][j]
	return Kcc

def data_preprocess():
	return_list=[]
	with open('input.txt') as f:
		for line  in f:
			tokenizer = RegexpTokenizer(r'\w+')
			line=tokenizer.tokenize(line)
			tagged_line=nltk.pos_tag(line)
			final_tagged_line=[]
			i=0
			while i < len(tagged_line):
				substr=tagged_line[i][0]
				while i<len(tagged_line)-1 and tagged_line[i][1]==tagged_line[i+1][1]:
					substr=substr+" "+tagged_line[i+1][0]
					i=i+1
				final_tagged_line.append((substr,tagged_line[i][1]))
				i=i+1
			return_list.append(final_tagged_line)
	return return_list


def print_iterate(root):
	if root == None:
		return
	print root.Type,root.Text,root.Role
	if root.child!=None:
		for i in range(0,len(root.child)):
			print_iterate(root.child[i])




if __name__=="__main__":
	tagged_sentences = data_preprocess()
	print tagged_sentences
	train=[]
	for i in range(0,len(tagged_sentences)):
		
		tagged=tagged_sentences[i]
		grammar = "NODE1: {(<NN>|<NNP>)+<IN><DT>?(<NN>|<NNP>)+}"  
		cp = nltk.RegexpParser(grammar)                               
		result1 = cp.parse(tagged)
		print result1
		grammar = "NODE1: {(<DT><VBZ>)?(<NNP>|<NN>)+(<VBD>|<VBZ>|<VBG>|<VB>)+<IN>?<DT>?<JJ>?(<NN>|<NNP>)+}"  
		cp = nltk.RegexpParser(grammar)
		result2 = cp.parse(tagged)
		print result2


		root_node=[]
		for sub in result1.subtrees():
			if sub.label() == "NODE1":
				root_node.append(sub)
		root_node=root_node[0]
		nodes=[]
		relation=[]
		root=node(root_node[0][1],root_node[0][0],"Member")
		for i in range(0,len(root_node)):
			element=root_node[i]
			nodes.append(node(element[1],element[0],None,None,root))
		nodes[len(nodes)-1].Role="AFFILATION"
		root.child=nodes
		root.size=len(nodes)
		train.append(root)
		


		root_node=[]
		for sub in result1.subtrees():
			if sub.label() == "NODE1":
				root_node.append(sub)
		root_node=root_node[0]
		nodes=[]
		relation=[]
		root=node(root_node[0][1],root_node[0][0])
		for i in range(0,len(root_node)):
			element=root_node[i]
			nodes.append(node(element[1],element[0],None,None,root))
		nodes[len(nodes)-1].Role="AFFILATION"
		nodes[0].Role="MEMBER"
		root.child=nodes
		root.size=len(nodes)                                                                                                                           
		train.append(root)



		root_node=[]
		for sub in result1.subtrees():
			if sub.label() == "NODE1":
				root_node.append(sub)
		root_node=root_node[0]
		nodes=[]     
		relation=[]
		for i in range(0,len(root_node)):
			element=root_node[i]
			nodes.append(node(element[1],element[0]))
		nodes[len(nodes)-1].Role="AFFILATION"
		root_node=[]
		for sub in result2.subtrees():
			if sub.label() == "NODE1":
				root_node.append(sub)
		root_node=root_node[0]
		nodes2=[]
		relation=[]
		for i in range(0,len(root_node)):
			element=root_node[i]
			nodes2.append(node(element[1],element[0]))
		nodes2[0].Role="MEMBER"	
		nodes2[len(nodes2)-1].child=nodes
		for i in range(0,len(nodes)):
			nodes[i].parent=nodes2[len(nodes2)-1]

		root=node("sentence")
		root.child=nodes2
		root.size=len(nodes2)
		for i in range(0,len(nodes2)):
			nodes2[i].parent=root
		train.append(root)
	Kernel=np.zeros((len(train),len(train)))
	Kernel_normalised=np.zeros((len(train),len(train)))
	for i in range(0,len(train)):
	 	for j in range(0,len(train)):
			Kernel[i][j]=K(train[i],train[j])
	for i in range(0,len(train)):
	 	for j in range(0,len(train)):
			Kernel_normalised[i][j]=Kernel[i][j]/math.sqrt(((float)(Kernel[i][i])*Kernel[j][j]))
	print Kernel_normalised                   
	# print_iterate(train[2])
	# print_iterate(train[5])
	# print K(train[5],train[5])
	# for i in range(0,len(t rain)):
	# 	print_iterate(train[i])      

	# 	nodes=[]
	# 	child_nodes=[]
	# 	sentence=tagged_sentences[i]
	# 	j=0
	# 	while j < len(sentence) and sentence[j][1] != 'NNP' and sentence[j][1] == 'NN':
	# 		j=j+1
	# 	k=j-1
	# 	while( k >= 0 and sentence[k][1] != 'DT'):
	# 		k=k-1
	# 	if k >=0 and sentence[k][1] == 'DT':
	# 		while( k <= j ):
	# 			tup=sentence[k]
	# 			temp=node(sentence[k][1],sentence[k][0])
	# 			nodes.append(temp)
	# 			k=k+1
	# 	for k in range(0,j+1):
	# 		tup=sentence[k]
	# 		temp=node(sentence[k][1],sentence[k][0])
	# 		nodes.append(temp)
	# 	relation_list=[]
	# 	for k in range(0,j+1):
	# 		relation_list.append(relation(nodes[k],[]))
	# 	temp=node(sentence[j][1],sentence[j][0])
	# 	temp_tree=relation(temp,relation_list)
	# 	temp_tree.setChildParent()
	# 	nodes2=[]
	# 	for k in range(j,len(sentence)):
	# 		if sentence[k][1] == 'NNP':
	# 			break
	# 	if k != len(sentence):
	# 		for i in range(j+1,k+1):
	# 			temp2=node(sentence[i][1],sentence[i][0])
	# 			nodes2.append(temp2)
	# 	new_node=node(sentence[j][1],sentence[j][0])
	# 	relation_list=[]
	# 	relation_list.append(temp_tree)
	# 	for k in range(0,len(nodes2)):
	# 		rel_temp=relation(nodes2[k],[])
	# 		relation_list.append(rel_temp)
	# 	new_node_rel=relation(new_node,relation_list)
	# 	new_node_rel.setChildParent()
	# 	nodes_top=[]
	# 	for k in range(0,j):
	# 		node3=node(sentence[k][1],sentence[k][0])
	# 		nodes_top.append(node3)
	# 	relation_list=[]
	# 	for k in range(0,len(nodes_top)):
	# 		rel_temp=relation(nodes_top[k],[])
	# 		relation_list.append(rel_temp)
	# 	relation_list.append(new_node_rel)
	# 	P1_rel=relation(P1,relation_list,True)
	# 	P1_rel.setChildParent()
	# 	child=P1_rel.child
		# for i in range(0,len(child)):
		# 	print child[i].first.Text
