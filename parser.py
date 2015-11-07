import math
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
import numpy as np
from sklearn import svm
import numpy as np
from nltk.stem.porter import *

# 
lambda1= 0.5
NN_list=['NN','NNS','NNP','NNPS']
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
		#print "Type and Role dont match"
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
	#print L_Matrix
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
	#print "R1,",(R1_size)
	#print "R2,",(R2_size)
	Kcc=0
	for i in range(0,R1_size+1):
		for j in range(0,R2_size+1):
			Kcc=Kcc+C_Matrix[i][j]
	return Kcc


def compare(tag1,tag2):
	NN_list=['NN','NNS','NNP','NNPS']
	VB_list=['VB','VBS','VBZ','VBD','VBN']
	if tag1 in NN_list and tag2 in NN_list:
		return True
	if tag1 in VB_list and tag2 in VB_list:
		return True


import pickle
test=[]
def data_preprocess():
	return_list=[]
	label_list=[]
	favorite_color = pickle.load( open( "data_positive.pickle", "rb" ) )
	#with open('input.txt') as f:
	for i in range(0,min(len(favorite_color),50)):
		line=favorite_color[i]['sentence']
		try:
			line = line.encode("ascii")
			test.append(favorite_color[i]['organisation'].encode("ascii"))
		except:
			continue
		tokenizer = RegexpTokenizer(r'\w+')
		line2=line
		line=tokenizer.tokenize(line2)
		tagged_line=nltk.pos_tag(line)
		final_tagged_line=[]
		i=0
		while i < len(tagged_line):
			substr=tagged_line[i][0]
			while i<len(tagged_line)-1 and ((tagged_line[i][1]==tagged_line[i+1][1]) or compare(tagged_line[i][1],tagged_line[i+1][1])):
				substr=substr+" "+tagged_line[i+1][0]
				i=i+1
			final_tagged_line.append((substr,tagged_line[i][1]))
			i=i+1
		return_list.append(final_tagged_line)
		label_list.append(1)
	favorite_color = pickle.load( open( "data_negative.pickle", "rb" ) )
	#with open('input.txt') as f:
	for i in range(0,min(len(favorite_color),50)):
		line=favorite_color[i]['sentence']
		try:
			line = line.encode("ascii")
			test.append(favorite_color[i]['organisation'].encode("ascii"))
		except:
			continue
		tokenizer = RegexpTokenizer(r'\w+')
		line2=line
		line=tokenizer.tokenize(line2)
		tagged_line=nltk.pos_tag(line)
		final_tagged_line=[]
		i=0
		while i < len(tagged_line):
			substr=tagged_line[i][0]
			while i<len(tagged_line)-1 and ((tagged_line[i][1]==tagged_line[i+1][1]) or compare(tagged_line[i][1],tagged_line[i+1][1])):
				substr=substr+" "+tagged_line[i+1][0]
				i=i+1
			final_tagged_line.append((substr,tagged_line[i][1]))
			i=i+1
		return_list.append(final_tagged_line)
		label_list.append(-1)
	# with open('test.txt') as f:
	# 	for line in f:
	# 		line=line.strip()
	# 		tokenizer = RegexpTokenizer(r'\w+')
	# 		line2=line.replace("of","")
	# 		line=tokenizer.tokenize(line2)
	# 		tagged_line=nltk.pos_tag(line)
	# 		final_tagged_line=[]
	# 		i=0
	# 		while i < len(tagged_line):
	# 			substr=tagged_line[i][0]
	# 			while i<len(tagged_line)-1 and ((tagged_line[i][1]==tagged_line[i+1][1]) or compare(tagged_line[i][1],tagged_line[i+1][1])):
	# 				substr=substr+" "+tagged_line[i+1][0]
	# 				i=i+1
	# 			final_tagged_line.append((substr,tagged_line[i][1]))
	# 			i=i+1
	# 		return_list.append(final_tagged_line)

	return (return_list,label_list)


def print_iterate(root):
	if root == None:
		return
	print root.Type,root.Text,root.Role
	if root.child!=None:
		for i in range(0,len(root.child)):
			print_iterate(root.child[i])



def tree_creation(tagged_sentences,label_list=[]):
	train=[]
	train_label=[]
	for i in range(0,len(tagged_sentences)):
		tagged=tagged_sentences[i]
		#print tagged
		grammar = "NODE1: {(<NN>|<NNP>|<NNS>|<PRP>)+<RB>?(<VBD>|<VBN>|<VBZ>|<VBG>|<VB>)?<IN><DT>?(<NN>|<NNP>)+}"  
		cp = nltk.RegexpParser(grammar)
		result1 = cp.parse(tagged)
		print result1
		grammar = "NODE1: {(<DT><VBZ>)?(<NNP>|<NN>|<NNS>|<NNPS>)+(<JJ>)?(<VBD>|<VBZ>|<VBG>|<VB>)+<IN>?<DT>?<JJ>?(<NN>|<NNP>)+}"  
		cp = nltk.RegexpParser(grammar)
		result2 = cp.parse(tagged)
		print result2
		try:
			root_node=[]
			for sub in result1.subtrees():
				if sub.label() == "NODE1":
					root_node.append(sub)
			root_node=root_node[len(root_node)-1]
			nodes=[]
			relation=[]
			root=node(root_node[0][1],root_node[0][0],"Member")
			for j in range(0,len(root_node)):
				element=root_node[j] 
				nodes.append(node(element[1],element[0],None,None,root))
			nodes[len(nodes)-1].Role="AFFILATION"
			root.child=nodes
			root.size=len(nodes)
			train.append(root)
			train_label.append(label_list[i])
		except:
			pass

		try:
			root_node=[]
			for sub in result1.subtrees():
				if sub.label() == "NODE1":
					root_node.append(sub)
			root_node=root_node[len(root_node)-1]
			nodes=[]
			relation=[]
			root=node(root_node[0][1],root_node[0][0])
			for j in range(0,len(root_node)):
				element=root_node[j]
				nodes.append(node(element[1],element[0],None,None,root))
			nodes[len(nodes)-1].Role="AFFILATION"
			nodes[0].Role="MEMBER"
			root.child=nodes
			root.size=len(nodes)                                                                                                                           
			train.append(root)
			train_label.append(label_list[i])
		except:
			pass

		try:
			root_node=[]
			for sub in result1.subtrees():
				if sub.label() == "NODE1":
					root_node.append(sub)
			root_node=root_node[len(root_node)-1]
			nodes=[]     
			relation=[]
			for j in range(0,len(root_node)):
				element=root_node[j]
				nodes.append(node(element[1],element[0]))
			nodes[len(nodes)-1].Role="AFFILATION"
		except:
			pass
		try:
			root_node=[]
			for sub in result2.subtrees():
				if sub.label() == "NODE1":
					root_node.append(sub)
			root_node=root_node[0]
			nodes2=[]
			relation=[]
			for j in range(0,len(root_node)):
				element=root_node[j]
				nodes2.append(node(element[1],element[0]))
			nodes2[0].Role="MEMBER"	
			nodes2[len(nodes2)-1].child=nodes
			for j in range(0,len(nodes)):
				nodes[j].parent=nodes2[len(nodes2)-1]
		except:
			pass
		try:
			root=node("sentence")
			root.child=nodes2
			root.size=len(nodes2)
			for j in range(0,len(nodes2)):
				nodes2[j].parent=root
		except:
			pass
		train.append(root)
		try:
			train_label.append(label_list[i])
		except:
			pass
	return train,train_label

if __name__=="__main__":
	tagged_sentences,label_list = data_preprocess()
	print "data_preprocess has been done"
	
	#print tagged_sentences
	train=[]
	train_label=[]
	train,train_label=tree_creation(tagged_sentences,label_list)
	Kernel=np.zeros((len(train),len(train)))
	Kernel_normalised=np.zeros((len(train),len(train)))
	for i in range(0,len(train)):
	 	for j in range(0,len(train)):
			Kernel[i][j]=K(train[i],train[j])
	for i in range(0,len(train)):
	 	for j in range(0,len(train)):
			Kernel_normalised[i][j]=Kernel[i][j]/math.sqrt(((float)(Kernel[i][i])*Kernel[j][j]))
	final_tagged_l=[]
	clf = svm.SVC(kernel='precomputed')
	clf.fit(Kernel_normalised,train_label)
			
	with open('test.txt') as f:
		
		for line in f:
			final_tagged_l=[]
			line=line.strip()
			temp=line
			tokenizer = RegexpTokenizer(r'\w+')
			#line2=line.replace("of","")
			line2=line
			line=tokenizer.tokenize(line2)
			tagged_line=nltk.pos_tag(line)
			for jdx in range(0,len(tagged_line)-1):
				if tagged_line[jdx][0]=='of' and tagged_line[jdx+1][1] in NN_list and tagged_line[jdx-1][1] in NN_list:
					line2=temp.replace("of","")
					line=tokenizer.tokenize(line2)
			tagged_line=nltk.pos_tag(line)
			i=0
			final_tagged_line=[]
			while i < len(tagged_line):
				substr=tagged_line[i][0]
				while i<len(tagged_line)-1 and ((tagged_line[i][1]==tagged_line[i+1][1]) or compare(tagged_line[i][1],tagged_line[i+1][1])):
					substr=substr+" "+tagged_line[i+1][0]
					i=i+1
				final_tagged_line.append((substr,tagged_line[i][1]))
				i=i+1
			final_tagged_l.append(final_tagged_line)
			test,test_label=tree_creation(final_tagged_l)

			# for jdx in range(0,len(test)):
			# 	print_iterate(test[jdx])
			# print K(test[2],test[2])
			Kernel_test=np.zeros((len(test),len(train)))
			for i in range(0,len(test)):
			 	for j in range(0,len(train)):
					Kernel_test[i][j]=K(test[i],train[j])
			
	# for i in range(0,len(test)):
	#  	for j in range(0,len(train)):
	# 		Kernel_test[i][j]=Kernel[i][j]/math.sqrt(((float)(Kernel[i][i])*Kernel[j][j]))
	# print len(train_label),len(Kernel_normalised),len(Kernel_normalised[0])
	# print train_label
			y_pred = clf.predict(Kernel_test)
			print y_pred
			res=sum(y_pred)
			if res >0:
				sent=test[len(test)-1]
				if not sent.child:
					try:
						print_iterate(test[0])
					except:
						print_iterate(sent)
				else:
					print_iterate(test[len(test)-1])
				print "---------------treee-----------------------------"
			else:
				print "Relation not found"
				print "---------------notree----------------------------"
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
