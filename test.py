import numpy as np

L_Matrix=np.zeros((5,5))
R2_size=4
R1_size=4
row=4
col=1 
while col<=R2_size:
	temp_row=row-1
	temp_col=col-1
	while temp_col >= 0 and temp_row >=0 :
		L_Matrix[temp_row][temp_col]=L_Matrix[temp_row+1][temp_col+1]+1
		temp_col=temp_col-1
		temp_row=temp_row-1
		print L_Matrix
	col=col+1
row=R1_size-1
col=R2_size
while row >0 :                                 
	temp_row=row-1
	temp_col=col-1
	while temp_col >= 0 and temp_row >=0 :
		L_Matrix[temp_row][temp_col]=L_Matrix[temp_row+1][temp_col+1]+1
		temp_col=temp_col-1
		temp_row=temp_row-1
		print L_Matrix
	row=row-1
print L_Matrix
# return L_Matrix
