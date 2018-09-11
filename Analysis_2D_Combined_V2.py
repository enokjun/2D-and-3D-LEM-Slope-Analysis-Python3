'''
2D LEM Slope Analysis:
1. Ordinary Method of Slices
2. Modified Bishop Method 
3. Spencer Method (1967)
4. Morgenstern-Price Method (1965)
'''

'''
purpose: Compute 2D FS
Input: slice geometry, interslice function, shear strength,  external load, seismic load, water, tension crack, center point, ...
Output: FS, interslice scaling factor, interslice force angle

description of the input file - subjected to change

## Input file column legend for very first row 1 
0(A) - 2D or 3D analysis (2 = 2D; 3 = 3D)
1(B) - id of analysis (id links to a specific model)
2(C) - total number of slip surfaces

## Input file column legend for row 1 
0(A) - 2D or 3D analysis (2 = 2D; 3 = 3D)
1(B) - id of analysis (id links to a specific model)
2(C) - id of slip surface (id links to a particular slip surface)
3(D) - number of slices 
4(E) - Direction of slope movement (0 = right to left; 1 = left to right) 
5(F) - X coordinate of center of rotation (Pt0x)
6(G) - Y coordinate of center of rotation (Pt0y)
7(H) - seismic coefficient for horizontal force (k)
8(I) - resultant water forces - from tension crack or submersion (A)
9(J) - perpendicular distance from the resultant water force to the center of rotation (a)
10(K) - shear strength id (each id links to certain shear strength:
			1 = Mohr-Coulomb
			2 = undrained (Su)
			3 = stress-(c & phi') relationship 
			4 = stress-shear relationship
11(L) - L factor - janbu corrected
12(M) - d factor - janbu corrected
			
## Input file column legend for row 2 and below
0(A) - slice number
1(B) - slice horizontal width (b)
2(C) - slice base length (l)
3(D) - slice base angle from horizontal (alpha)
4(E) - slice top angle from horizontal (beta)
5(F) - radius (R) 
6(G) - perpendicular offset from center of rotation(f)
7(H) - horizontal distance from slice to the center of rotation (x)
8(I) - vertical distance from C.G. of slice to the center of rotation (e)
9(J) - slice total weight force (W) 
10(K) - pore-water force at the top of slice (U_t)
11(L) - pore-water force at normal to the base of slice (U_b)
12(M) - pore-water force at the left side of slice (U_l)
13(N) - pore-water force at the right side of slice (U_r)
14(O) - external load - line load (L)
15(P) - line load orientation from horizontal (omega)
16(Q) - perpendicular distance from the line load to the center of rotation (L-d)
17(R) - resultant maximum tensile force from support (T) 
18(S) - angle of support force from horizontal (i) 
19(T) - Soil Shear Strength force (Sm)
20(U) - Mohr-Coulomb shear strength - cohesion (c')
21(V) - Mohr-Coulomb shear strength - angle for friction (phi')
22(W) - tension crack coefficient 
			1 = no tension crack
			0 = all base tension cracked
			in-between = partial tension crack on the base length 
23(X) - Ru coefficient
'''

# making_list_with_floats.py

def csv2txtlistSlide3(fileName):
	import csv
	with open(fileName, 'r') as f:
		reader = csv.reader(f)
		csvListTxt = list(reader)

	csvList = []
	tempList = []
	for idR in range(len(csvListTxt)):	
		csvListTxt[idR] = [x for x in csvListTxt[idR] if x != '']

		for i in range(len(csvListTxt[idR])):
			if csvListTxt[idR][i] == '':
				continue

			elif i==0 and (idR==0 or idR%22 ==0):
				indList = list(csvListTxt[idR][i])
				blankInx = indList.index(' ')
				columnID = ''.join(csvListTxt[idR][i][(blankInx+1):])
				tempList.append(int(columnID))

			elif i!=0 and csvListTxt[idR][i].find(',') == -1:
				tempList.append(float(csvListTxt[idR][i]))

			elif i!=0 and csvListTxt[idR][i].find(',') != -1:
				commaIdx = csvListTxt[idR][i].index(',')
				tempList.append(float(csvListTxt[idR][i][0:commaIdx]))
				tempList.append(float(csvListTxt[idR][i][(commaIdx+2):]))

		if idR!=0 and len(tempList)==22: 
			csvList.append(tempList)
			tempList = []

	return csvList

def csv2list(fileName):
	import csv
	with open(fileName, 'r') as f:
		reader = csv.reader(f)
		csvListTxt = list(reader)
	
	csvListNum = []
	for idR in range(len(csvListTxt)):	
		csvListTxt[idR] = [x for x in csvListTxt[idR] if x != '']
		tempList = [float(i) for i in csvListTxt[idR]]
		csvListNum.append(tempList)

	return csvListNum

def making_float_list(startNum, endNum, spacing):
	result = []
	length = 1 + abs(startNum - endNum)/abs(spacing)
	for i in range(int(length)):
		x = startNum + spacing*i
		result.append(float(x))
	return result

def concatenate_lists(parentList, addingList):
	result = parentList[:]
	for i in range(len(addingList)):
		result.append(float(addingList[i]))
	return result

def listAtColNum(listName,colNum):
	result = []
	for i in range(len(listName)):
		result.append(float(listName[i][colNum]))
	return result

def listAtColNumTxt(listName,colNum):
	result = []
	for i in range(len(listName)):
		result.append(listName[i][colNum])
	return result

def arrayAtColNum(arrayName,colNum):
	import numpy as np
	result = []
	for i in range(len(arrayName)):
		result.append(float(arrayName[i][colNum]))
	result = np.array(result)
	return result

# csv_file = filename of csv exported from list
# csv_column = column titles
# data_list = list data
def exportList2CSV(csv_file,data_list,csv_columns=None):
	# export files
	import csv

	with open(csv_file, 'w',newline='') as csvfile:
		writer = csv.writer(csvfile, delimiter=',')
		if csv_columns != None:
			writer.writerow(csv_columns)
		for data in data_list:
			writer.writerow(data)   

# Analysis_2D_Ordinary_V4_06_09_2018.py
def ordinary_method(filename):
	import math

	# converting the input csv file into a list
	analysisInput = csv2list(filename)

	# total number of slip surfaces
	totalSlipSurface = int(analysisInput[0][2])	

	# create an empty list for FS of all slip surfaces
	FS = []

	# cut inputfile into separate lists
	for surface_num in range(totalSlipSurface):
		if surface_num == 0:
			startingRowN=1
		else:
			startingRowN = endingRowN+1
		endingRowN = startingRowN + int(analysisInput[startingRowN][3])

		analysisInfo= analysisInput[startingRowN]
		sliceInfo= analysisInput[startingRowN+1:endingRowN+1]

		# create variables used in ordinary method equation
		numerator=0 #numerator of the equation
		sum_Wx=0
		sum_Pf=0
		sum_kWe=0
		sum_Aa=analysisInfo[8]*analysisInfo[9]
		sum_Ld=0

		# add values from each slice to the total
		for slice in sliceInfo:
			# water forces
			u=slice[11]-slice[10]
			l=slice[2]*slice[22]
			P=(slice[9] + slice[10])*math.cos(math.radians(slice[3])) - analysisInfo[7]*slice[9]*math.sin(math.radians(slice[3])) 
			numerator+=(slice[20]*l+(P-u)*math.tan(math.radians(slice[21])))*slice[5]
			sum_Wx+=(slice[9])*slice[7]
			sum_Pf+=slice[6]*P
			sum_kWe+=analysisInfo[7]*slice[9]*slice[8]
			sum_Ld+=slice[14]*slice[16]
		
		# add the FS of each slip surface to the list
		FS.append(numerator/(sum_Wx-sum_Pf+sum_kWe+sum_Aa+sum_Ld))

	return FS

# Analysis_2D_Modified_Bishop_Method_V6.py
def modified_bishop(filename,tol=0.0001,iterationNMax=100):
	import math

	# converting the input csv file into a list
	analysisInput= csv2list(filename)

	# total number of slip surfaces
	totalSlipSurface= int(analysisInput[0][2])

	# using ordinary method as a first guess
	FSguess = ordinary_method(filename)

	# create an empty list for FS of all slip surfaces
	FS=[]

	# cut inputfile into separate lists
	for surface_num in range(totalSlipSurface):
		iterationN=0
		if surface_num == 0:
			startingRowN=1
		else:
			startingRowN= endingRowN+1
		endingRowN=startingRowN + int(analysisInput[startingRowN][3])

		analysisInfo=analysisInput[startingRowN]
		sliceInfo= analysisInput[startingRowN+1:endingRowN+1]

		# set initial difference bigger than the tolerance
		difference = tol+1

		while difference > tol:

			# create variables used in ordinary method equation
			numerator=0
			sum_Wx=0
			sum_Pf=0
			sum_kWe=0
			sum_Aa=analysisInfo[8]*analysisInfo[9]
			sum_Ld=0
			sum_T=0

			# add values from each slice to the total
			for slice in sliceInfo:
				# water forces
				l=slice[2]*slice[22] #slice[1]/math.cos(math.radians(slice[3]))
				u=(slice[11]-slice[10])
				m_alpha=math.cos(math.radians(slice[3]))+(math.sin(math.radians(slice[3]))*math.tan(math.radians(slice[21])))/FSguess[surface_num]
				P=(slice[9]-(slice[20]*l*math.sin(math.radians(slice[3])))/FSguess[surface_num] + (u*math.tan(math.radians(slice[21]))*math.sin(math.radians(slice[3])))/FSguess[surface_num])/m_alpha
				numerator+=(slice[20]*l*slice[5]+(P-u)*slice[5]*math.tan(math.radians(slice[21]))) 
				sum_Wx+=slice[9]*slice[7]
				sum_Pf+=slice[6]*P
				sum_kWe+=analysisInfo[7]*slice[9]*slice[8]
				sum_Ld+=slice[14]*slice[16]
				sum_T+=slice[17]

			F=numerator/(sum_Ld-sum_Pf+sum_Wx+sum_Aa+sum_kWe)
			# find the difference between the guess and the result
			difference=abs(FSguess[surface_num]-F)
			if difference <= tol:
				FS.append(F)
				iterationN+=1
			# stop the loop when number of iterations is over the limit
			elif iterationN >=iterationNMax:
				if 'NONE' in FS:
					FS.append('NONE')
					break
				else:
					FS.append('NONE')
					print ('too many iterations (iterationNN) - check code or increase maximum iteration number')
					break
			else:
				FSguess[surface_num]=F
				iterationN+=1
	return FS

#main function - 2D slope stability analysis using Janbu's Simplified Method
def janbu_simplified(filename,tol,iterationNMax,f0_used):
	import math
	import statistics

	# converting the input csv file into a list
	analysisInput= csv2list(filename)

	# total number of slip surfaces
	totalSlipSurface= int(analysisInput[0][2])

	# using ordinary method as a first guess
	FSguess=ordinary_method(filename)

	# create an empty list for FS of all slip surfaces
	FS=[]

	# cut inputfile into separate lists
	for surface_num in range(totalSlipSurface):
		iterationN=0
		if surface_num == 0:
			startingRowN=1
		else:
			startingRowN= endingRowN+1
		endingRowN=startingRowN + int(analysisInput[startingRowN][3])

		analysisInfo=analysisInput[startingRowN]
		sliceInfo= analysisInput[startingRowN+1:endingRowN+1]

		# set initial difference bigger than the tolerance
		difference = tol+1

		# put all cohesion factor and friction angles into lists and find the most common element
		c_list=[]
		phi_list=[]
		for slice in sliceInfo:
			c_list.append(slice[20])
			phi_list.append(slice[21])

		# using the most common element in the lists, find the according b1 values
		'''
		if statistics.mode(c_list) != 0 and statistics.mode(phi_list) !=0:
			b1=0.5
		elif statistics.mode(c_list) ==0 and statistics.mode(phi_list) !=0:
			b1=0.31
		elif statistics.mode(c_list) !=0 and statistics.mode(phi_list) ==0:
			b1=0.69
		'''
		if round(sum(c_list)) != 0 and round(sum(phi_list)) !=0:
			b1=0.5
		elif round(sum(c_list)) ==0 and round(sum(phi_list)) !=0:
			b1=0.31
		elif round(sum(c_list)) !=0 and round(sum(phi_list)) ==0:
			b1=0.69

		# if we don't want correction, set f0 to 1
		if f0_used== True:
			f0=1+b1*(analysisInfo[12]/analysisInfo[11]-1.4*(analysisInfo[12]/analysisInfo[11])**2)
		else:
			f0=1

		while difference>tol:
			numerator=0
			sum_Psina=0
			sum_kW=0
			sum_A=analysisInfo[8]
			sum_Lcosw=0 
			for slice in sliceInfo:
				l=slice[2]*slice[22] #slice[1]/math.cos(math.radians(slice[3]))
				u=(slice[11]-slice[10])/slice[2]
				m_alpha=math.cos(math.radians(slice[3]))+(math.sin(math.radians(slice[3]))*math.tan(math.radians(slice[21])))/FSguess[surface_num]
				P=(slice[9]-(slice[20]*l*math.sin(math.radians(slice[3])))/FSguess[surface_num]+(u*l*math.tan(math.radians(slice[21]))*math.sin(math.radians(slice[3])))/FSguess[surface_num])/m_alpha
				numerator += slice[20]*l*math.cos(math.radians(slice[3]))+(P-u*l)*math.tan(math.radians(slice[21]))*math.cos(math.radians(slice[3]))
				sum_Psina += P*math.sin(math.radians(slice[3]))
				sum_kW += analysisInfo[7]*slice[9]
				sum_Lcosw += slice[14]*math.cos(math.radians(slice[15]))

			F_0=numerator/(sum_Psina+sum_kW+sum_A+sum_Lcosw)
			F=F_0*f0

			# find the difference between the guess and the result
			difference=abs(FSguess[surface_num]-F)
			if difference <= tol:
				FS.append(F)
				iterationN+=1
			# stop the loop when number of iterations is over the limit
			elif iterationN >=iterationNMax:
				if 'NONE' in FS:
					FS.append('NONE')
					break
				else:
					FS.append('NONE')
					print ('too many iterations (iterationNN) - check code or increase maximum iteration number')
					break
			else:
				FSguess[surface_num]=F
				iterationN+=1
	
	# tell the user whether the result was modified using the correction factor
	'''
	if f0_used==True:
		message='correction factor was used'
	else:
		message='correction factor was not used'
	print(message)
	'''
	return FS

# Analysis_2D_Spencer_v2_06_05_2018.py
def changeIntersliceTheta_Spencer(thetaInter, FS_moment_f, FS_force_f, tolaranceFS):
	# create total number of change criteria based on decimal points   
	if tolaranceFS >= 1:
		decimalPoint = 1
	elif tolaranceFS < 0.0001:
		dpListed = list(str(tolaranceFS))
		idx = dpListed.index('-')
		dPstring = ''.join(dpListed[idx+1:])
		decimalPoint = int(dPstring)
	else:
		decimalPoint = len(list(str(tolaranceFS)))-2	

	if decimalPoint >= 5:
		decimalPoint = 5
		tolaranceFS = 0.00001
		
	dFSLimList = [1]
	for loop1 in range(decimalPoint):
		if loop1 == decimalPoint-1:
			dFSLimList.append(tolaranceFS)
		elif tolaranceFS >= 0.0001 and loop1 == decimalPoint-2:
			dFSLimList.append(tolaranceFS*5)
		else:
			dFSLimList.append(0.1*float('1E-'+str(loop1)))

	# change the interslice force angle
	completeValueChangeSet = [10, 5, 1, 0.1, 0.01]
	valueChangeList = completeValueChangeSet[-(decimalPoint):]

	# changing thetaInter higher or lower value
	if FS_moment_f>FS_force_f:
		UorD = 1
	else:
		UorD = -1

	absDiffFS = abs(FS_moment_f - FS_force_f)
	#print(FS_moment_f)
	#print(FS_force_f)
	#print(absDiffFS)
	for loop2 in range(decimalPoint):
		if absDiffFS <= tolaranceFS:
			valueChange = valueChangeList[-1]
			break
		elif absDiffFS <= dFSLimList[loop2] and absDiffFS > dFSLimList[loop2+1]:
			valueChange = valueChangeList[loop2]
			break
		elif loop2 == decimalPoint-1 and absDiffFS > dFSLimList[0]:
			valueChange = valueChangeList[0]
	
	thetaInter += valueChange*UorD

	return thetaInter

'''main function - Spencer Method 2D LEM slope stability analysis'''
def analysis2DSpencer(inputFileName, tolaranceFS, iterationNMax, changeThetaInter=None):
	# import function from Math Library
	#import numpy as np
	import math
 
	# take the inputfile and convert it into list
	analysisInput = csv2list(inputFileName)

	# initial trial of FS
	FS_initials = ordinary_method(inputFileName) 

	totalSlipSurface = int(analysisInput[0][2])	# total number of slip surfaces

	# cut into separate files
	results2DLEMSpencer = []
	for loopSlip in range(totalSlipSurface):
		# starting and ending row numbers
		if loopSlip == 0:
			startingRowN = 1
		else:
			startingRowN = endingRowN+1
		endingRowN = startingRowN + int(analysisInput[startingRowN][3])

		analysisInfo = analysisInput[startingRowN]
		sliceInfo = analysisInput[startingRowN+1:endingRowN+1]
		
		# trial interslice angle (theta)
		thetaInter = 0
		iterationNN = 1
		iteration1 = True

		while iteration1:

			FS_force_i = FS_initials[loopSlip]		# inital trial value of FS for force equilibrium
			FS_moment_i = FS_initials[loopSlip]			# inital trial value of FS for moment equilibrium
			iterationN = 1
			dE_list = []
			iteration2 = True
			
			while iteration2:
				
				# FS for force calculated
				FS_force = 0
				FS_f_nom = 0
				sum_Psina=0
				sum_kW=0
				sum_A=analysisInfo[8]
				sum_Lcosw=0 

				# FS for moment calculated
				FS_moment = 0			
				FS_m_nom=0
				sum_Wx=0
				sum_Pf=0
				sum_kWe=0
				sum_Aa=analysisInfo[8]*analysisInfo[9]
				sum_Ld=0
				sum_T=0

				# iterate trough slice
				for loopSlice in range(len(sliceInfo)):	

					# net pore-water pressure
					#u_net_base = (slice[11] - slice[10])/slice[2]
					u_net_side = 0 #abs(sliceInfo[loopSlice][12] - sliceInfo[loopSlice][13])

					# interslice assumption for first analysis
					if iterationN == 1:
						dX_f = 0 #math.tan(math.radians(thetaInter))*u_net_side
						dX_m = 0 #math.tan(math.radians(thetaInter))*u_net_side	   # change in vertical interslice force (dX = X_L-X_R)
					# using FS from previous calculation dX is calculated
					else:
						dX_f = math.tan(math.radians(thetaInter))*dE_list[loopSlice][0]
						dX_m = math.tan(math.radians(thetaInter))*dE_list[loopSlice][1]
						#loopSlice += 1

					# actual resisting base length = base length * tension crack coefficient
					#b_len_r = sliceInfo[loopSlice][2]*sliceInfo[loopSlice][22]
					l=sliceInfo[loopSlice][2]*sliceInfo[loopSlice][22] #sliceInfo[loopSlice][1]/math.cos(math.radians(sliceInfo[loopSlice][3]))
					u=(sliceInfo[loopSlice][11]-sliceInfo[loopSlice][10])

					# calcualte FS for moment 
					m_alpha_m=math.cos(math.radians(sliceInfo[loopSlice][3]))+(math.sin(math.radians(sliceInfo[loopSlice][3]))*math.tan(math.radians(sliceInfo[loopSlice][21])))/FS_moment_i
					P_moment=(sliceInfo[loopSlice][9] - dX_m -(sliceInfo[loopSlice][20]*l*math.sin(math.radians(sliceInfo[loopSlice][3])))/FS_moment_i + (u*sliceInfo[loopSlice][22]*math.tan(math.radians(sliceInfo[loopSlice][21]))*math.sin(math.radians(sliceInfo[loopSlice][3])))/FS_moment_i)/m_alpha_m
					FS_m_nom+=(sliceInfo[loopSlice][20]*l*sliceInfo[loopSlice][5]+(P_moment-u*sliceInfo[loopSlice][22])*sliceInfo[loopSlice][5]*math.tan(math.radians(sliceInfo[loopSlice][21]))) 
					sum_Wx+=sliceInfo[loopSlice][9]*sliceInfo[loopSlice][7]
					sum_Pf+=sliceInfo[loopSlice][6]*P_moment
					sum_kWe+=analysisInfo[7]*sliceInfo[loopSlice][9]*sliceInfo[loopSlice][8]
					sum_Ld+=sliceInfo[loopSlice][14]*sliceInfo[loopSlice][16]
					sum_T+=sliceInfo[loopSlice][17]

					# calcualte FS for force 
					m_alpha_f=math.cos(math.radians(sliceInfo[loopSlice][3]))+(math.sin(math.radians(sliceInfo[loopSlice][3]))*math.tan(math.radians(sliceInfo[loopSlice][21])))/FS_force_i
					P_force=(sliceInfo[loopSlice][9] - dX_f -(sliceInfo[loopSlice][20]*l*math.sin(math.radians(sliceInfo[loopSlice][3])))/FS_force_i + (u*sliceInfo[loopSlice][22]*math.tan(math.radians(sliceInfo[loopSlice][21]))*math.sin(math.radians(sliceInfo[loopSlice][3])))/FS_force_i)/m_alpha_f
					FS_f_nom += sliceInfo[loopSlice][20]*l*math.cos(math.radians(sliceInfo[loopSlice][3]))+(P_force-u*sliceInfo[loopSlice][22])*math.tan(math.radians(sliceInfo[loopSlice][21]))*math.cos(math.radians(sliceInfo[loopSlice][3]))
					sum_Psina += P_force*math.sin(math.radians(sliceInfo[loopSlice][3]))
					sum_kW += analysisInfo[7]*sliceInfo[loopSlice][9]
					sum_Lcosw += sliceInfo[loopSlice][14]*math.cos(math.radians(sliceInfo[loopSlice][15]))
					
					# calculate dE for next iteration
					# dE = change in horizontal interslice force (dE = E_L-E_R)
					dE_f = u_net_side + P_force*math.sin(math.radians(sliceInfo[loopSlice][3])) - (math.cos(math.radians(sliceInfo[loopSlice][3]))/FS_force_i)*(sliceInfo[loopSlice][20]*l +(P_force-u*sliceInfo[loopSlice][22])*math.tan(math.radians(sliceInfo[loopSlice][21]))) #+ analysisInfo[7]*sliceInfo[loopSlice][9] 
					dE_m = u_net_side + P_moment*math.sin(math.radians(sliceInfo[loopSlice][3])) - (math.cos(math.radians(sliceInfo[loopSlice][3]))/FS_moment_i)*(sliceInfo[loopSlice][20]*l +(P_moment-u*sliceInfo[loopSlice][22])*math.tan(math.radians(sliceInfo[loopSlice][21]))) #+ analysisInfo[7]*sliceInfo[loopSlice][9] 

					if iterationN == 1:
						dE_list.append([dE_f, dE_m])
					else:
						dE_list[loopSlice] = [dE_f, dE_m]

				# calculated FS
				#FS_force = FS_f_nom/FS_f_de
				FS_force=FS_f_nom/(sum_Psina+sum_kW+sum_A+sum_Lcosw)
				#FS_moment = FS_m_nom/FS_m_de
				FS_moment=FS_m_nom/(sum_Ld-sum_Pf+sum_Wx+sum_Aa+sum_kWe)
				
				if iterationN >= iterationNMax:
					print('too many iterations - check code or increase maximum iteration number')
					iteration2 = False
				elif abs(FS_force_i - FS_force) > tolaranceFS or abs(FS_moment_i - FS_moment) > tolaranceFS:
					FS_force_i = FS_force
					FS_moment_i = FS_moment
					FS_force = 0
					FS_moment = 0
					iterationN += 1
				else:
					FS_force_i = FS_force
					FS_moment_i = FS_moment
					iteration2 = False

			FS_force_f = FS_force_i
			FS_moment_f = FS_moment_i
			#print(thetaInter, FS_force_f, FS_moment_f)
			
			if iterationN >= iterationNMax or iterationNN >= iterationNMax:
				print('too many iterations - check code or increase maximum iteration number')
				iteration1 = False
				FS_final = 'None'
			elif abs(FS_moment_f - FS_force_f) > tolaranceFS:
				iterationNN += 1
				if changeThetaInter == None:
					thetaInter = changeIntersliceTheta_Spencer(thetaInter, FS_moment_f, FS_force_f, tolaranceFS)
				elif changeThetaInter != None:
					thetaInter += changeThetaInter
			else:
				FS_final = FS_force_f
				iteration1 = False
		
		results2DLEMSpencer.append([analysisInfo[0:3], [FS_final, thetaInter]])

	return results2DLEMSpencer

# Analysis_2D_Morgenstern_Price_v1_06_03_2018.py

'''change the interslice angle Lambda based on the difference of FS'''
def changeIntersliceLambda_MP(scaleLambda, FS_moment_f, FS_force_f, tolaranceFS):
	# create total number of change criteria based on decimal points   
	if tolaranceFS >= 1:
		decimalPoint = 1
	elif tolaranceFS < 0.0001:
		dpListed = list(str(tolaranceFS))
		idx = dpListed.index('-')
		dPstring = ''.join(dpListed[idx+1:])
		decimalPoint = int(dPstring)
	else:
		decimalPoint = len(list(str(tolaranceFS)))-2	

	if decimalPoint >= 5:
		decimalPoint = 5
		tolaranceFS = 0.00001
		
	dFSLimList = [0.5]
	for loop1 in range(decimalPoint):
		if loop1 == decimalPoint-1:
			dFSLimList.append(tolaranceFS)
		#elif tolaranceFS >= 0.0001 and loop1 == decimalPoint-2:
		#	dFSLimList.append(tolaranceFS*5)
		else:
			dFSLimList.append(0.1*float('1E-'+str(loop1)))

	# change the interslice force angle
	completeValueChangeSet = [0.5, 0.1, 0.05, 0.01, 0.001]
	valueChangeList = completeValueChangeSet[-(decimalPoint):]
	
	# changing Lambda higher or lower value
	if FS_moment_f > FS_force_f:
		UorD = 1
	else:
		UorD = -1

	absDiffFS = abs(FS_moment_f - FS_force_f)
	for loop2 in range(decimalPoint):
		if absDiffFS <= tolaranceFS:
			valueChange = valueChangeList[-1]
			break
		elif absDiffFS <= dFSLimList[loop2] and absDiffFS > dFSLimList[loop2+1]:
			valueChange = valueChangeList[loop2]
			break
		else:
			valueChange = valueChangeList[-1]
			break

	scaleLambda += valueChange*UorD

	return scaleLambda

'''the interslice function'''
def intersliceFx_MorgensternPrice(sliceInfo, FxType, inputFx):
	# import modules
	import math

	# designate x in F(x) interslice function to each slice
	interSliceFx = []
	
	sumX = sum(listAtColNum(sliceInfo,1))
	x = 0

	# when Fxtype = 4: parameters from the input to create linear line equations
	if FxType == 4:
		Fxline = []
		for loopInFx in range(len(inputFx)-1):
			start_x = inputFx[loopInFx][0] 
			end_x = inputFx[loopInFx+1][0] 
			gradient = (inputFx[loopInFx+1][1]-inputFx[loopInFx][1])/(inputFx[loopInFx+1][0]-inputFx[loopInFx][0])
			intercept = inputFx[loopInFx][1] - gradient*inputFx[loopInFx][0]
			Fxline.append([start_x, end_x, gradient, intercept])

	for loopX in range(len(sliceInfo)):
		# x position of each slice
		x += 0.5*sliceInfo[loopX][1]/sumX	# normalized to vary between 0 and 1

		# interslice function
		if FxType == 1:	# constant
			Fx = 1.0

		elif FxType == 2:	# half-sine
			Fx = round(math.sin((math.pi)*x),3)

		elif FxType == 3:	# clipped sine
			# parameters from the input
			startFx = inputFx[0]
			endFx = inputFx[1]

			# find phase of sine curve
			start_x_angle = math.asin(startFx)
			end_x_angle = math.asin(endFx)
			if start_x_angle > end_x_angle:
				phase_angle = (math.pi - end_x_angle) - start_x_angle
			else:
				phase_angle = end_x_angle - start_x_angle

			Fx = round(math.sin(phase_angle*x + start_x_angle),3)

		elif FxType == 4:	# general - user-defined
			for loopRow in range(len(Fxline)):
				if x >= Fxline[loopRow][0] and x <= Fxline[loopRow][1]:
					Fx = x*Fxline[loopRow][2] + Fxline[loopRow][3]
					break

		interSliceFx.append([sliceInfo[loopX][0], x, Fx])

	return interSliceFx

'''main function - Spencer Method 2D LEM slope stability analysis'''
def analysis2DMorgensternPrice(inputFileName, tolaranceFS, FxType, inputFx, iterationNMax, changeScaleLambda=None):
	# import function from Math Library
	#import numpy as np
	import math
 
	# take the inputfile and convert it into list
	analysisInput = csv2list(inputFileName)

	# initial trial of FS
	FS_initials = ordinary_method(inputFileName) 

	totalSlipSurface = int(analysisInput[0][2])	# total number of slip surfaces

	# cut into separate files
	results2DMP = []
	for loopSlip in range(totalSlipSurface):
		# starting and ending row numbers
		if loopSlip == 0:
			startingRowN = 1
		else:
			startingRowN = endingRowN+1
		endingRowN = startingRowN + int(analysisInput[startingRowN][3])

		analysisInfo = analysisInput[startingRowN]
		sliceInfo = analysisInput[startingRowN+1:endingRowN+1]
		
		# trial interslice angle (theta)
		scaleLambda = 0 #-0.5
		intersliceFxList = intersliceFx_MorgensternPrice(sliceInfo, FxType, inputFx)
		iterationNN = 1
		iteration1 = True
				
		while iteration1:

			FS_force_i = FS_initials[loopSlip]		# inital trial value of FS for force equilibrium
			FS_moment_i = FS_initials[loopSlip]			# inital trial value of FS for moment equilibrium
			iterationN = 1
			dE_list = []
			iteration2 = True
			
			while iteration2:
				
				# FS for force calculated
				FS_force = 0
				FS_f_nom = 0
				sum_Psina=0
				sum_kW=0
				sum_A=analysisInfo[8]
				sum_Lcosw=0 

				# FS for moment calculated
				FS_moment = 0			
				FS_m_nom=0
				sum_Wx=0
				sum_Pf=0
				sum_kWe=0
				sum_Aa=analysisInfo[8]*analysisInfo[9]
				sum_Ld=0
				sum_T=0

				# iterate trough slice
				for loopSlice in range(len(sliceInfo)):	

					# net pore-water pressure
					#u_net_base = (slice[11] - slice[10])/slice[2]
					u_net_side = 0 #abs(sliceInfo[loopSlice][12] - sliceInfo[loopSlice][13])

					# interslice assumption for first analysis
					if iterationN == 1:
						dX_f = 0 #math.tan(math.radians(thetaInter))*u_net_side
						dX_m = 0 #math.tan(math.radians(thetaInter))*u_net_side	   # change in vertical interslice force (dX = X_L-X_R)
					# using FS from previous calculation dX is calculated
					else:
						dX_f = scaleLambda*intersliceFxList[loopSlice][2]*dE_list[loopSlice][0]
						dX_m = scaleLambda*intersliceFxList[loopSlice][2]*dE_list[loopSlice][1]
						#loopSlice += 1

					# actual resisting base length = base length * tension crack coefficient
					#b_len_r = sliceInfo[loopSlice][2]*sliceInfo[loopSlice][22]
					l=sliceInfo[loopSlice][2]*sliceInfo[loopSlice][22] #sliceInfo[loopSlice][1]/math.cos(math.radians(sliceInfo[loopSlice][3]))
					u=(sliceInfo[loopSlice][11]-sliceInfo[loopSlice][10])

					# calcualte FS for moment 
					m_alpha_m=math.cos(math.radians(sliceInfo[loopSlice][3]))+(math.sin(math.radians(sliceInfo[loopSlice][3]))*math.tan(math.radians(sliceInfo[loopSlice][21])))/FS_moment_i
					P_moment=(sliceInfo[loopSlice][9] - dX_m -(sliceInfo[loopSlice][20]*l*math.sin(math.radians(sliceInfo[loopSlice][3])))/FS_moment_i + (u*sliceInfo[loopSlice][22]*math.tan(math.radians(sliceInfo[loopSlice][21]))*math.sin(math.radians(sliceInfo[loopSlice][3])))/FS_moment_i)/m_alpha_m
					FS_m_nom+=(sliceInfo[loopSlice][20]*l*sliceInfo[loopSlice][5]+(P_moment-u*sliceInfo[loopSlice][22])*sliceInfo[loopSlice][5]*math.tan(math.radians(sliceInfo[loopSlice][21]))) 
					sum_Wx+=sliceInfo[loopSlice][9]*sliceInfo[loopSlice][7]
					sum_Pf+=sliceInfo[loopSlice][6]*P_moment
					sum_kWe+=analysisInfo[7]*sliceInfo[loopSlice][9]*sliceInfo[loopSlice][8]
					sum_Ld+=sliceInfo[loopSlice][14]*sliceInfo[loopSlice][16]
					sum_T+=sliceInfo[loopSlice][17]

					# calcualte FS for force 
					m_alpha_f=math.cos(math.radians(sliceInfo[loopSlice][3]))+(math.sin(math.radians(sliceInfo[loopSlice][3]))*math.tan(math.radians(sliceInfo[loopSlice][21])))/FS_force_i
					P_force=(sliceInfo[loopSlice][9] - dX_f -(sliceInfo[loopSlice][20]*l*math.sin(math.radians(sliceInfo[loopSlice][3])))/FS_force_i + (u*sliceInfo[loopSlice][22]*math.tan(math.radians(sliceInfo[loopSlice][21]))*math.sin(math.radians(sliceInfo[loopSlice][3])))/FS_force_i)/m_alpha_f
					FS_f_nom += sliceInfo[loopSlice][20]*l*math.cos(math.radians(sliceInfo[loopSlice][3]))+(P_force-u*sliceInfo[loopSlice][22])*math.tan(math.radians(sliceInfo[loopSlice][21]))*math.cos(math.radians(sliceInfo[loopSlice][3]))
					sum_Psina += P_force*math.sin(math.radians(sliceInfo[loopSlice][3]))
					sum_kW += analysisInfo[7]*sliceInfo[loopSlice][9]
					sum_Lcosw += sliceInfo[loopSlice][14]*math.cos(math.radians(sliceInfo[loopSlice][15]))
					
					# calculate dE for next iteration
					# dE = change in horizontal interslice force (dE = E_L-E_R)
					dE_f = u_net_side + P_force*math.sin(math.radians(sliceInfo[loopSlice][3])) - (math.cos(math.radians(sliceInfo[loopSlice][3]))/FS_force_i)*(sliceInfo[loopSlice][20]*l +(P_force-u*sliceInfo[loopSlice][22])*math.tan(math.radians(sliceInfo[loopSlice][21]))) #+ analysisInfo[7]*sliceInfo[loopSlice][9] 
					dE_m = u_net_side + P_moment*math.sin(math.radians(sliceInfo[loopSlice][3])) - (math.cos(math.radians(sliceInfo[loopSlice][3]))/FS_moment_i)*(sliceInfo[loopSlice][20]*l +(P_moment-u*sliceInfo[loopSlice][22])*math.tan(math.radians(sliceInfo[loopSlice][21]))) #+ analysisInfo[7]*sliceInfo[loopSlice][9] 

					if iterationN == 1:
						dE_list.append([dE_f, dE_m])
					else:
						dE_list[loopSlice] = [dE_f, dE_m]

				# calculated FS
				#FS_force = FS_f_nom/FS_f_de
				FS_force=FS_f_nom/(sum_Psina+sum_kW+sum_A+sum_Lcosw)
				#FS_moment = FS_m_nom/FS_m_de
				FS_moment=FS_m_nom/(sum_Ld-sum_Pf+sum_Wx+sum_Aa+sum_kWe)
				
				if iterationN >= iterationNMax:
					#print('too many iterations - check code or increase maximum iteration number')
					iteration2 = False
				elif abs(FS_force_i-FS_force) > tolaranceFS or abs(FS_moment_i-FS_moment) > tolaranceFS:
					FS_force_i = FS_force
					FS_moment_i = FS_moment
					FS_force = 0
					FS_moment = 0
					iterationN += 1
				else:
					FS_force_i = FS_force
					FS_moment_i = FS_moment
					iteration2 = False

			FS_force_f = FS_force_i
			FS_moment_f = FS_moment_i
			
			#print(scaleLambda, FS_force_f, FS_moment_f)

			if iterationNN >= iterationNMax:
				print('too many iterations (iterationNN) - check code or increase maximum iteration number')
				iteration1 = False
				FS_final = 0.5*(FS_force_f+FS_moment_f)

			elif abs(FS_moment_f-FS_force_f) > tolaranceFS:
				'''
				if iterationNN == 1:
					absDiffFS_p, scaleLambda = changeIntersliceLambda_MP(0.1, scaleLambda, FS_moment_f, FS_force_f, tolaranceFS)
				else:
					absDiffFS_p, scaleLambda = changeIntersliceLambda_MP(absDiffFS_p, scaleLambda, FS_moment_f, FS_force_f, tolaranceFS)
				'''
				iterationNN += 1

				if changeScaleLambda == None:
					scaleLambda = changeIntersliceLambda_MP(scaleLambda, FS_moment_f, FS_force_f, tolaranceFS)
				elif changeScaleLambda != None:
					scaleLambda += changeScaleLambda
			else:
				FS_final = FS_force_f
				iteration1 = False
		
		results2DMP.append([analysisInfo[0:3], [FS_final, scaleLambda, FxType]])
		#print(iterationNN)

	return results2DMP

# for input method: ordinary=1, modified bishop=2, simplified Janbu=3, corected Janbu=4, spencer=5, morgenstern price=6
'''
FxType,inputFx
interslice function options:
	1 = constant
	2 = half-sine
	3 = clipped sine 	-> inputFx = [F(x=0), F(x=1)]
	4 = user-specified	-> inputFx = (format = (x,Fx)) [[0,F(x=0)], [x1,F(x=x1)], [x2,F(x=x2)], ... [1,F(x=1)]
'''
def selectMethod(inputFileName,method,tolaranceFS,iterationNMax,FxType,inputFx):
	if method == 1:
		FS2D = ordinary_method(inputFileName)
		return FS2D#[0]
	elif method == 2:
		FS2D = modified_bishop(inputFileName,tolaranceFS,iterationNMax)
		return FS2D#[0]
	elif method == 3:
		FS2D = janbu_simplified(inputFileName,tolaranceFS,iterationNMax,False)
		return FS2D#[0]
	elif method == 4:
		FS2D =  janbu_simplified(inputFileName,tolaranceFS,iterationNMax,True)
		return FS2D#[0]
	elif method == 5:
		FS2D = analysis2DSpencer(inputFileName, tolaranceFS, iterationNMax, changeThetaInter=0.5)
		return FS2D#[0][1][0]
	elif method == 6:
		FS2D = analysis2DMorgensternPrice(inputFileName, tolaranceFS, FxType, inputFx, iterationNMax, changeScaleLambda=0.005)
		return FS2D#[0][1][0]
	else:
		return 'Invalid Input Method'

#testing

'''
print('from initial input')
print (selectMethod('test inputs for analysis.csv',1,0.001,1000,0,0))
print (selectMethod('test inputs for analysis.csv',2,0.001,1000,0,0))
print (selectMethod('test inputs for analysis.csv',3,0.001,1000,0,0))
#print (selectMethod('test inputs for analysis.csv',4,0.001,1000,0,0))
print (selectMethod('test inputs for analysis.csv',5,0.001,1000,0,0))
print (selectMethod('test inputs for analysis.csv',6,0.001,1000,2,0))
'''

import time
time_start = time.clock()

inputFileName = '2DanalysisInputFile.csv'
#method = 2
tolaranceFS = 0.001
iterationNMax = 1000
FxType = 1
inputFx = 0
'''
print('method=1')
print (selectMethod(inputFileName,1,tolaranceFS,iterationNMax,FxType,inputFx))
print()
print('method=2')
print (selectMethod(inputFileName,2,tolaranceFS,iterationNMax,FxType,inputFx))
print()
'''
print('method=3')
print (selectMethod(inputFileName,3,tolaranceFS,iterationNMax,FxType,inputFx))
print()
print('method=4')
print (selectMethod(inputFileName,4,tolaranceFS,iterationNMax,FxType,inputFx))
print()
'''
print('method=5')
print (selectMethod(inputFileName,5,tolaranceFS,iterationNMax,FxType,inputFx))
print()
print('method=6')
print (selectMethod(inputFileName,6,tolaranceFS,iterationNMax,FxType,inputFx))
'''

time_elapsed = (time.clock() - time_start)
print(time_elapsed)  # tells us the computation time in seconds