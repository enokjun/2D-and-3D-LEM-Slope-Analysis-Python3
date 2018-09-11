'''
3D LEM Slope Analysis:
1. Hungr Bishop
2. Hungr Janbu 
3. Hungr Janbu corrected
4. Cheng and Yip Bishop
5. Cheng and Yip Janbu
6. Cheng and Yip Spencer
'''

'''
# file input
0(A) columnId
1(B),2(C) Coordinates X and Y
3(D) Base Normal Stress (psf)
4(E) Base Effective Normal Stress (psf)
5(F) Pore Pressure (psf)
6(G) Excess Pore Pressure (psf)
7(H) Initial Pore Pressure (psf)
8(I) Shear Strength (psf)
9(J) Shear Stress (psf)
10(K) Base Shear Force (lbs)
11(L) Base Normal Force (lbs)
12(M) Base Cohesion (psf)
13(N) Base Friction Angle (deg)
14(O) Column Weight (lbs)
15(P) Column Volume (ft3)
16(Q) Column Weight/Area (lbs/ft2)
17(R) Dip of Column Base (deg)
18(S) Dip Direction of Column Base (deg)
19(T) Matric Suction (psf)
20(U) Column Center Z Top (ft)
21(V) Column Center Z Bottom (ft)

# may be added later
22(W),23(X),24(Y),25(Z) = area of side 1,2,3,4
26(AA) = shear model type
27(AB) = material name ID
28(AC) = tensile strength - support
29(AD) = tension crack coefficient
30(AE) = Ru coefficient
31(AF) = Dip of Column Top (deg)
32(AG) = Dip Direction of Column Top (deg)
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

# pts = [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3)]
def angleBtw3Pts(pts):
	import numpy as np

	ptA, ptB, ptC = pts[0], pts[1], pts[2]
	x1, y1, z1 = float(ptA[0]), float(ptA[1]), float(ptA[2])
	x2, y2, z2 = float(ptB[0]), float(ptB[1]), float(ptB[2])
	x3, y3, z3 = float(ptC[0]), float(ptC[1]), float(ptC[2])

	# pts is the central point
	vector1 = np.array([[x2-x1], [y2-y1], [z2-z1]])
	vector2 = np.array([[x3-x1], [y3-y1], [z3-z1]])

	mag1 = np.linalg.norm(vector1)
	mag2 = np.linalg.norm(vector2)

	dotProduct = (vector1[0]*vector2[0]) + (vector1[1]*vector2[1]) + (vector1[2]*vector2[2])

	angle = np.degrees(np.arccos(dotProduct/(mag1*mag2)))

	return angle

# main function - slope stability base shear parameters with class input 
## Input file column legend
'''
1 = Mohr-Coulomb - [phi', c']
2 = undrained - depth - [shear_max, shear_min, Su_top, diff_Su]
3 = undrained - datum - [shear_max, shear_min, Su_datum, diff_Su, z_datum]
4 = power curve - [P_atm, a, b]
5 = shear-normal user defined function - [fileName]
unsaturated - [phi_b, aev, us_max]
'''
def shearModel2cphi(materialClass, materialName, Z, Ztop, eff_normal_stress, matricSuction):
	#import math
	#import making_list_with_floats as makelist
	import numpy as np

	# materialClass takes input 
	# class[name] = [modelType, inputPara]

	material = materialClass[materialName]
	modelType = material[0]
	inputPara = material[1]

	calcPhiC=[]	
	#userShearNormal_mc=[]

	# Mohr-Coulomb Model
	if modelType in [1,10]:
		calcPhiC.append([inputPara[0], inputPara[1]])
		
	# undrained - depth
	elif modelType in [2,20]:
		calcShear = inputPara[2] + inputPara[3]*(Ztop - Z)

		if calcShear < inputPara[1]: # lower than min shear
			calcShear = inputPara[1]
		if calcShear > inputPara[0]: # higher than max shear
			calcShear = inputPara[0]
		
		calcPhiC.append([0, calcShear])

	# undrained - datum
	elif modelType in [3,30]:
		calcShear = inputPara[2] + inputPara[3]*abs(inputPara[4] - Z)

		if calcShear < inputPara[1]: # lower than min shear
			calcShear = inputPara[1]
		if calcShear > inputPara[0]: # higher than max shear
			calcShear = inputPara[0]
		
		calcPhiC.append([0, calcShear])

	# power curve
	elif modelType in [4,40]:
		calcShear = inputPara[0]*inputPara[1]*(eff_normal_stress/inputPara[0])**inputPara[2]

		calcPhiC.append([0, calcShear])

	# user defined shear-normal force
	elif modelType in [5,50]:
		
		userShearNormal =  csv2list(inputPara[0])

		for loopSN in range(len(userShearNormal)-1):
			if inputPara[1] <= userShearNormal[loopSN+1][0] and inputPara[1] >= userShearNormal[loopSN][0]:
				
				gradient = (userShearNormal[loopSN+1][1] - userShearNormal[loopSN][1])/(userShearNormal[loopSN+1][0] - userShearNormal[loopSN][0])
				intercept = userShearNormal[loopSN][1] - gradient*userShearNormal[loopSN][0]
				#userShearNormal_mc.append([gradient, intercept])
			
				calcShear = gradient*eff_normal_stress + intercept
				break

			elif inputPara[1] >= userShearNormal[len(userShearNormal)][0]:
				#print('out of range of nonlinear curve')
				calcShear = userShearNormal[len(userShearNormal)][1]
				break

			else:
				continue

		calcPhiC.append([0, calcShear])

	# adding unsaturated strength
	if round(modelType/10) in [1,2,3,4,5]:
		if matricSuction >= inputPara[1]:
			unsatShearStrength = min([matricSuction, inputPara[2]])*np.tan(np.radians(inputPara[0]))
			calcPhiC[1] += unsatShearStrength

	return calcPhiC


# return list of soil columns that is at the exterior
def colIDExternal(analysisInput):
	import numpy as np

	idxList = np.arange(len(analysisInput)-1).tolist()
	idList = listAtColNum(analysisInput,0)	# soil column ID number
	idList = [int(idx) for idx in idList]
	xList = sorted(np.unique(listAtColNum(analysisInput,1)))	# unique list of x coordinates
	yList = sorted(np.unique(listAtColNum(analysisInput,2)))	# unique list of y coordinates

	# sort soil columns into rows with same X coordinates
	XsortedIDList = []
	XsortedIDXList = []
	for xCoord in xList:
		tempX = []
		tempIDX = []
		for loopColID in idxList:
			if analysisInput[loopColID][1] == xCoord:
				tempX.append(idList[loopColID])
				tempIDX.append(loopColID)
				idxList.remove(loopColID)

		XsortedIDList.append(tempX)
		XsortedIDXList.append(tempIDX)

	# take the exteriors from each row of soil columns
	extXsortedIDList = []
	extXsortedIDXList = []
	exposedSidesDict = {}

	for loopExt in range(len(XsortedIDList)):

		# first soil column of each column
		extXsortedIDList.append(XsortedIDList[loopExt][0])
		extXsortedIDXList.append(XsortedIDXList[loopExt][0])
		
		sideCheck1 = [[False]*4][0]
		sideCheck1[0] = True # side 1 is automatically open

		# side 2 
		if yList[-1] == analysisInput[XsortedIDXList[loopExt][0]][2] or (XsortedIDList[loopExt][0]+1) not in idList:
			sideCheck1[1] = True 	# right most soil column - side 2 open

		# side 4
		if yList[0] == analysisInput[XsortedIDXList[loopExt][0]][2] or (XsortedIDList[loopExt][0]-1) not in idList:
			sideCheck1[3] = True 	# left most soil column - side 4 open

		exposedSidesDict[XsortedIDXList[loopExt][0]] = sideCheck1

		# last soil column of each column
		extXsortedIDList.append(XsortedIDList[loopExt][-1])
		extXsortedIDXList.append(XsortedIDXList[loopExt][-1])

		sideCheck2 = [[False]*4][0]
		sideCheck2[2] = True # side 3 is automatically open

		# side 2 
		if yList[-1] == analysisInput[XsortedIDXList[loopExt][-1]][2] or (XsortedIDList[loopExt][-1]+1) not in idList:
			sideCheck2[1] = True 	# right most soil column - side 2 open

		# side 4
		if yList[0] == analysisInput[XsortedIDXList[loopExt][-1]][2] or (XsortedIDList[loopExt][-1]-1) not in idList:
			sideCheck2[3] = True 	# left most soil column - side 4 open

		exposedSidesDict[XsortedIDXList[loopExt][-1]] = sideCheck2

	return XsortedIDList, XsortedIDXList, extXsortedIDList, extXsortedIDXList, exposedSidesDict


# Analysis_3D_HungrBishop1989_v4_06_28_2018.py
def analysis3DHungrBishop1989(fileName, seismicK, centerPt0, materialClass, materialNameList, iterationNMax=200, tolFS=0.0005, occuranceFactor=0.5, tolDirection_user=None, spacingDirection=0.5, avDipDirectionB_user=None):
	# import libraries
	import numpy as np
	from scipy.stats import mode
	
	# take the inputfile and convert it into list
	analysisInput = csv2list(fileName)
	#print(analysisInput[0])

	# seismic coefficient
	seismicKx = seismicK[0]
	seismicKy = seismicK[1]
	seismicKxy = np.sqrt(seismicKx**2 + seismicKy**2)

	''' direction of column dip direction '''
	if avDipDirectionB_user == None:
		dipDirectionList =  listAtColNum(analysisInput, 18)	
		avDipDirectionB, countN = mode(np.array(dipDirectionList),axis=None)

		if countN >= occuranceFactor*len(analysisInput):
			avDipDirectionB = round(float(avDipDirectionB),1)
			tolDirection = 0
		else:
			avDipDirectionB = round(float(np.mean(dipDirectionList)),1)
			if tolDirection_user == None:
				tolDirection = 10
			else:
				tolDirection = tolDirection_user
	else:
		avDipDirectionB = avDipDirectionB_user
		tolDirection = 0
	
	#print(avDipDirectionB)

	directionOfSlidingBList =  making_float_list(avDipDirectionB-tolDirection, avDipDirectionB+tolDirection, spacingDirection)
	#directionOfSlidingPList =  making_float_list(avDipDirectionP-tolDirection, avDipDirectionP+tolDirection, spacingDirection)

	#changeFS = 0.01

	''' FS computation '''
	FS_results = []
	
	for dirLoop in range(len(directionOfSlidingBList)):

		iterationN = 1
		FSm_i = 3			#inital guess of FS

		# iterate through to find global 3D FS
		iterationFS = True
		while iterationFS:

			# calculating the FS 
			FSmNu = 0
			#FSmDe = 0
			FSmDeWx = 0
			FSmDeNf = 0
			FSmDekWe = 0 
			FSmDeEd = 0

			for loopCol in range(len(analysisInput)):
				#print(analysisInput[loopCol])
				#print('columnID=%i'%loopCol)
				
				# base inclination
				# inclination of base relative to the positive x (right) and positive y (up) direction
				dipRad = np.radians(analysisInput[loopCol][17])

				if analysisInput[loopCol][17] == 0:
					dipDirectionRad = np.radians(directionOfSlidingBList[dirLoop])
				else:
					dipDirectionRad = np.radians(analysisInput[loopCol][18])

				baseAngleXraw = np.arctan(np.sin(dipDirectionRad)*np.tan(dipRad))
				baseAngleYraw = np.arctan(np.cos(dipDirectionRad)*np.tan(dipRad))
				baseAngleX = abs(baseAngleXraw)
				baseAngleY = abs(baseAngleYraw)
				cosBaseAngleGamma = ((np.tan(baseAngleX))**2 + (np.tan(baseAngleY))**2 + 1)**(-0.5)
				#baseAngleX = abs(baseAngleXraw)
				#baseAngleY = abs(baseAngleYraw)

				# direction of sliding base inclination
				if analysisInput[loopCol][17] == 0:
					thetaDiff = 0
				else:
					thetaDiff = abs(directionOfSlidingBList[dirLoop]-analysisInput[loopCol][18])		
		
				dipSlidingDirectionRaw = np.arctan(np.tan(dipRad)*np.cos(np.radians(thetaDiff)))
				dipSlidingDirection = abs(dipSlidingDirectionRaw)

				# lever arm for moment analysis
				# find Ri and fi, which are distance of moment arm for shear strength and normal force
				RX = abs(centerPt0[0] - analysisInput[loopCol][1])
				RY = abs(centerPt0[1] - analysisInput[loopCol][2])
				RZ = abs(centerPt0[2] - analysisInput[loopCol][21])
				#RmidZ = abs(centerPt0[2] - 0.5*(analysisInput[loopCol][20] + analysisInput[loopCol][21]))

				angleBearing = angleBtw3Pts([(centerPt0[0], centerPt0[1], 0), (analysisInput[loopCol][1], analysisInput[loopCol][2], 0), (centerPt0[0], 100+centerPt0[1], 0)])%180
				angleXi = np.radians(abs(angleBearing[0] - (directionOfSlidingBList[dirLoop]%180)))
				if angleXi < 0.5*np.pi:
					xi = np.sqrt(RX**2 + RY**2)*np.sin(angleXi)	# horizontal distance of column weight from centerPt0
				else:
					xi = np.sqrt(RX**2 + RY**2)*np.cos(angleXi%(0.5*np.pi))	# horizontal distance of column weight from centerPt0
				
				# compute moment arm of Ri and fi
				#RiThetaXY = abs(np.degrees(np.atan(RZ/xi)))
				RR = np.sqrt(RX**2 + RY**2 + RZ**2)
				#RR = np.sqrt((RX**2 + RY**2 + RZ**2) - RY**2)
				deltaRangleXY = round(90 - (np.degrees(dipSlidingDirection) + abs(np.degrees(np.arctan(RZ/xi)))),2)

				Ri = RR*np.cos(np.radians(deltaRangleXY))
				fi = RR*np.sin(np.radians(deltaRangleXY))
				
				# base A
				baseA = (analysisInput[loopCol][14]/analysisInput[loopCol][16])#/np.cos(dipRad)

				Wi = analysisInput[loopCol][14]							# soil column weight
				ui = analysisInput[loopCol][5]							# pore-water pressure force
				matricSuction = analysisInput[loopCol][19]	
				Ei = 0													# applied line load
				E_d = 0													# moment arm of applied line load

				# Shear strength - initial
				ci = analysisInput[loopCol][12]							# base cohesion - Mohr Coloumb failure
				phiRad = np.radians(analysisInput[loopCol][13]) 		# base friction angle - Mohr Coloumb failure
				
				# Normal
				m_alpha = cosBaseAngleGamma + np.sin(dipSlidingDirection)*np.tan(phiRad)/FSm_i
				Ni = (Wi - (ci - ui*np.tan(phiRad))*(baseA*np.sin(dipSlidingDirection)/FSm_i))/m_alpha

				# change of phi and c with the new Ni 
				if analysisInput[loopCol][26] in [4,5]:
					newShear = shearModel2cphi(materialClass, analysisInput[loopCol][27], 0, 0, (Ni/baseA - ui), matricSuction)

					ci = newShear[0][1]
					phiRad = np.radians(newShear[0][0])
			
				# Moment equilibrium
				FSmNu += (ci*baseA + (Ni - ui*baseA)*np.tan(phiRad))*Ri
				FSmDeWx += Wi*xi
				FSmDeNf += Ni*fi*(cosBaseAngleGamma/np.cos(dipSlidingDirection))
				FSmDekWe += seismicKxy*Wi*RZ
				FSmDeEd += Ei*E_d
				#FSmDe += Wi*xi - Ni*fi*(cosBaseAngleGamma/np.cos(dipSlidingDirection)) + seismicKxy*Wi*RmidZ + Ei*E_d

			# computed FS	
			FS_c = FSmNu/(FSmDeWx - FSmDeNf + FSmDekWe + FSmDeEd)

			# compare computed and inital guess FS
			if iterationN >= iterationNMax:
				#print('too many iterations (iterationNN) - check code or increase maximum iteration number')
				iterationFS = False
				FSm_f = np.nan #'non-converging'
				FS_results.append([iterationN, directionOfSlidingBList[dirLoop], FSm_i, FS_c, FSm_f])

			elif abs(FS_c-FSm_i) <= tolFS:

				''' find decimal points of FS tolarance allowed '''
				if tolFS >= 1:
					decimalPoint = 1
				elif tolFS < 0.0001:
					dpListed = list(str(tolFS))
					idx = dpListed.index('-')
					dPstring = ''.join(dpListed[idx+1:])
					decimalPoint = int(dPstring)
				else:
					decimalPoint = len(list(str(tolFS)))-2	

				FSm_f = round(FS_c,decimalPoint+2)
				FS_results.append([iterationN, directionOfSlidingBList[dirLoop], FSm_i, FS_c, FSm_f])
				iterationN = 0
				iterationFS = False
			else:
				FSm_f = np.nan #'non-converging'
				#FS_results.append([iterationN, directionOfSlidingBList[dirLoop], FSm_i, FS_c, FSm_f])
				iterationN += 1
				FSm_i = FS_c
				#FSm_i -= changeFS


	if len(FS_results) > 1:
		FS_final_list =  listAtColNum(FS_results,4)

		FS_final_list_min = min(FS_final_list)
		FS_final_list_min_IDX = FS_final_list.index(FS_final_list_min)
		FS_final_info = FS_results[FS_final_list_min_IDX]

		return FS_final_info #, FS_results#[-1]
	else:
		return FS_results

# Analysis_3D_HungrJanbu1989_v4_06_28_2018.py
def analysis3DHungrJanbu1989(fileName, seismicK, materialClass, materialNameList, correctFS=None, iterationNMax=200, tolFS=0.0005, occuranceFactor=0.5, tolDirection_user=None, spacingDirection=0.5, avDipDirectionB_user=None, sideResistance=True):
	# import libraries
	import numpy as np
	from scipy.stats import mode
	
	# take the inputfile and convert it into list
	analysisInput =  csv2list(fileName)
	#print(analysisInput[0])

	# seismic coefficient
	seismicKx = seismicK[0]
	seismicKy = seismicK[1]
	seismicKxy = np.sqrt(seismicKx**2 + seismicKy**2)

	if sideResistance == True:
		XsortedIDList, XsortedIDXList, extXsortedIDList, extXsortedIDXList, exposedSidesDict = colIDExternal(analysisInput)

	''' direction of column dip direction '''
	#directionOfSlidingBList, directionOfSlidingPList = findGenSldiingDirection3D(analysisInput, occuranceFactor, tolDirection, spacingDirection)
	if avDipDirectionB_user == None:
		dipDirectionList =  listAtColNum(analysisInput, 18)	
		avDipDirectionB, countN = mode(np.array(dipDirectionList),axis=None)

		if countN >= occuranceFactor*len(analysisInput):
			avDipDirectionB = round(float(avDipDirectionB),1)
			tolDirection = 0
		else:
			avDipDirectionB = round(float(np.mean(dipDirectionList)),1)
			if tolDirection_user == None:
				tolDirection = 10
			else:
				tolDirection = tolDirection_user
	else:
		avDipDirectionB = avDipDirectionB_user
		if tolDirection_user == None:
			tolDirection = 0
		else:
			tolDirection = tolDirection_user

	# find sliding direction based on dip and dip direction - polar direction
	#print(avDipDirectionB)
	avDipDirectionP = 0
	if avDipDirectionB < 0:
		avDipDirectionB = avDipDirectionB + 360

	if avDipDirectionB > 360:
		avDipDirectionB = avDipDirectionB%360

	if avDipDirectionB >= 0 and avDipDirectionB < 270:
		avDipDirectionP = 90-avDipDirectionB
	elif avDipDirectionB >= 270 and avDipDirectionB <= 360:
		avDipDirectionP = 450-avDipDirectionB
	
	#avDipDirectionP = (450-avDipDirectionB)%360

	# store information of sliding direction
	if avDipDirectionB_user != None:
		directionOfSlidingBList = [avDipDirectionB]
		directionOfSlidingPList = [avDipDirectionP]
	else:
		directionOfSlidingBList =  making_float_list(avDipDirectionB-tolDirection, avDipDirectionB+tolDirection, spacingDirection)
		directionOfSlidingPList =  making_float_list(avDipDirectionP-tolDirection, avDipDirectionP+tolDirection, spacingDirection)

	''' FS computation '''
	FS_results = []

	for dirLoop in range(len(directionOfSlidingBList)):

		''' Janbu Correction Factor '''
		if correctFS == None:
			# factor b1
			sumPhiList = round(sum( listAtColNum(analysisInput,13)),2)
			sumCList = round(sum( listAtColNum(analysisInput,12)),2)
			if sumPhiList > 0 and sumCList > 0:
				b1 = 0.50
			elif sumPhiList > 0 and sumCList == 0:
				b1 = 0.31
			elif sumPhiList == 0 and sumCList > 0:
				b1 = 0.69

			# d and L factor
			
			Lx = abs(max( listAtColNum(analysisInput,1)) - min( listAtColNum(analysisInput,1)))/2
			Ly = abs(max( listAtColNum(analysisInput,2)) - min( listAtColNum(analysisInput,2)))/2

			L = 2*(Lx*Ly)/np.sqrt((Lx*np.sin(directionOfSlidingPList[dirLoop]))**2 + (Ly*np.cos(directionOfSlidingPList[dirLoop]))**2 )

			dZList = []
			for loopCol in range(len(analysisInput)):
				#baseDip = analysisInput[loopCol][17]

				dipRad = np.radians(analysisInput[loopCol][31])
				if analysisInput[loopCol][31] == 0:
					thetaDiff = 0 
				else:
					thetaDiff = abs(directionOfSlidingBList[dirLoop]-analysisInput[loopCol][32])
				apparantDip = abs(np.arctan(np.tan(dipRad)*np.cos(np.radians(thetaDiff))))

				dZ = abs(analysisInput[loopCol][21] - analysisInput[loopCol][20])
				dZList.append(dZ*np.cos(np.radians(apparantDip)))
			d = max(dZList)
			
			'''
			point1 = [newSliceN[min(sliceKeyList)][0], newSliceN[min(sliceKeyList)][1][0][0]]
			point2 = [newSliceN[max(sliceKeyList)][0], newSliceN[max(sliceKeyList)][3][0][0]]
			L_factor = np.sqrt((point1[1]-point2[1])**2 + (point1[0]-point2[0])**2)
			L_factor_gradient = (point1[1]-point2[1])/(point1[0]-point2[0])
			L_factor_intercept = point1[1] - point1[0]*L_factor_gradient

			dList = []
			for sliceN in sliceKeyList: 
				point3 = [0.5*(newSliceN[sliceN][0]+newSliceN[sliceN][2]), 0.5*(newSliceN[sliceN][1][0][0]+newSliceN[sliceN][3][0][0])]

				tempX = (L_factor_intercept-point3[1]-(point3[0]/L_factor_gradient))/(-L_factor_gradient-(1/L_factor_gradient))
				tempY = L_factor_gradient*tempX + L_factor_intercept

				tempDist = np.sqrt((point3[1]-tempY)**2 + (point3[0]-tempX)**2)
				dList.append(tempDist)
			d_factor = max(dList)
			'''
			'''
			Lx = abs(max( listAtColNum(analysisInput,1)) - min( listAtColNum(analysisInput,1)))
			Ly = abs(max( listAtColNum(analysisInput,2)) - min( listAtColNum(analysisInput,2)))
			Lxy = np.sqrt(Lx**2 + Ly**2)
			Llist = [Lx, Ly, Lxy]
			Lmin = min(Llist)
			Lindex = Llist.index(Lmin)
			Lz = abs(max( listAtColNum(analysisInput,20)) - min( listAtColNum(analysisInput,20)))
			LangleRad = np.arctan(Lz/Lmin)
			LangleDeg = np.degrees(np.arctan(Lz/Lmin))
			L = np.sqrt(Lmin**2 + Lz**2)

			ZbottomList =  listAtColNum(analysisInput,21)
			dZstart = min(ZbottomList)
			if Lindex == 0:
				dstart = min( listAtColNum(analysisInput,1))
				dIDX = 1
			elif Lindex == 1:
				dstart = min( listAtColNum(analysisInput,2))
				dIDX = 2
			elif Lindex == 2:
				dstartx = min( listAtColNum(analysisInput,1))
				dstarty = min( listAtColNum(analysisInput,2))
				dIDX = 3

			dList = []
			for loopZBot in range(len(ZbottomList)):
				if dIDX in [1,2]:
					dLen = max(abs(analysisInput[loopZBot][dIDX] - dstart), 0.0001)
				elif dIDX == 3:
					dLen = max(np.sqrt((analysisInput[loopZBot][1] - dstartx)**2 + (analysisInput[loopZBot][2] - dstarty)**2), 0.0001)


				dZ = analysisInput[loopZBot][20] - dZstart
				dangleRad = LangleRad - np.arctan(dZ/dLen)	

				dList.append(abs(dLen*np.sin(dangleRad)))
			d = max(dList)
			'''
			correctionFactor = 1 + b1*((d/L) - 1.4*((d/L)**2))
			#print('correction factor = %f'%correctionFactor)
		else:
			correctionFactor = correctFS

		iterationN = 1
		FS_i = 1			#inital guess of FS
		sumW = 0
		sumV = 0 
		
		# iterate through to find global 3D FS
		iterationFS = True
		while iterationFS:

			FSNu = 0
			Qs = 0
			#FSDe = 0
			FSDeNcosa = 0
			FSDekW = 0
			FSDeE = 0

			for loopCol in range(len(analysisInput)):
				#print(analysisInput[loopCol])
				#print('columnID=%i'%loopCol)
				
				# base inclination
				# inclination of base relative to the positive x (right) and positive y (up) direction
				dipRad = np.radians(analysisInput[loopCol][17])

				if analysisInput[loopCol][17] == 0:
					dipDirectionRad = np.radians(directionOfSlidingBList[dirLoop])
				else:
					dipDirectionRad = np.radians(analysisInput[loopCol][18])

				#baseAngleX = np.radians(round(np.degrees(np.arctan(np.sin(dipDirectionRad)*np.tan(dipRad))),2))
				#baseAngleY = np.radians(round(np.degrees(np.arctan(np.cos(dipDirectionRad)*np.tan(dipRad))),2))
				baseAngleX = np.arctan(np.sin(dipDirectionRad)*np.tan(dipRad))
				baseAngleY = np.arctan(np.cos(dipDirectionRad)*np.tan(dipRad))
				baseAngleX = abs(baseAngleX)
				baseAngleY = abs(baseAngleY)
				cosBaseAngleGamma = np.cos(dipRad) #((np.tan(baseAngleX))**2 + (np.tan(baseAngleY))**2 + 1)**(-0.5)

				# direction of sliding base inclination
				if analysisInput[loopCol][17] == 0:
					thetaDiff = 0
				else:
					thetaDiff = abs(directionOfSlidingBList[dirLoop]-analysisInput[loopCol][18])
					
				#print('thetaDiff=%f'%thetaDiff)
				dipSlidingDirection = abs(np.arctan(np.tan(dipRad)*np.cos(np.radians(thetaDiff))))
				#print(thetaDiff)
				#print(dipSlidingDirection*180/np.pi)

				# base A
				baseA = (analysisInput[loopCol][14]/analysisInput[loopCol][16]) #/np.cos(dipRad)

				Wi = analysisInput[loopCol][14]							# soil column weight
				sumW += Wi
				sumV += analysisInput[loopCol][15]	
				ui = analysisInput[loopCol][5]							# pore-water pressure 
				matricSuction = analysisInput[loopCol][19]
				Ei = 0													# applied line load
				
				# Shear strength - initial
				ci = analysisInput[loopCol][12]							# base cohesion - Mohr Coloumb failure
				phiRad = np.radians(analysisInput[loopCol][13]) 		# base friction angle - Mohr Coloumb failure
				
				# Normal
				m_alpha = cosBaseAngleGamma + np.sin(dipSlidingDirection)*np.tan(phiRad)/FS_i
				Ni = (Wi - (ci*baseA - ui*baseA*np.tan(phiRad))*(np.sin(dipSlidingDirection)/FS_i))/m_alpha

				# change of phi and c with the new Ni 
				if analysisInput[loopCol][26] in [4,5]:
					newShear = shearModel2cphi(materialClass, materialNameList[analysisInput[loopCol][27]-1], 0, 0, (Ni/baseA - ui), matricSuction)

					ci = newShear[0][1]
					phiRad = np.radians(newShear[0][0])

				# side resistance
				if sideResistance == True:
					sideA = 0
					if analysisInput[loopCol][0] in extXsortedIDXList:
						K0 = 1-np.sin(phiRad)
						Ka = (1-np.sin(phiRad))/(1+np.sin(phiRad))
						K_tau = 0.5*(K0+Ka)
						c_im = K_tau*((Ni/baseA)-ui)*np.tan(phiRad)

						sideAreaExposed = exposedSidesDict[analysisInput[loopCol][0]]

						if (315 <= directionOfSlidingBList[dirLoop] <= 360) and (45 > directionOfSlidingBList[dirLoop] >= 0):
							if sideAreaExposed[1]:
								sideA += analysisInput[loopCol][23]
							if sideAreaExposed[3]: 
								sideA += analysisInput[loopCol][25]

						elif 45 <= directionOfSlidingBList[dirLoop] < 135:
							if sideAreaExposed[0]:
								sideA += analysisInput[loopCol][22]
							if sideAreaExposed[2]: 
								sideA += analysisInput[loopCol][24]

						elif 135 <= directionOfSlidingBList[dirLoop] < 225:
							if sideAreaExposed[1]:
								sideA += analysisInput[loopCol][23]
							if sideAreaExposed[3]: 
								sideA += analysisInput[loopCol][25]

						elif 225 <= directionOfSlidingBList[dirLoop] < 315:
							if sideAreaExposed[0]:
								sideA += analysisInput[loopCol][22]
							if sideAreaExposed[2]: 
								sideA += analysisInput[loopCol][24]

						Qs += c_im*sideA

				# force equilibrium
				FSNu += (ci*baseA + (Ni - ui*baseA)*np.tan(phiRad))*np.cos(dipSlidingDirection)
				#FSDe += Ni*cosBaseAngleGamma*np.tan(dipSlidingDirection) + seismicKxy*Wi + Ei
				FSDeNcosa += Ni*cosBaseAngleGamma*np.tan(dipSlidingDirection)
				FSDekW += seismicKxy*Wi
				FSDeE += Ei
		
			# computed FS	
			FS_c = (Qs+FSNu)/(FSDeNcosa + FSDekW + FSDeE)

			# compare computed and inital guess FS
			if iterationN >= iterationNMax:
				#print('too many iterations (iterationNN) - check code or increase maximum iteration number')
				iterationFS = False
				FS_f = np.nan #'non-converging'
				FS_results.append([iterationN, directionOfSlidingBList[dirLoop], FS_i, FS_c, correctionFactor, FS_f])
			elif abs(FS_c-FS_i) <= tolFS:

				''' find decimal points of FS tolarance allowed '''
				if tolFS >= 1:
					decimalPoint = 1
				elif tolFS < 0.0001:
					dpListed = list(str(tolFS))
					idx = dpListed.index('-')
					dPstring = ''.join(dpListed[idx+1:])
					decimalPoint = int(dPstring)
				else:
					decimalPoint = len(list(str(tolFS)))-2	

				FS_f = round(FS_c*correctionFactor, decimalPoint+2)
				FS_results.append([iterationN, directionOfSlidingBList[dirLoop], FS_i, FS_c, correctionFactor, FS_f])
				iterationN = 0
				iterationFS = False
			else:
				FS_f = np.nan #'non-converging'
				#FS_results.append([iterationN, directionOfSlidingBList[dirLoop], FS_i, FS_c, correctionFactor, FS_f])
				iterationN += 1
				FS_i = FS_c
				#FS_i -= changeFS

	#print(sumW)
	#print(sumV)
	if len(FS_results) > 1:
		FS_final_list =  listAtColNum(FS_results,5)
		
		FS_final_list_min = min(FS_final_list)
		FS_final_list_min_IDX = FS_final_list.index(FS_final_list_min)
		FS_final_info = FS_results[FS_final_list_min_IDX]
	
		return FS_final_info #, FS_results # None  #  FS_results[-1]
	else:
		return FS_results


'''change the interslice angle Lambda based on the difference of FS'''

def changeLambda3D(scaleLambdaXList, scaleLambdaYList, FS_List, tolaranceFS):
	from numpy import mean, nan
	from numpy.random import choice

	# create total number of change criteria based on decimal points   
	if tolaranceFS >= 1:
		decimalPoint = 1
	elif tolaranceFS <= 0.0001:
		decimalPoint = 4
		tolaranceFS = 0.0001
	else:
		decimalPoint = len(list(str(tolaranceFS)))-2
		
	dFSLimList = [1]
	for loop1 in range(decimalPoint):
		if loop1 == decimalPoint-1:
			dFSLimList.append(tolaranceFS)
		#elif tolaranceFS >= 0.0001 and loop1 == decimalPoint-2:
		#	dFSLimList.append(tolaranceFS*5)
		else:
			dFSLimList.append(0.1*float('1E-'+str(loop1)))

	# change the interslice force angle
	completeValueChangeSet = [1, 0.5, 0.1, 0.05] # [2, 1, 0.5, 0.1] # 
	valueChangeList = completeValueChangeSet[:(decimalPoint)]
	
	# error condition
	if nan in FS_List[-1]:
		'''
		randomNum = choice(range(100))
		if (randomNum%2) == 0: # even number
			scaleLambdaXList.append(scaleLambdaXList[-1])
			scaleLambdaYList.append(scaleLambdaYList[-1] + 0.1)
		else:  # odd number
			scaleLambdaXList.append(scaleLambdaXList[-1] + 0.1)
			scaleLambdaYList.append(scaleLambdaYList[-1])
		'''
		if scaleLambdaXList[-1] >= scaleLambdaYList[-1]:
			scaleLambdaXList.append(scaleLambdaXList[-1])
			scaleLambdaYList.append(scaleLambdaYList[-1] + 0.1)
		elif scaleLambdaXList[-1] < scaleLambdaYList[-1]:
			scaleLambdaXList.append(scaleLambdaXList[-1] + 0.1)
			scaleLambdaYList.append(scaleLambdaYList[-1])
		
		return scaleLambdaXList, scaleLambdaYList

	# dFS - [FSmx_f, FSmy_f, FSx_f, FSy_f]
	dFS_m = abs(FS_List[-1][0] - FS_List[-1][1])
	dFS_f = abs(FS_List[-1][2] - FS_List[-1][3])

	if dFS_m > tolaranceFS and dFS_f > tolaranceFS:
		dFS_fm = (max(FS_List[-1][0:2]) - max(FS_List[-1][2:]))
	elif dFS_m <= tolaranceFS and dFS_f > tolaranceFS:
		dFS_fm = (mean(FS_List[-1][0:2]) - max(FS_List[-1][2:]))
	elif dFS_m > tolaranceFS and dFS_f <= tolaranceFS:
		dFS_fm = (max(FS_List[-1][0:2]) - mean(FS_List[-1][2:]))
	else:
		dFS_fm = (mean(FS_List[-1][0:2]) - mean(FS_List[-1][2:]))

	# changing Lambda higher or lower value
	absDiffFS = abs(dFS_fm)
	print(absDiffFS)

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

	# initial condition - no intercolumn force computed
	if len(scaleLambdaXList) == 1 and len(scaleLambdaYList) == 1:
		scaleLambdaXList.append(scaleLambdaXList[-1] + valueChange)
		scaleLambdaYList.append(scaleLambdaYList[-1])
	
	# not initial condition
	else:
		if nan in FS_List[-2]:
			if scaleLambdaXList[-1] >= scaleLambdaYList[-1]:
				scaleLambdaXList.append(scaleLambdaXList[-1])
				scaleLambdaYList.append(scaleLambdaYList[-1] + valueChange)
			elif scaleLambdaXList[-1] < scaleLambdaYList[-1]:
				scaleLambdaXList.append(scaleLambdaXList[-1] + valueChange)
				scaleLambdaYList.append(scaleLambdaYList[-1])
			return scaleLambdaXList, scaleLambdaYList
			
		else:
			dFS_mO = abs(FS_List[-2][0] - FS_List[-2][1])
			dFS_fO = abs(FS_List[-2][2] - FS_List[-2][3])

			if dFS_mO > tolaranceFS and dFS_fO > tolaranceFS:
				dFS_fmO = (max(FS_List[-2][0:2]) - max(FS_List[-2][2:]))
			elif dFS_mO <= tolaranceFS and dFS_fO > tolaranceFS:
				dFS_fmO = (mean(FS_List[-2][0:2]) - max(FS_List[-2][2:]))
			elif dFS_mO > tolaranceFS and dFS_fO <= tolaranceFS:
				dFS_fmO = (max(FS_List[-2][0:2]) - mean(FS_List[-2][2:]))
			else:
				dFS_fmO = (mean(FS_List[-2][0:2]) - mean(FS_List[-2][2:]))

			absDiffFS_old = abs(dFS_fmO)
			print(absDiffFS_old)

			dlambX = scaleLambdaXList[-2] - scaleLambdaXList[-1]
			dlambY = scaleLambdaYList[-2] - scaleLambdaYList[-1]

			# deviation of FS due to lambda
			deviationCheck1 = ((dFS_mO <= tolaranceFS) and (dFS_m > tolaranceFS)) or ((dFS_fO <= tolaranceFS) and (dFS_f > tolaranceFS))
			deviationCheck2 = ((dFS_m > 0.3) or (dFS_f > 0.3)) 
			if deviationCheck1 or deviationCheck2:
				'''
				if abs(scaleLambdaXList[-1]) >= abs(scaleLambdaYList[-1]):
					scaleLambdaXList.append(scaleLambdaXList[-1])
					scaleLambdaYList.append(scaleLambdaXList[-1])
				elif abs(scaleLambdaXList[-1]) < abs(scaleLambdaYList[-1]):
					scaleLambdaXList.append(scaleLambdaYList[-1])
					scaleLambdaYList.append(scaleLambdaYList[-1])
				'''
				scaleLambdaXList.append(max([scaleLambdaYList[-1],scaleLambdaXList[-1]]))
				scaleLambdaYList.append(max([scaleLambdaYList[-1],scaleLambdaXList[-1]]))
		
			elif dlambX == 0 and dlambY != 0:
				# if changing lambdaY improves dFS change lambdaY, if not change lambdaX
				if absDiffFS_old < absDiffFS:
					scaleLambdaXList.append(scaleLambdaXList[-1])
					scaleLambdaYList.append(scaleLambdaYList[-1] + valueChange)
				else:
					scaleLambdaXList.append(scaleLambdaXList[-1] + valueChange)
					scaleLambdaYList.append(scaleLambdaYList[-1])
			
			# not initial condition
			elif dlambX != 0 and dlambY == 0:
				# if changing lambdaX improves dFS change lambdaX, if not change lambdaY
				if absDiffFS_old < absDiffFS:
					scaleLambdaXList.append(scaleLambdaXList[-1] + valueChange)
					scaleLambdaYList.append(scaleLambdaYList[-1])
				else:
					scaleLambdaXList.append(scaleLambdaXList[-1])
					scaleLambdaYList.append(scaleLambdaYList[-1] + valueChange)

	return scaleLambdaXList, scaleLambdaYList

'''
method - 3D analysis method used:
	1 = morgenstern-price -> half-sine function
	2 = spencer
	3 = bishop
	4 = janbu
'''
# intial values of sliding direction
def analysis3DChengnYip2007_initialSlidingDir(avDipDirectionB_user, occuranceFactor, analysisInput, spacingDirection, tolDirection_user):
	import numpy as np 
	from scipy.stats import mode

	# find sliding direction based on dip and dip direction - bearing direction
	if avDipDirectionB_user == None:
		dipDirectionList =  listAtColNum(analysisInput, 18)	
		avDipDirectionB, countN = mode(np.array(dipDirectionList),axis=None)

		if countN >= occuranceFactor*len(analysisInput):
			avDipDirectionB = round(float(avDipDirectionB)*2)/2		#round to nearest 0.5
			tolDirection = 0
		else:
			avDipDirectionB = round(float(np.mean(dipDirectionList))*2)/2		#round to nearest 0.5
			if tolDirection_user == None:
				tolDirection = 5
			else:
				tolDirection = tolDirection_user
	else:
		avDipDirectionB = avDipDirectionB_user
		if tolDirection_user == None:
			tolDirection = 0
		else:
			tolDirection = tolDirection_user
	
	# find sliding direction based on dip and dip direction - polar direction
	#print(avDipDirectionB)
	avDipDirectionP = 0
	if avDipDirectionB < 0:
		avDipDirectionB = avDipDirectionB + 360

	if avDipDirectionB > 360:
		avDipDirectionB = avDipDirectionB%360

	if avDipDirectionB >= 0 and avDipDirectionB < 270:
		avDipDirectionP = 90-avDipDirectionB
	elif avDipDirectionB >= 270 and avDipDirectionB <= 360:
		avDipDirectionP = 450-avDipDirectionB
	
	#avDipDirectionP = (450-avDipDirectionB)%360

	# store information of sliding direction
	if avDipDirectionB_user != None:
		directionOfSlidingBList = [avDipDirectionB]
		directionOfSlidingPList = [avDipDirectionP]
	else:
		directionOfSlidingBList =  making_float_list(avDipDirectionB-tolDirection, avDipDirectionB+tolDirection, spacingDirection)
		directionOfSlidingPList =  making_float_list(avDipDirectionP-tolDirection, avDipDirectionP+tolDirection, spacingDirection)

	# find shearing direction, i.e. opposite direction of sliding direction 
	#directionOfShearBList = []
	directionOfShearPList = []
	for dirSlBLoop in range(len(directionOfSlidingBList)):
		shearDB = (directionOfSlidingBList[dirSlBLoop] + 180)%360

		if shearDB >= 0 and shearDB < 270:
			shearDP = 90-shearDB
		elif shearDB >= 270 and shearDB <= 360:
			shearDP = 450-shearDB

		#directionOfShearBList.append(shearDB)
		directionOfShearPList.append(shearDP)

	return directionOfSlidingBList, directionOfSlidingPList, directionOfShearPList #,directionOfShearBList

# function to calculate corrected Janbu FS factors
def analysis3DChengnYip2007_correctFSfactor(correctFS, analysisInput, slidingDiP, slidingDiB):
	import numpy as np 

	if correctFS == None:
		# factor b1
		sumPhiList = round(sum( listAtColNum(analysisInput,13)),2)
		sumCList = round(sum( listAtColNum(analysisInput,12)),2)
		if sumPhiList > 0 and sumCList > 0:
			b1 = 0.5
		elif sumPhiList > 0 and sumCList == 0:
			b1 = 0.31
		elif sumPhiList == 0 and sumCList > 0:
			b1 = 0.69

		# d and L factor
		#xxx =  listAtColNum(analysisInput,1)
		#yyy =  listAtColNum(analysisInput,2)
		#print([xxx, yyy])

		Lx = abs(max( listAtColNum(analysisInput,1)) - min( listAtColNum(analysisInput,1)))/2
		Ly = abs(max( listAtColNum(analysisInput,2)) - min( listAtColNum(analysisInput,2)))/2

		L = 2*(Lx*Ly)/np.sqrt((Ly*np.cos(slidingDiP))**2 + (Lx*np.sin(slidingDiP))**2 )

		dZList = []
		for loopCol in range(len(analysisInput)):
			#baseDip = analysisInput[loopCol][17]

			dipRad = np.radians(analysisInput[loopCol][31])
			if analysisInput[loopCol][31] == 0:
				thetaDiff = 0 
			else:
				thetaDiff = abs(slidingDiB-analysisInput[loopCol][32])
			apparantDip = abs(np.arctan(np.tan(dipRad)*np.cos(np.radians(thetaDiff))))

			dZ = abs(analysisInput[loopCol][21] - analysisInput[loopCol][20])
			dZList.append(dZ*np.cos(np.radians(apparantDip)))
		d = max(dZList)

		#print(Lx, Ly, slidingDiP, L, d)

		'''
		Lx = abs(max( listAtColNum(analysisInput,1)) - min( listAtColNum(analysisInput,1)))
		Ly = abs(max( listAtColNum(analysisInput,2)) - min( listAtColNum(analysisInput,2)))
		Lxy = np.sqrt(Lx**2 + Ly**2)
		Llist = [Lx, Ly, Lxy]
		Lmin = min(Llist)
		Lindex = Llist.index(Lmin)
		Lz = abs(max( listAtColNum(analysisInput,20)) - min( listAtColNum(analysisInput,21)))
		LangleRad = np.arctan(Lz/Lmin)
		LangleDeg = np.degrees(np.arctan(Lz/Lmin))
		L = np.sqrt(Lmin**2 + Lz**2)

		print([Lx, Ly, Lxy, Lmin, Lz, LangleRad, LangleDeg, L])
		print(b1)

		ZbottomList =  listAtColNum(analysisInput,21)
		dZstart = min(ZbottomList)
		if Lindex == 0:
			dstart = min( listAtColNum(analysisInput,1))
			dIDX = 1
		elif Lindex == 1:
			dstart = min( listAtColNum(analysisInput,2))
			dIDX = 2
		elif Lindex == 2:
			dstartx = min( listAtColNum(analysisInput,1))
			dstarty = min( listAtColNum(analysisInput,2))
			dIDX = 3

		dList = []
		for loopZBot in range(len(ZbottomList)):
			if dIDX in [1,2]:
				dLen = max(abs(analysisInput[loopZBot][dIDX] - dstart), 0.0001)
			elif dIDX == 3:
				dLen = max(np.sqrt((analysisInput[loopZBot][1] - dstartx)**2 + (analysisInput[loopZBot][2] - dstarty)**2), 0.0001)

			dZ = analysisInput[loopZBot][20] - dZstart
			dangleRad = LangleRad - np.arctan(dZ/dLen)	

			dList.append(abs(dLen*np.sin(dangleRad)))
		d = max(dList)
		correctionFactor = 1 + b1*((d/L) - 1.4*((d/L)**2))
		'''
		correctionFactor = 1 + b1*((d/L) - 1.4*((d/L)**2))
		#print('correction factor = %f'%correctionFactor)
	else:
		correctionFactor = correctFS

	return correctionFactor

def analysis3DChengnYip2007(fileName, seismicK, centerPt0, method, materialClass, lambdaIteration=None, iterationNMax=100, tolFS=0.001, correctFS=None, occuranceFactor=0.5, tolDirection_user=None, spacingDirection=0.5, avDipDirectionB_user=None):
	# import libraries
	import numpy as np
	
	# take the inputfile and convert it into list
	analysisInput =  csv2list(fileName)
	#print(analysisInput[0])

	# seismic coefficient
	seismicKx = seismicK[0]
	seismicKy = seismicK[1]

	''' direction of column dip direction '''
	directionOfSlidingBList, directionOfSlidingPList, directionOfShearPList = analysis3DChengnYip2007_initialSlidingDir(avDipDirectionB_user, occuranceFactor, analysisInput, spacingDirection, tolDirection_user)

	#changeFS = 0.005
	#changeFS = 0.01
	changeLambda = 0.1

	FS_results = []
	loopFS = int(0)

	iterationN_a = 0
	iterationN_FS = 0

	FSg_cur = np.nan
	FSg_IDX = np.nan

	# iterate through sliding direction 
	for loopSDir in range(len(directionOfSlidingBList)):
		#print(directionOfSlidingBList[loopSDir])
		shearDirectionRadP = np.radians(directionOfShearPList[loopSDir])
		slidingDirection = directionOfSlidingBList[loopSDir]
		slidingDirectionP = directionOfSlidingPList[loopSDir]

		# intercolumn function
		if method == 4: # simplified Janbu method
			Fxy = 0		# intercoloum function
			lambdaXY = 0 	# horizontal direction intercoloum force scaling factor
			scaleLambdaXList = [0] # making_float_list(-1, 3, 0.05)
			scaleLambdaYList = [0] # making_float_list(-1, 3, 0.05)
			correctionFactor = analysis3DChengnYip2007_correctFSfactor(correctFS, analysisInput, slidingDirectionP, slidingDirection)
		elif method == 3: # simplified bishop method
			Fxy = 0		# intercoloum function
			lambdaXY = 0 	# horizontal direction intercoloum force scaling factor
			scaleLambdaXList = [0] # making_float_list(-1, 3, 0.05)
			scaleLambdaYList = [0] # making_float_list(-1, 3, 0.05)
		elif method == 2: # spencer method
			Fxy = 1		# intercoloum function
			lambdaXY = 0 	# horizontal direction intercoloum force scaling factor
			if lambdaIteration == None:
				scaleLambdaXList = [0] #[-1]
				scaleLambdaYList = [0] #[-1]
			elif lambdaIteration != None:
				scaleLambdaXList = [lambdaIteration]
				scaleLambdaYList = [lambdaIteration]
		elif method == 1:  # Morgenstern-Price method
			lambdaXY = 0 	# horizontal direction intercoloum force scaling factor
			if lambdaIteration == None:
				scaleLambdaXList = [0] #[-1]
				scaleLambdaYList = [0] #[-1]
			elif lambdaIteration != None:
				scaleLambdaXList = [lambdaIteration]
				scaleLambdaYList = [lambdaIteration]

		angleDatas = [] 	# store information of base angles for a given direction of sliding
		fVectorDatas = []	# store information of unit vector of shear, i.e. [f1, f2, f3], for a given direction of sliding
		gVectorDatas = []	# store information of unit vector of normal, i.e. [g1, g2, g3], for a given direction of sliding
		dExDatas = []
		dEyDatas = []
		QDatas = []

		FS_results_Temp = []
		tempFSforceList = []
		tempFSmomentList = []

		# iterate through intercolumn scaling factors for x 
		iterationLambda = True
		while iterationLambda:	

			#final computed of FS
			FSx_f = 0
			FSy_f = 0
			FSmx_f = 0
			FSmy_f = 0

			#inital trial of FS
			FSx_i = 1
			FSy_i = 1 
			FSmx_i = 1
			FSmy_i = 1
		   
			# iterate through to find global 3D FS
			iterationFS = True
			counts = 0
			while iterationFS:

				# calculating the FS 
				
				FSxNu = 0
				#FSxDe = 0
				FSxDeAg1 = 0
				FSxDeQx = 0
				FSxDekxW = 0
				FSxDeDHx = 0

				FSyNu = 0
				#FSyDe = 0
				FSyDeAg2 = 0
				FSyDeQy = 0
				FSyDekyW = 0
				FSyDeDHy = 0

				FSmxNu = 0
				#FSmxDe = 0
				FSmxDeWRY = 0
				FSmxDeKyWRZ = 0
				FSmxDeNig2RZ = 0
				FSmxDeNig3RY = 0
				FSmxDeQmx = 0

				FSmyNu = 0
				#FSmyDe = 0
				FSmyDeWRx = 0
				FSmyDeKxWRZ = 0
				FSmyDeNig2RZ = 0
				FSmyDeNig3RX = 0
				FSmyDeQmy = 0
			
				for loopCol in range(len(analysisInput)):
					#print(analysisInput[loopCol])

					if iterationN_FS == 0:
						# base inclination
						# inclination of base relative to the positive x (right) and positive y (up) direction
						dipRad = np.radians(analysisInput[loopCol][17])
						if analysisInput[loopCol][17] == 0:
							dipDirectionRad = np.radians(directionOfSlidingBList[loopSDir])
						else:
							dipDirectionRad = np.radians(analysisInput[loopCol][18])
						baseAngleX = abs(np.arctan(np.sin(dipDirectionRad)*np.tan(dipRad)))
						baseAngleY = abs(np.arctan(np.cos(dipDirectionRad)*np.tan(dipRad)))

						angleDatas.append([shearDirectionRadP, loopCol, baseAngleX, baseAngleY])
						
						if analysisInput[loopCol][17] == 0:
							thetaDiff_f = abs(slidingDirection-directionOfSlidingBList[loopSDir])
						else:
							thetaDiff_f = abs(slidingDirection-analysisInput[loopCol][18])
						baseAngleSliding_f = abs(np.arctan(np.tan(dipRad)*np.cos(np.radians(thetaDiff_f))))

						f1 = -np.cos(baseAngleSliding_f)*np.sin(np.radians(slidingDirection))
						f2 = -np.cos(baseAngleSliding_f)*np.cos(np.radians(slidingDirection))
						f3 = np.sin(baseAngleSliding_f)
						
						fVectorDatas.append([f1,f2,f3])

						# unit vectors for normal force (N)
						
						J = np.sqrt((np.tan(baseAngleX))**2 + (np.tan(baseAngleY))**2 + 1)
						g1 = np.tan(baseAngleX)/J
						g2 = np.tan(baseAngleY)/J
						g3 = 1/J
						#g_check = g1**2 + g2**2 + g3**2
						#print([loopCol, g1, g2, g3, g_check])
						'''
						dipRad_g = 0.5*np.pi - dipRad
						g1 = -np.cos(dipRad_g)*np.cos(np.radians(dipDirectionRad))
						g2 = -np.cos(dipRad_g)*np.sin(np.radians(dipDirectionRad))
						g3 = np.sin(dipRad_g)
						'''
						gVectorDatas.append([g1,g2,g3])

						# intercolumn force
						dEx = 0
						dEy = 0		
						dExDatas.append(dEx)
						dEyDatas.append(dEy)

						# externally applied forces and moments
						Qxi = 0
						Qyi = 0
						Qzi = 0
						Qmx = 0
						Qmy = 0
						Qmz = 0
						Qmz_local = 0
						QDatas.append([Qxi, Qyi, Qzi, Qmx, Qmy, Qmz, Qmz_local])

					else: 
						baseAngleX = angleDatas[loopCol][2]
						baseAngleY = angleDatas[loopCol][3]
						#baseAngleTheta = angleDatas[loopCol][4]
						#baseAngleSliding = angleDatas[loopCol][5]

						f1 = fVectorDatas[loopCol][0]
						f2 = fVectorDatas[loopCol][1]
						f3 = fVectorDatas[loopCol][2]

						g1 = gVectorDatas[loopCol][0]
						g2 = gVectorDatas[loopCol][1]
						g3 = gVectorDatas[loopCol][2]
						
						if method in [3,4]:
							dEx = 0
							dEy = 0			
						elif method in [1,2]:
							dEx = dExDatas[loopCol]
							dEy = dEyDatas[loopCol]	

						Qxi = QDatas[loopCol][0]
						Qyi = QDatas[loopCol][1]
						Qzi = QDatas[loopCol][2]
						Qmx = QDatas[loopCol][3]
						Qmy = QDatas[loopCol][4]
						Qmz = QDatas[loopCol][5]
						Qmz_local = QDatas[loopCol][6]
					

					# base area
					baseA = (analysisInput[loopCol][14]/analysisInput[loopCol][16])#/np.cos(dipRad)
					#columnWidth = np.sqrt(baseArea)

					# forces
					Ui = baseA*analysisInput[loopCol][5]	
					matricSuction = analysisInput[loopCol][19]
					Wi = analysisInput[loopCol][14]

					# inital strength
					Ci = baseA*analysisInput[loopCol][12]
					phiRad = np.radians(analysisInput[loopCol][13])	

					# intercolumn forces
					lambdaX = scaleLambdaXList[-1] 
					lambdaY = scaleLambdaYList[-1]
					dHx = Fxy*lambdaX*dEx
					dHy = Fxy*lambdaY*dEy
					#dXx = Fxy*lambdaXY*dEx
					#dXy = Fxy*lambdaXY*dEy
					#print([dEx, dEy, dHx, dHy, dXx, dXy])

					# Shear Force coefficients based on vertical equilibrium (\Sum(Fv) = 0)
					Ai = (lambdaX*dEx + lambdaY*dEy + Wi + Qzi)/g3
					Bi = -f3/g3

					# new normal force  
					Si_mx = (Ci + (Ai - Ui)*np.tan(phiRad))/(FSmx_i-(Bi*np.tan(phiRad)))
					Si_my = (Ci + (Ai - Ui)*np.tan(phiRad))/(FSmy_i-(Bi*np.tan(phiRad)))

					Ni_mx = Ai + Bi*Si_mx
					Ni_my = Ai + Bi*Si_my

					# change of phi and c with the new Ni 
					if analysisInput[loopCol][26] in [4,5]:
						Ni = 0.5*(Ni_my+Ni_mx)
						newShear = shearModel2cphi(materialClass, analysisInput[loopCol][27], 0, 0, (Ni/baseA - Ui/baseA), matricSuction)

						Ci = baseA*newShear[0][1]
						phiRad = np.radians(newShear[0][0])

					'''
					print('columnID=%i'%loopCol)
					print('Ai = %f'%Ai)
					print('Bi = %f'%Bi)
					'''
					# Janbu's method - force equilibrium method						
					if method in [1,2,4]:

						# Force equilibrium - X direction
						Axi = (Ci + (Ai - Ui)*np.tan(phiRad))/(1-(Bi*np.tan(phiRad)/FSx_i))
						FSxNu += Axi*(f1-(Bi*g1))
						#FSxDe += Ai*g1 - Qxi + seismicKx*Wi - dHx + dEx
						FSxDeAg1 += Ai*g1 
						FSxDeQx += Qxi
						FSxDekxW += seismicKx*Wi
						FSxDeDHx += dHx
						#print([Axi,  Axi*(f1-Bi*g1), Ai*g1 - Qxi + seismicKx*Wi - dHx + dEx])

						# Force equilibrium - Y direction
						Ayi = (Ci + (Ai - Ui)*np.tan(phiRad))/(1-(Bi*np.tan(phiRad)/FSy_i))
						FSyNu += Ayi*(f2-(Bi*g2))
						#FSyDe += Ai*g2 - Qyi + seismicKy*Wi - dHy + dEy
						FSyDeAg2 += Ai*g2 
						FSyDeQy += Qyi
						FSyDekyW += seismicKy*Wi
						FSyDeDHy += dHy
						#print([Ayi,  Ayi*(f2-Bi*g2), Ai*g2 - Qyi + seismicKy*Wi - dHy + dEy])
						

					if method in [1,2,3]:
						# lever arm for moment analysis
						RX = abs(centerPt0[0] - analysisInput[loopCol][1])
						RY = abs(centerPt0[1] - analysisInput[loopCol][2])
						#RZ = abs(centerPt0[2] - 0.5*(analysisInput[loopCol][20] + analysisInput[loopCol][21]))
						RZ = abs(centerPt0[2] - analysisInput[loopCol][21])
						
						Si_mx = (Ci + (Ai - Ui)*np.tan(phiRad))/(FSmx_i-(Bi*np.tan(phiRad)))
						Si_my = (Ci + (Ai - Ui)*np.tan(phiRad))/(FSmy_i-(Bi*np.tan(phiRad)))

						Ni_mx = Ai + Bi*Si_mx
						Ni_my = Ai + Bi*Si_my

						# Moment equilibrium - XX
						Kmxi = (Ci + (Ai - Ui)*np.tan(phiRad))/(1-(Bi*np.tan(phiRad)/FSmx_i))
						FSmxNu += Kmxi*(f2*RZ+f3*RY)
						#FSmxDe += Wi*(RY + seismicKy*RZ) + Ni_mx*(g2*RZ - g3*RY) + Qmx
						FSmxDeWRY += Wi*RY
						FSmxDeKyWRZ += Wi*seismicKy*RZ
						FSmxDeNig2RZ += Ni_mx*g2*RZ
						FSmxDeNig3RY += Ni_mx*g3*RY
						FSmxDeQmx += Qmx

						# Moment equilibrium - YY
						Kmyi = (Ci + (Ai - Ui)*np.tan(phiRad))/(1-(Bi*np.tan(phiRad)/FSmy_i))
						FSmyNu += Kmyi*(f1*RZ+f3*RX)
						#FSmyDe += Wi*(RX + seismicKx*RZ) + Ni_my*(g2*RZ - g3*RX) + Qmy
						FSmyDeWRx += Wi*RX
						FSmyDeKxWRZ += Wi*seismicKx*RZ
						FSmyDeNig2RZ += Ni_my*g2*RZ
						FSmyDeNig3RX += Ni_my*g3*RX
						FSmyDeQmy += Qmy
						
						'''
						# Force equilibrium - X direction
						Kxi = (Ci + (((Wi + Qzi)/g3) - Ui)*np.tan(phiRad))/(1 + (f3*np.tan(phiRad))/(g3*FSmx_i))
						FSmxNu += Kxi*(f2*RZ + f3*RY)
						FSmxDe += (Wi + Qzi)*RY + Ni_mx*(g2*RZ - g3*RY)
						#print([Axi, Axi*(f1+(f3*g1)/g3), (g1/g3)*(Wi + Qzi)])

						# Force equilibrium - Y direction
						Kyi = (Ci + (((Wi + Qzi)/g3) - Ui)*np.tan(phiRad))/(1 + (f3*np.tan(phiRad))/(g3*FSmy_i))
						FSmyNu += Kyi*(f1*RZ + f3*RX)
						FSmyDe += (Wi + Qzi)*RX + Ni_my*(g1*RZ - g3*RX)
						#print([Ayi, Ayi*(f1+(f3*g2)/g3), (g2/g3)*(Wi + Qzi)])
						'''
				'''
				print(iterationN)
				print(iterationN_FS)
				print(slidingDirection_i)
				'''

				# calculate 3D FS for each equilibrium
				if method in [1,2,4]:		
					FSx = FSxNu/(FSxDeAg1 - FSxDeQx + FSxDekxW - FSxDeDHx)
					FSy = FSyNu/(FSyDeAg2 - FSyDeQy + FSyDekyW - FSyDeDHy)

				if method in [1,2,3]:
					FSmx = FSmxNu/(FSmxDeWRY + FSmxDeKyWRZ + FSmxDeNig2RZ - FSmxDeNig3RY + FSmxDeQmx)
					FSmy = FSmyNu/(FSmyDeWRx + FSmyDeKxWRZ + FSmyDeNig2RZ - FSmyDeNig3RX + FSmyDeQmy)

				# compare computed and inital guess FS
				checkForceFS = 0
				checkMomentFS = 0
				
				# Janbu
				if method == 4:	
					if abs(FSx-FSx_i) <= tolFS and abs(FSy-FSy_i) <= tolFS:
						FSx_f = FSx
						FSy_f = FSy
						checkForceFS = 1
					elif abs(FSx-FSx_i) <= tolFS and abs(FSy-FSy_i) > tolFS:
						iterationN_FS += 1
						FSx_f = FSx
						FSy_i = FSy
						#FSy_i -= changeFS
					elif abs(FSx-FSx_i) > tolFS and abs(FSy-FSy_i) <= tolFS:
						iterationN_FS += 1
						#FSx_i -= changeFS
						FSx_i = FSx
						FSy_f = FSy
					else:
						iterationN_FS += 1
						#FSx_i -= changeFS
						#FSy_i -= changeFS
						FSx_i = FSx
						FSy_i = FSy

					if counts >= iterationNMax:
						print('too many iterations (iterationNN) - check code or increase maximum iteration number')
						# take failed result for method 3 or 4
						FS_results.append([4, iterationN_a, iterationN_FS, slidingDirection, 0, 0, 0, np.nan, np.nan, np.nan, np.nan, np.nan])						
						counts = 0
						iterationN_FS = 0
						iterationLambda = False
						iterationFS = False
						#FS_final = 0.5*(FS_force_f+FS_moment_f)
					else:
						if checkForceFS != 0:
							FS_results.append([4, iterationN_a, iterationN_FS, slidingDirection, 0, 0, 0, np.nan, np.nan, FSx_f, FSy_f, np.nan])
							counts = 0
							iterationN_FS = 0
							iterationLambda = False
							iterationFS = False
						
					counts += 1
					
					'''
					print(FSx_i)
					print(FSx)
					print(FSy_i)
					print(FSy)
					'''
				
				# Bishop
				elif method == 3:
					if abs(FSmx-FSmx_i) <= tolFS and abs(FSmy-FSmy_i) <= tolFS:
						FSmx_f = FSmx
						FSmy_f = FSmy
						checkMomentFS = 1
					elif abs(FSmx-FSmx_i) <= tolFS and abs(FSmy-FSmy_i) > tolFS:
						iterationN_FS += 1
						FSmx_f = FSmx
						FSmy_i = FSmy
						#FSmy_i -= changeFS
					elif abs(FSmx-FSmx_i) > tolFS and abs(FSmy-FSmy_i) <= tolFS:
						iterationN_FS += 1
						FSmx_i = FSmx
						FSmy_f = FSmy
						#FSmx_i -= changeFS
					else:
						iterationN_FS += 1
						FSmx_i = FSmx
						FSmy_i = FSmy
						#FSmx_i -= changeFS
						#FSmy_i -= changeFS

					if counts >= iterationNMax:
						print('too many iterations (iterationNN) - check code or increase maximum iteration number')
						# take failed result for method 3 or 4
						FS_results.append([3, iterationN_a, iterationN_FS, slidingDirection, 0, 0, 0, np.nan, np.nan, np.nan, np.nan, np.nan])						
						counts = 0
						iterationN_FS = 0
						iterationLambda = False
						iterationFS = False
						#FS_final = 0.5*(FS_force_f+FS_moment_f)
					else:
						if checkMomentFS != 0:
							FS_results.append([3, iterationN_a, iterationN_FS, slidingDirection, 0, 0, 0, FSmx_f, FSmy_f, np.nan, np.nan, np.nan])
							counts = 0
							iterationN_FS = 0
							iterationLambda = False
							iterationFS = False
					
					counts += 1
					#print(counts)

				# Spencer or Morgenstern-Price				
				elif method in [1,2]:
					if abs(FSx-FSx_i) <= tolFS and abs(FSy-FSy_i) <= tolFS:
						FSx_f = FSx
						FSy_f = FSy
						checkForceFS = 1
					elif abs(FSx-FSx_i) <= tolFS and abs(FSy-FSy_i) > tolFS:
						FSx_f = FSx
						FSy_i = FSy
						#FSy_i -= changeFS
					elif abs(FSx-FSx_i) > tolFS and abs(FSy-FSy_i) <= tolFS:
						FSx_i = FSx
						#FSx_i -= changeFS
						FSy_f = FSy
					else:
						FSx_i = FSx
						FSy_i = FSy
						#FSx_i -= changeFS
						#FSy_i -= changeFS

					if abs(FSmx-FSmx_i) <= tolFS and abs(FSmy-FSmy_i) <= tolFS:
						FSmx_f = FSmx
						FSmy_f = FSmy
						checkMomentFS = 1
					elif abs(FSmx-FSmx_i) <= tolFS and abs(FSmy-FSmy_i) > tolFS:				
						FSmx_f = FSmx
						FSmy_i = FSmy
						#FSmy_i -= changeFS
					elif abs(FSmx-FSmx_i) > tolFS and abs(FSmy-FSmy_i) <= tolFS:
						#FSmx_i -= changeFS
						FSmx_i = FSmx
						FSmy_f = FSmy
					else:
						FSmx_i = FSmx
						FSmy_i = FSmy
						#FSmx_i -= changeFS
						#FSmy_i -= changeFS

					if counts >= iterationNMax:
						print('too many iterations (iterationNN) - check code or increase maximum iteration number')
						# take failed result for method 3 or 4
						#FS_results.append([method, iterationN_a, iterationN_FS, slidingDirection, lambdaX, lambdaY, lambdaXY, np.nan, np.nan, np.nan, np.nan, np.nan])
						FS_results_Temp.append([np.nan, np.nan, np.nan, np.nan])
						if scaleLambdaXList[-1] > scaleLambdaYList[-1]:
							scaleLambdaYList.append(scaleLambdaYList[-1]+changeLambda)
						else:
							scaleLambdaXList.append(scaleLambdaXList[-1]+changeLambda)
						#scaleLambdaXList, scaleLambdaYList = changeLambda3D(scaleLambdaXList, scaleLambdaYList, FS_results_Temp, tolFS)
						iterationN_FS = 0
						counts = 0
						print(FSmx_f, FSmy_f, FSx_f, FSy_f)
						print(scaleLambdaXList[-1])
						print(scaleLambdaYList[-1])
						#iterationFS = False
						#iterationLambda = False
						continue

					else:
						if checkForceFS != 0 and checkMomentFS != 0:
							
							iterationN_FS = 0
							counts = 0
							iterationFS = False

							'''
							if round(slidingDirection,2) in [90, 270]:
								tempFSforceList.append(FSx_f)
								tempFSmomentList.append(FSmy_f)

							elif round(slidingDirection,2) in [0, 180, 360]:
								tempFSforceList.append(FSy_f)
								tempFSmomentList.append(FSmx_f)
							
							elif round(FSx_f, 2) == 0 and round(FSy_f, 2) != 0:
								if round(FSmx_f, 2) == 0 and round(FSmy_f, 2) != 0:
									tempFSforceList.append(FSy_f)
									tempFSmomentList.append(FSmy_f)
								elif round(FSmx_f, 2) != 0 and round(FSmy_f, 2) == 0:
									tempFSforceList.append(FSy_f)
									tempFSmomentList.append(FSmx_f)

							elif round(FSx_f, 2) != 0 and round(FSy_f, 2) == 0:
								if round(FSmx_f, 2) == 0 and round(FSmy_f, 2) != 0:
									tempFSforceList.append(FSx_f)
									tempFSmomentList.append(FSmy_f)
								elif round(FSmx_f, 2) != 0 and round(FSmy_f, 2) == 0:
									tempFSforceList.append(FSx_f)
									tempFSmomentList.append(FSmx_f)
							else:
								tempFSforceList.append(0.5*(FSx_f+FSy_f))
								tempFSmomentList.append(0.5*(FSmx_f+FSmy_f))
							'''

							#FS_results_Temp.append([method, iterationN_a, iterationN_FS, slidingDirection, lambdaX, lambdaY, lambdaXY, FSmx_f, FSmy_f, FSx_f, FSy_f, np.nan])
							print([method, iterationN_a, iterationN_FS, slidingDirection, lambdaX, lambdaY, lambdaXY, FSmx_f, FSmy_f, FSx_f, FSy_f, np.nan])
							FS_results_Temp.append([FSmx_f, FSmy_f, FSx_f, FSy_f])
							#print(FS_results_Temp)

							for loopCol in range(len(analysisInput)):
								f1 = fVectorDatas[loopCol][0]
								f2 = fVectorDatas[loopCol][1]
								f3 = fVectorDatas[loopCol][2]

								g1 = gVectorDatas[loopCol][0]
								g2 = gVectorDatas[loopCol][1]
								g3 = gVectorDatas[loopCol][2]

								dEx = dExDatas[loopCol]
								dEy = dEyDatas[loopCol]

								Qxi = QDatas[loopCol][0]
								Qyi = QDatas[loopCol][1]
								Qzi = QDatas[loopCol][2]

								baseArea = analysisInput[loopCol][14]/analysisInput[loopCol][16]
								Ui = baseArea*analysisInput[loopCol][5]	
								Wi = analysisInput[loopCol][14]
								Ci = baseArea*analysisInput[loopCol][12]
								phiRad = np.radians(analysisInput[loopCol][13])

								Ai = (lambdaX*dEx + lambdaY*dEy + Wi + Qzi)/g3
								Bi = -f3/g3

								Si_xx = (Ci + (Ai - Ui)*np.tan(phiRad))/(FSx_f-(Bi*np.tan(phiRad)))
								Si_yy = (Ci + (Ai - Ui)*np.tan(phiRad))/(FSy_f-(Bi*np.tan(phiRad)))

								Ni_xx = Ai + Bi*Si_xx
								Ni_yy = Ai + Bi*Si_yy

								dEx_new = (Si_xx*f1 - Ni_xx*g1 - seismicKx*Wi + Qxi)/(1-lambdaXY*Fxy)
								dEy_new = (Si_yy*f2 - Ni_yy*g2 - seismicKy*Wi + Qyi)/(1-lambdaXY*Fxy)

								#print(dEx_new, dEy_new)

								dExDatas[loopCol] = dEx_new
								dEyDatas[loopCol] = dEy_new

							if scaleLambdaXList[-1] > scaleLambdaYList[-1]:
								scaleLambdaYList.append(scaleLambdaYList[-1]+changeLambda)
							elif scaleLambdaXList[-1] < scaleLambdaYList[-1]:
								scaleLambdaXList.append(scaleLambdaXList[-1]+changeLambda)
							elif round(abs(scaleLambdaXList[-1] - scaleLambdaYList[-1]),2) < 0.01:
								randomNum = np.random.choice(2)
								if (randomNum%2) == 0: # even number
									scaleLambdaXList.append(scaleLambdaXList[-1]+changeLambda)
								else:  # odd number
									scaleLambdaYList.append(scaleLambdaYList[-1]+changeLambda)

							#if abs(tempFSforceList[-1]-tempFSmomentList[-1]) > tolFS:
								#scaleLambdaXList, scaleLambdaYList = changeLambda3D(scaleLambdaXList, scaleLambdaYList, FS_results_Temp, tolFS)
								
							FS_results.append([method, iterationN_a, iterationN_FS, slidingDirection, lambdaX, lambdaY, lambdaXY, FSmx_f, FSmy_f, FSx_f, FSy_f, np.nan])
							
							if round(slidingDirection,2) in [0, 180, 360]:
								checkFSsM = 0 
								checkFSsF = 0 
								checkFSsFM = abs(FSmx_f - FSy_f)
							elif round(slidingDirection,2) in [90, 270]:
								checkFSsM = 0 
								checkFSsF = 0 
								checkFSsFM = abs(FSmy_f - FSx_f)
							elif round(FSmx_f, 2) == 0 or round(FSmy_f, 2) == 0 or round(FSx_f, 2) == 0 or round(FSy_f, 2) == 0:
								checkFSsM = 0 
								checkFSsF = 0 
								checkFSsFM = abs(0.5*(FSmx_f+FSmy_f) - 0.5*(FSx_f+FSy_f))
							else:
								checkFSsM = abs(FSmx_f - FSmy_f)
								checkFSsF = abs(FSx_f - FSy_f)
								checkFSsFM = abs(0.5*(FSmx_f+FSmy_f) - 0.5*(FSx_f+FSy_f))
							
							#print('checkFSsM=%f'%checkFSsM)
							#print('checkFSsF=%f'%checkFSsF)
							#print('checkFSsFM=%f'%checkFSsFM)

							if (checkFSsM < tolFS) and (checkFSsF < tolFS) and (checkFSsFM < tolFS):
								iterationLambda = False

						else:
							iterationN_FS += 1

					counts += 1
					#print(counts)
				
				#iterationN_a
				#iterationN_FS

				#if iterationN_a >= 100: #(len(iterationLambdaX)*len(iterationLambdaY)):
				#	print('too many iterations (100) - check code or increase maximum iteration number')
				#	iterationFS = False

				#print(iterationN_a)
				#print(iterationN_FS)
				'''
				FSx_f
				FSy_f
				FSmy_f
				FSmx_f
				'''
	
		# Bishop Method
		if method == 3:

			#print(FS_results)
			#print(loopFS)

			FSmx_f = FS_results[loopFS][7]
			FSmy_f = FS_results[loopFS][8]

			# FSs_x - FSs_y
			checkFSsM = 0 
			if round(slidingDirection,2) in [0, 90, 180, 270, 360]:
				checkFSsM = 0 
			elif round(FSmx_f, 2) == 0 or round(FSmy_f, 2) == 0:
				checkFSsM = 0 
			else:		
				checkFSsM = abs(FSmx_f - FSmy_f)
			
			#print(checkFSsM)

			if checkFSsM < tolFS:
				# find global FS for a given direction
				'''
				if round(slidingDirection,2) in [0, 90, 180, 270, 360]:
					FS_results[loopFS][11] = max([FSmy_f, FSmx_f])
				'''
				if round(slidingDirection,2) in [90, 270]:
					FS_results[loopFS][11] = FSmy_f

				elif round(slidingDirection,2) in [0, 180, 360]:
					FS_results[loopFS][11] = FSmx_f		

				elif round(FSmx_f, 2) == 0 and round(FSmy_f, 2) != 0:
					FS_results[loopFS][11] = FSmy_f

				elif round(FSmx_f, 2) != 0 and round(FSmy_f, 2) == 0:
					FS_results[loopFS][11] = FSmx_f

				else:
					FS_results[loopFS][11] = (FSmx_f+FSmy_f)/2
				
				# find min global FS for a given direction
				if loopFS == 0 and not(np.isnan(FS_results[loopFS][11])):
						FSg_cur = FS_results[loopFS][11]
						FSg_IDX = loopFS
					
				else:
					if not np.isnan(FS_results[loopFS][11]) and np.isnan(FSg_cur):
						FSg_cur = FS_results[loopFS][11]
						FSg_IDX = loopFS
			
					elif not(np.isnan(FS_results[loopFS][11])) and not(np.isnan(FSg_cur)) and (FSg_cur > FS_results[loopFS][11]):
						FSg_cur = FS_results[loopFS][11]
						FSg_IDX = loopFS
			
			else:
				if loopFS == 0 and not(np.isnan(FS_results[loopFS][11])):
					FSg_cur = FS_results[loopFS][11]
					FSg_IDX = loopFS
			

			loopFS += 1
			iterationN_a += 1
			counts = 0
			continue

		# Janbu Method
		elif method == 4:
			
			FSx_f = FS_results[loopFS][9]
			FSy_f = FS_results[loopFS][10]

			# FSs_x - FSs_y
			checkFSsF = 0 
			if round(slidingDirection,2) in [0, 90, 180, 270, 360]:
				checkFSsF = 0 
			elif round(FSx_f, 2) == 0 or round(FSy_f, 2) == 0:
				checkFSsF = 0 
			else:
				checkFSsF = abs(FSx_f - FSy_f)

			#print(checkFSsF)

			if checkFSsF < tolFS:
				# find global FS for a given direction
				'''
				if round(slidingDirection,2) in [0, 90, 180, 270, 360]:
					FS_results[loopFS][11] = max([FSy_f, FSx_f])
				'''
				
				if round(slidingDirection,2) in [90, 270]:
					FS_results[loopFS][11] = FSx_f*correctionFactor

				elif round(slidingDirection,2) in [0, 180, 360]:
					FS_results[loopFS][11] = FSy_f*correctionFactor
				
				elif round(FSx_f, 2) == 0 and round(FSy_f, 2) != 0:
					FS_results[loopFS][11] = FSy_f*correctionFactor

				elif round(FSx_f, 2) != 0 and round(FSy_f, 2) == 0:
					FS_results[loopFS][11] = FSx_f*correctionFactor

				else:
					FS_results[loopFS][11] = correctionFactor*(FSx_f+FSy_f)/2
				
				# find min global FS for a given direction
				if loopFS == 0 and not(np.isnan(FS_results[loopFS][11])):
						FSg_cur = FS_results[loopFS][11]
						FSg_IDX = loopFS
					
				else:
					if not np.isnan(FS_results[loopFS][11]) and np.isnan(FSg_cur):
						FSg_cur = FS_results[loopFS][11]
						FSg_IDX = loopFS
			
					elif not(np.isnan(FS_results[loopFS][11])) and not(np.isnan(FSg_cur)) and (FSg_cur > FS_results[loopFS][11]):
						FSg_cur = FS_results[loopFS][11]
						FSg_IDX = loopFS
			
			else:
				if loopFS == 0 and not(np.isnan(FS_results[loopFS][11])):
					FSg_cur = FS_results[loopFS][11]
					FSg_IDX = loopFS

			loopFS += 1
			iterationN_a += 1
			counts = 0
			continue
	
		elif method == 1 or method == 2:
			#print(FS_results)

			FSmx_f = FS_results[-1][7]
			FSmy_f = FS_results[-1][8]
			FSx_f = FS_results[-1][9]
			FSy_f = FS_results[-1][10]

			# FSs_x - FSs_y
			'''
			checkFSsM = 0 
			checkFSsF = 0 
			checkFSsFM = 0
			'''
			if round(slidingDirection,2) in [0, 180, 360]:
				checkFSsM = 0 
				checkFSsF = 0 
				checkFSsFM = abs(FSmx_f - FSy_f)
			elif round(slidingDirection,2) in [90, 270]:
				checkFSsM = 0 
				checkFSsF = 0 
				checkFSsFM = abs(FSmy_f - FSx_f)
			elif round(FSmx_f, 2) == 0 or round(FSmy_f, 2) == 0 or round(FSx_f, 2) == 0 or round(FSy_f, 2) == 0:
				checkFSsM = 0 
				checkFSsF = 0 
				checkFSsFM = abs(0.5*(FSmx_f+FSmy_f) - 0.5*(FSx_f+FSy_f))
			else:
				checkFSsM = abs(FSmx_f - FSmy_f)
				checkFSsF = abs(FSx_f - FSy_f)
				checkFSsFM = abs(0.5*(FSmx_f+FSmy_f) - 0.5*(FSx_f+FSy_f))

			#print('checkFSsM=%f'%checkFSsM)
			#print('checkFSsF=%f'%checkFSsF)
			print('checkFSsFM=%f'%checkFSsFM)

			if (checkFSsM < tolFS) and (checkFSsF < tolFS) and (checkFSsFM < tolFS):
				'''
				if round(slidingDirection,2) in [0, 90, 180, 270, 360]:
					FS_results[loopFS][11] = max([(FSmx_f+FSx_f)/2, (FSmy_f+FSy_f)/2])
				'''
				
				if round(slidingDirection,2) in [90, 270]:
					FS_results[-1][11] = (FSmy_f+FSx_f)/2

				elif round(slidingDirection,2) in [0, 180, 360]:
					FS_results[-1][11] = (FSmx_f+FSy_f)/2
				
				elif round(FSx_f, 2) == 0 and round(FSy_f, 2) != 0:
					if round(FSmx_f, 2) == 0 and round(FSmy_f, 2) != 0:
						FS_results[-1][11] = (FSy_f+FSmy_f)/2

					elif round(FSmx_f, 2) != 0 and round(FSmy_f, 2) == 0:
						FS_results[-1][11] = (FSy_f+FSmx_f)/2

				elif round(FSx_f, 2) != 0 and round(FSy_f, 2) == 0:
					if round(FSmx_f, 2) == 0 and round(FSmy_f, 2) != 0:
						FS_results[-1][11] = (FSx_f+FSmy_f)/2

					elif round(FSmx_f, 2) != 0 and round(FSmy_f, 2) == 0:
						FS_results[-1][11] = (FSx_f+FSmx_f)/2

				else:
					FS_results[-1][11] = (FSmx_f+FSx_f+FSmy_f+FSy_f)/4

				# find min global FS for a given direction
				'''
				if loopFS == 0 and not(np.isnan(FS_results[loopFS][11])):
					FSg_cur = FS_results[-1][11]
					FSg_IDX = loopFS
					
				else:
					if not np.isnan(FS_results[loopFS][11]) and np.isnan(FSg_cur):
						FSg_cur = FS_results[loopFS][11]
						FSg_IDX = loopFS
			
					elif not(np.isnan(FS_results[loopFS][11])) and not(np.isnan(FSg_cur)) and (FSg_cur > FS_results[loopFS][11]):
						FSg_cur = FS_results[loopFS][11]
						FSg_IDX = loopFS
				'''
			'''
			else:
				if loopFS == 0 and not(np.isnan(FS_results[loopFS][11])):
					FSg_cur = FS_results[loopFS][11]
					FSg_IDX = loopFS
			'''

			loopFS += 1
			iterationN_a += 1
			counts = 0
			#continue				

	#print(FS_results)
	#print(FSg_IDX)
	#print( listAtColNum(FS_results, 11))
	#print(FS_results[int(FSg_IDX)])

	return FS_results

'''
3D LEM Slope Analysis:
1. Hungr Bishop
2. Hungr Janbu 
3. Hungr Janbu corrected
4. Cheng and Yip Bishop
5. Cheng and Yip Janbu simplified
6. Cheng and Yip Janbu corrected
7. Cheng and Yip Spencer
'''
def select3DMethod(fileName, method, seismicK, materialClass, materialNameList,  centerPt0, useDirectionB=None, useriterationNMax=200, usertolFS=0.001, userSideR=True):
	if method == 1:
		FS = analysis3DHungrBishop1989(fileName, seismicK, centerPt0, materialClass, materialNameList, iterationNMax=useriterationNMax, tolFS=usertolFS, occuranceFactor=0.5, tolDirection_user=None, spacingDirection=0.5, avDipDirectionB_user=useDirectionB)
	elif method == 2:
		FS = analysis3DHungrJanbu1989(fileName, seismicK, materialClass, materialNameList, correctFS=1, iterationNMax=useriterationNMax, tolFS=usertolFS, occuranceFactor=0.5, tolDirection_user=None, spacingDirection=0.5, avDipDirectionB_user=useDirectionB, sideResistance=userSideR)
	elif method == 3:
		FS = analysis3DHungrJanbu1989(fileName, seismicK, materialClass, materialNameList, correctFS=None, iterationNMax=useriterationNMax, tolFS=usertolFS, occuranceFactor=0.5, tolDirection_user=None, spacingDirection=0.5, avDipDirectionB_user=useDirectionB, sideResistance=True)
	elif method == 4:
		FS = analysis3DChengnYip2007(fileName, seismicK, centerPt0, 3, materialClass, lambdaIteration=None, iterationNMax=useriterationNMax, tolFS=usertolFS, correctFS=None, occuranceFactor=0.5, tolDirection_user=None, spacingDirection=0.5, avDipDirectionB_user=useDirectionB)
	elif method == 5:
		FS = analysis3DChengnYip2007(fileName, seismicK, centerPt0, 4, materialClass, lambdaIteration=None, iterationNMax=useriterationNMax, tolFS=usertolFS, correctFS=1, occuranceFactor=0.5, tolDirection_user=None, spacingDirection=0.5, avDipDirectionB_user=useDirectionB)
	elif method == 6:
		FS = analysis3DChengnYip2007(fileName, seismicK, centerPt0, 4, materialClass, lambdaIteration=None, iterationNMax=useriterationNMax, tolFS=usertolFS, correctFS=None, occuranceFactor=0.5, tolDirection_user=None, spacingDirection=0.5, avDipDirectionB_user=useDirectionB)
	elif method == 7:
		FS = analysis3DChengnYip2007(fileName, seismicK, centerPt0, 2, materialClass, lambdaIteration=None, iterationNMax=useriterationNMax, tolFS=usertolFS, correctFS=None, occuranceFactor=0.5, tolDirection_user=None, spacingDirection=0.5, avDipDirectionB_user=useDirectionB)
	else:
		return 'Invalid Input Method'
	
	return FS[-1][-1]

#testing
#select3DMethod(fileName, method, seismicK, materialClass, centerPt0, iterationNMax=200, avDipDirectionB_user=None)
import time
time_start = time.clock()

#fileName = 'ColumnResults_circular_Bishop.csv'				# 180
#fileName = 'ColumnResults_planar.csv'						# 180
#fileName = 'ColumnResults_plane_270.csv'					# 270
#fileName = 'yeager_noGW_noGeogrid.csv'						# 170.5
#fileName = 'ChengNYip2007_example2_ColumnResults.csv'		# 225
#fileName = 'ChengNYip2007_example3_ColumnResults.csv'		# 225
#fileName = 'ChengNYip2007_example4_ColumnResults.csv'		# 153.4
#fileName = '3DanalysisInputFile_ss2.csv'						# 180
#fileName = '3DanalysisInputFile_ssa1.csv'						# 180
#fileName = '3DanalysisInputFile_ssb1.csv'						# 180
#fileName = '3DanalysisInputFile.csv'						# 180


materialClass = {'rr':[1,[[45,10000],[0,0,0],0.25],150,150,1],'w1':[1,[[10,0],[0,0,0],0.25],120,120,2],'m1':[1,[[20,600],[0,0,0],0.25],120,120,3]}
materialNameList = ['rr', 'w1', 'm1']

seismicK = [0, 0]

'''
3D LEM Slope Analysis:
1. Hungr Bishop simplified
2. Hungr Janbu simplified
3. Hungr Janbu corrected
4. Cheng and Yip Bishop
5. Cheng and Yip Janbu simplified
6. Cheng and Yip Janbu corrected
7. Cheng and Yip Spencer
'''
#method = 4

#centerPt0 = [5, 0, 5] 
#centerPt0 = [5, 0, 8.03973] 
#centerPt0 = [0, 5, 8.03973] 
#centerPt0 = [1795820, 498800, 1008.41]
#centerPt0 = [0, 0, 5]
#centerPt0 = [0, 0, 9.46981]
#centerPt0 = [4, 0, 6.53953]
centerPt0 = [80, 60, 90]  # [80, 50, 120] #

#select3DMethod(fileName, method, seismicK, materialClass, materialNameList,  centerPt0, useDirectionB=None, useriterationNMax=200, usertolFS=0.001, userSideR=True)
#print (select3DMethod(fileName, 1, seismicK, materialClass, centerPt0, useriterationNMax=1000, useDirectionB=180))
#print (select3DMethod(fileName, 2, seismicK, materialClass, centerPt0, useriterationNMax=1000, useDirectionB=180))
#print (select3DMethod(fileName, 3, seismicK, materialClass, centerPt0, useriterationNMax=1000, useDirectionB=180))
#print (select3DMethod(fileName, 4, seismicK, materialClass, centerPt0, useriterationNMax=1000, useDirectionB=180))
#print (select3DMethod(fileName, 5, seismicK, materialClass, centerPt0, useriterationNMax=1000, useDirectionB=180))
#print (select3DMethod(fileName, 6, seismicK, materialClass, centerPt0, useriterationNMax=1000, useDirectionB=180))
#print (select3DMethod(fileName, 7, seismicK, materialClass, centerPt0, useriterationNMax=1000, useDirectionB=180,  usertolFS=0.005))

print (select3DMethod('3DanalysisInputFile_ss2.csv', 2, seismicK, materialClass, materialNameList, centerPt0, useriterationNMax=1000, useDirectionB=180))
print (select3DMethod('3DanalysisInputFile_ss2.csv', 3, seismicK, materialClass, materialNameList, centerPt0, useriterationNMax=1000, useDirectionB=180))

#print (select3DMethod('3DanalysisInputFile_ssb1_40_1_std85.csv', 1, seismicK, materialClass, centerPt0, useriterationNMax=1000, useDirectionB=180))
#print (select3DMethod('3DanalysisInputFile_ssc1_40_1_std85.csv', 1, seismicK, materialClass, centerPt0, useriterationNMax=1000, useDirectionB=180))
'''
print (select3DMethod('3DanalysisInputFile_ssb3_std75.csv', 1, seismicK, materialClass, centerPt0, useriterationNMax=1000, useDirectionB=180))
print (select3DMethod('3DanalysisInputFile_ssc3_std75.csv', 1, seismicK, materialClass, centerPt0, useriterationNMax=1000, useDirectionB=180))
print (select3DMethod('3DanalysisInputFile_ssb4_std100.csv', 1, seismicK, materialClass, centerPt0, useriterationNMax=1000, useDirectionB=180))
print (select3DMethod('3DanalysisInputFile_ssc4_std100.csv', 1, seismicK, materialClass, centerPt0, useriterationNMax=1000, useDirectionB=180))
print (select3DMethod('3DanalysisInputFile_ssb5_std200.csv', 1, seismicK, materialClass, centerPt0, useriterationNMax=1000, useDirectionB=180))
print (select3DMethod('3DanalysisInputFile_ssc5_std200.csv', 1, seismicK, materialClass, centerPt0, useriterationNMax=1000, useDirectionB=180))
'''
time_elapsed = (time.clock() - time_start)
print(time_elapsed)  # tells us the computation time in seconds