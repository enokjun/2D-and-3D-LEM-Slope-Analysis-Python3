'''
Created on Wed 07/11/2018

@author: Enok C., Yuming, Anthony

'''

'''make_list_with_floats.py'''

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


'''
purpose: create slices with information at sides
Input: DEM surface, slice parameters, slip surfaces
Output: slices points

Interpolation methods used:
1. scipy interpolation interp1d 
	> linear 			- a1

2. kriging ordinary 	
	> linear			- b1
	> power				- b2
	> gaussian			- b3
	> spherical			- b4
	> exponentail		- b5
	> hole-effect		- b6
3. kriging universal
	> linear			- c1
	> power				- c2
	> gaussian			- c3
	> spherical			- c4
	> exponentail		- c5
	> hole-effect		- c6
'''
# interpolation method
def interpolation2D(interpolType, edgeXCoords, DEMname, stdMax, exportOption=0):
	import numpy as np

	tempArrayX, tempArrayY = np.array(csv2list(DEMname)).T
	csvXY = []

	''' interpolation method '''
	#library import
	if interpolType[0] == 'a':
		from scipy.interpolate import interp1d
	elif interpolType[0] == 'b':
		from pykrige.ok import OrdinaryKriging
	elif interpolType[0] == 'c':
		from pykrige.uk import UniversalKriging

	# scipy interpol1d
	if interpolType == 'a1': 
		tempInterpolated = interp1d(tempArrayX, tempArrayY, bounds_error=False)
		interpolY = tempInterpolated(edgeXCoords)

	# pykrige ordinary kriging
	elif interpolType == 'b1':
		tempInterpolated = OrdinaryKriging(tempArrayX, np.zeros(tempArrayX.shape), tempArrayY, variogram_model='linear')
		interpolY, stdY = tempInterpolated.execute('grid', edgeXCoords, np.array([0.]))
		interpolY = interpolY[0]
		stdY = stdY[0]
	elif interpolType == 'b2':
		tempInterpolated = OrdinaryKriging(tempArrayX, np.zeros(tempArrayX.shape), tempArrayY, variogram_model='power')
		interpolY,stdY = tempInterpolated.execute('grid', edgeXCoords, np.array([0.]))
		interpolY = interpolY[0]
		stdY = stdY[0]
	elif interpolType == 'b3':
		tempInterpolated = OrdinaryKriging(tempArrayX, np.zeros(tempArrayX.shape), tempArrayY, variogram_model='gaussian')
		interpolY,stdY = tempInterpolated.execute('grid', edgeXCoords, np.array([0.]))
		interpolY = interpolY[0]
		stdY = stdY[0]
	elif interpolType == 'b4':
		tempInterpolated = OrdinaryKriging(tempArrayX, np.zeros(tempArrayX.shape), tempArrayY, variogram_model='spherical')
		interpolY,stdY = tempInterpolated.execute('grid', edgeXCoords, np.array([0.]))
		interpolY = interpolY[0]
		stdY = stdY[0]
	elif interpolType == 'b5':
		tempInterpolated = OrdinaryKriging(tempArrayX, np.zeros(tempArrayX.shape), tempArrayY, variogram_model='exponential')
		interpolY,stdY = tempInterpolated.execute('grid', edgeXCoords, np.array([0.]))
		interpolY = interpolY[0]
		stdY = stdY[0]
	elif interpolType == 'b6':
		tempInterpolated = OrdinaryKriging(tempArrayX, np.zeros(tempArrayX.shape), tempArrayY, variogram_model='hole-effect')
		interpolY,stdY = tempInterpolated.execute('grid', edgeXCoords, np.array([0.]))
		interpolY = interpolY[0]
		stdY = stdY[0]

	# pykrige universal kriging
	elif interpolType == 'c1':
		tempInterpolated = UniversalKriging(tempArrayX, np.zeros(tempArrayX.shape), tempArrayY, variogram_model='linear')
		interpolY,stdY = tempInterpolated.execute('grid', edgeXCoords, np.array([0.]))
		interpolY = interpolY[0]
		stdY = stdY[0]
	elif interpolType == 'c2':
		tempInterpolated = UniversalKriging(tempArrayX, np.zeros(tempArrayX.shape), tempArrayY, variogram_model='power')
		interpolY,stdY = tempInterpolated.execute('grid', edgeXCoords, np.array([0.]))
		interpolY = interpolY[0]
		stdY = stdY[0]
	elif interpolType == 'c3':
		tempInterpolated = UniversalKriging(tempArrayX, np.zeros(tempArrayX.shape), tempArrayY, variogram_model='gaussian')
		interpolY,stdY = tempInterpolated.execute('grid', edgeXCoords, np.array([0.]))
		interpolY = interpolY[0]
		stdY = stdY[0]
	elif interpolType == 'c4':
		tempInterpolated = UniversalKriging(tempArrayX, np.zeros(tempArrayX.shape), tempArrayY, variogram_model='spherical')
		interpolY,stdY = tempInterpolated.execute('grid', edgeXCoords, np.array([0.]))
		interpolY = interpolY[0]
		stdY = stdY[0]
	elif interpolType == 'c5':
		tempInterpolated = UniversalKriging(tempArrayX, np.zeros(tempArrayX.shape), tempArrayY, variogram_model='exponential')
		interpolY,stdY = tempInterpolated.execute('grid', edgeXCoords, np.array([0.]))
		interpolY = interpolY[0]
		stdY = stdY[0]
	elif interpolType == 'c6':
		tempInterpolated = UniversalKriging(tempArrayX, np.zeros(tempArrayX.shape), tempArrayY, variogram_model='hole-effect')
		interpolY,stdY = tempInterpolated.execute('grid', edgeXCoords, np.array([0.]))
		interpolY = interpolY[0]
		stdY = stdY[0]
	
	# for pykrige, eliminate points that has a large standard deviation
	#print(interpolY)
	#print(stdY)
	if interpolType[0] in ['b','c']:
		for loopYPred in range(len(interpolY)):
			if stdY[loopYPred] > stdMax:
				interpolY[loopYPred] = np.nan			

	#print(interpolY)
	#print(stdY)
	interpolY = interpolY.tolist()
	
	for loopXY in range(len(interpolY)):
		csvXY.append([edgeXCoords[loopXY], interpolY[loopXY]])
	
	# export the interpolated data into csv file
	if exportOption==0:
		exportList2CSV('interpolated_'+DEMname, csvXY)

	return csvXY

# function that will calculate y-coordinates (from csv files) for given x-coordinates of slice edges
def DEM_2D_slices(DEMNameList, DEMTypeList, DEMInterpolTypeList, canvasRange, sliceNumberMax, stdMax=10):
	# import python libraries
	import numpy as np 
	#import scipy
	#import matplotlib.pyplot as plt

	# slice X coordinates
	edgeXCoords = np.linspace(canvasRange[0], canvasRange[1], sliceNumberMax+1)
	#print(edgeXCoords)
	
	'''Input file sorting'''
	Y_pred = []
	for loopFile in range(len(DEMNameList)):
		Y_pred.append(interpolation2D(DEMInterpolTypeList[loopFile], edgeXCoords, DEMNameList[loopFile], stdMax, exportOption=0))
				
	#print(Y_pred)
	edgesX = {}
	sliceN = {}
	for loopX in range(len(edgeXCoords)):
		tempPtY = [float(canvasRange[2])]
		tempPtM = ['bb']

		# input information to dictionary of edgesX
		for loopFile in range(len(DEMNameList)):
			interpolY_edge = Y_pred[loopFile][loopX][1]
			#print(interpolY_edge)
			if not(np.isnan(interpolY_edge)):
				if DEMTypeList[loopFile] == 'tt' and interpolY_edge > canvasRange[3]:
					tempPtY.append(float(canvasRange[3]))
					tempPtM.append('tt')
				elif DEMTypeList[loopFile] == 'tt' and interpolY_edge <= canvasRange[3]:
					tempPtY.append(interpolY_edge)
					tempPtM.append('tt')
				elif DEMTypeList[loopFile] == 'rr' and interpolY_edge > canvasRange[2]:
					tempPtY[0] = interpolY_edge
					tempPtM[0] = 'rr'
				else:
					tempPtY.append(interpolY_edge)
					tempPtM.append(DEMTypeList[loopFile])

		edgesX[edgeXCoords[loopX]] = [tempPtY, tempPtM]

		if loopX >= 1:
			sliceN[loopX] = [edgeXCoords[loopX-1], edgesX[edgeXCoords[loopX-1]], edgeXCoords[loopX], edgesX[edgeXCoords[loopX]]]

	#print(edgesX[36])
	#print(sliceN[20])

	return sliceN

''' DEM points of 2D slip surface '''
'''
SSTypeList = 1	-> user-defined surface
SSTypeList = 2	-> grid circular search
'''
# find base y-coordinates for a given slip surface
def SS_2D_slices(SSTypeList, inputPara, canvasRange, sliceNumberMax, sliceN, stdMax=10):
	# import python libraries
	import numpy as np 
	#import scipy
	#from scipy.interpolate import interp1d
	#from pykrige.ok import OrdinaryKriging
	#from pykrige.uk import UniversalKriging
	#import matplotlib.pyplot as plt

	newSliceN = {}
	ss_csv = []

	# slice X coordinates
	edgeXCoords = np.linspace(canvasRange[0], canvasRange[1], sliceNumberMax+1)
	#print(edgeXCoords)
	
	'''Input SS for each slice X coordiantes'''
	ss_Y_pred = {}

	if SSTypeList == 2:		# circular grid search
		# extract inputPara 
		pt0X = inputPara[0]
		pt0Y = inputPara[1]
		R = inputPara[2]

		for loopX in range(len(edgeXCoords)):
			if (R**2 - (edgeXCoords[loopX] - pt0X)**2) >= 0:
				#tempPtYSS1 = pt0Y + np.sqrt(R**2 - (edgeXCoords[loopX] - pt0X)**2)
				tempPtYSS = pt0Y - np.sqrt(R**2 - (edgeXCoords[loopX] - pt0X)**2)
			else:
				tempPtYSS = np.nan
		
			ss_Y_pred[loopX] = [edgeXCoords[loopX], tempPtYSS]

	elif SSTypeList == 1:		#  user-defined surface
		# extract inputPara 
		DEMNameList = inputPara[0]
		interpolType = inputPara[1]

		tempPtYSS = interpolation2D(interpolType, edgeXCoords, DEMNameList, stdMax, exportOption=0)
		for loopX in range(len(edgeXCoords)):
			ss_Y_pred[loopX] = tempPtYSS[loopX]

	#print(ss_Y_pred)
	firstLeft = np.nan
	lastRight = np.nan
	for loopSlice in range(1,sliceNumberMax+1):

		ssLY = ss_Y_pred[loopSlice-1][1]
		ssRY = ss_Y_pred[loopSlice][1]

		sliceLedgeY = sliceN[loopSlice]
		sliceLedgeY = sliceLedgeY[1][0]
		sliceRedgeY = sliceN[loopSlice]
		sliceRedgeY = sliceRedgeY[3][0]

		#print(ssLY, ssRY, max(sliceLedgeY), max(sliceRedgeY))
		#print(loopSlice)
		#print(np.isnan(ssLY) or np.isnan(ssRY) or ssLY >= max(sliceLedgeY) or ssRY >= max(sliceRedgeY) or ssLY < min(sliceLedgeY) or ssRY < min(sliceRedgeY))

		if np.isnan(ssLY) or np.isnan(ssRY) or ssLY >= max(sliceLedgeY) or ssRY >= max(sliceRedgeY) or ssLY < min(sliceLedgeY) or ssRY < min(sliceRedgeY):
			#if (np.isnan(ssLY) and not(np.isnan(ssRY))) or (not(np.isnan(ssLY)) and (np.isnan(ssRY)))
			
			if np.isnan(ssLY) or np.isnan(ssRY) or ssLY >= max(sliceLedgeY) or ssRY >= max(sliceRedgeY):
				if loopSlice == 1:
					ss_csv.append([ss_Y_pred[loopSlice-1][0], np.nan])
				ss_csv.append([ss_Y_pred[loopSlice][0], np.nan])

			elif ssLY < min(sliceLedgeY) or ssRY < min(sliceRedgeY):
				if loopSlice == 1:
					ss_csv.append([ss_Y_pred[loopSlice-1][0], min(sliceLedgeY)])
				ss_csv.append([ss_Y_pred[loopSlice][0], min(sliceRedgeY)])

		elif not(np.isnan(ssLY)) and not(np.isnan(ssRY)) and (min(sliceLedgeY) <= ssLY < max(sliceLedgeY)) and (min(sliceRedgeY) <= ssRY < max(sliceRedgeY)):
			if firstLeft == 0:
				firstLeft = loopSlice-1
			elif lastRight == 0 and firstLeft != 0:
				lastRight = loopSlice+1
			
			# add slip surface to slice edges - left
			nsliceLedgeY = [0]
			nsliceLedgeYtype = [0]

			sliceLedgeYtype = sliceN[loopSlice]
			sliceLedgeYtype = sliceLedgeYtype[1][1]

			tempYss = []
			tempYssType = []
			for loopLayer in range(len(sliceLedgeYtype)):
				if sliceLedgeYtype[loopLayer][0] in ['t','m','g']:
					nsliceLedgeY.append(sliceLedgeY[loopLayer])
					nsliceLedgeYtype.append(sliceLedgeYtype[loopLayer])

				elif sliceLedgeYtype[loopLayer][0] in ['r','w','b']:
					if loopLayer == 0:
						tempYss.append(ssLY) 
						tempYssType.append('ss') 

					if ssLY <= sliceLedgeY[loopLayer]:
						tempYss.append(sliceLedgeY[loopLayer]) 
						tempYssType.append(sliceLedgeYtype[loopLayer]) 
			
			#print(tempYss)
			#print(tempYssType)
			maxEdgeBottom = max(tempYss)
			maxEdgeBottomIDX = tempYss.index(maxEdgeBottom)
			maxEdgeBottomType = tempYssType[maxEdgeBottomIDX]
			
			nsliceLedgeY[0] = maxEdgeBottom
			nsliceLedgeYtype[0] = maxEdgeBottomType

			if loopSlice == 1:
				ss_csv.append([ss_Y_pred[loopSlice-1][0], maxEdgeBottom])

			# add slip surface to slice edges - right
			nsliceRedgeY = [0]
			nsliceRedgeYtype = [0]

			sliceRedgeYtype = sliceN[loopSlice]
			sliceRedgeYtype = sliceRedgeYtype[3][1]

			tempYss = []
			tempYssType = []
			for loopLayer in range(len(sliceRedgeYtype)):
				if sliceRedgeYtype[loopLayer][0] in ['t','m','g']:
					nsliceRedgeY.append(sliceRedgeY[loopLayer])
					nsliceRedgeYtype.append(sliceRedgeYtype[loopLayer])

				elif sliceRedgeYtype[loopLayer][0] in ['r','w','b']:
					if loopLayer == 0:
						tempYss.append(ssRY) 
						tempYssType.append('ss') 

					if ssLY <= sliceRedgeY[loopLayer]:
						tempYss.append(sliceRedgeY[loopLayer]) 
						tempYssType.append(sliceRedgeYtype[loopLayer]) 
			
			#print(tempYss)
			#print(tempYssType)
			maxEdgeBottom = max(tempYss)
			maxEdgeBottomIDX = tempYss.index(maxEdgeBottom)
			maxEdgeBottomType = tempYssType[maxEdgeBottomIDX]

			nsliceRedgeY[0] = maxEdgeBottom
			nsliceRedgeYtype[0] = maxEdgeBottomType

			ss_csv.append([ss_Y_pred[loopSlice][0], maxEdgeBottom])

			newSliceN[loopSlice] = [ss_Y_pred[loopSlice-1][0], [nsliceLedgeY, nsliceLedgeYtype], ss_Y_pred[loopSlice][0], [nsliceRedgeY, nsliceRedgeYtype]]
		
	if not np.isnan(firstLeft):

		sliceLedge = sliceN[firstLeft]
		sliceLedgeX = sliceLedge[0]
		sliceLedgeY = sliceLedge[1][0]
		sliceLedgeYtype = sliceLedge[1][1]
		sliceRedge = sliceN[firstLeft]
		sliceRedgeX = sliceRedge[2]
		sliceRedgeY = sliceRedge[3][0]
		sliceRedgeYtype = sliceRedge[3][1]

		# add slip surface to slice edges - left
		nsliceLedgeY = [0]
		nsliceLedgeYtype = [0]

		tt_sliceLedgeY_idx = sliceRedgeYtype.index('tt')

		if SSTypeList == 2:		# circular grid search
			# extract inputPara 
			pt0X = inputPara[0]
			pt0Y = inputPara[1]
			R = inputPara[2]

			tempPtXSS1 = pt0X + np.sqrt(R**2 - (sliceLedgeY[tt_sliceLedgeY_idx] - pt0Y)**2)
			tempPtXSS2 = pt0X - np.sqrt(R**2 - (sliceLedgeY[tt_sliceLedgeY_idx] - pt0Y)**2)
		
			if tempPtXSS1 > sliceLedgeX and tempPtXSS1 < sliceRedgeX:
				ss_csv.append([tempPtXSS1, sliceLedgeY[tt_sliceLedgeY_idx]])
				nsliceLedgeY[0] = sliceLedgeY[tt_sliceLedgeY_idx]
				nsliceLedgeYtype[0] = 'tt'

			elif tempPtXSS2 > sliceLedgeX and tempPtXSS2 < sliceRedgeX:
				ss_csv.append([tempPtXSS2, sliceLedgeY[tt_sliceLedgeY_idx]])
				nsliceLedgeY[0] = sliceLedgeY[tt_sliceLedgeY_idx]
				nsliceLedgeYtype[0] = 'tt'

		elif SSTypeList == 1:	# user defined
			DEMNameList = inputPara[0]
			interpolType = inputPara[1]

			tempPtYSS = interpolation2D(interpolType, (sliceLedgeX+sliceRedgeX)/2, DEMNameList, stdMax, exportOption=0)

			ss_csv.append([(sliceLedgeX+sliceRedgeX)/2, tempPtYSS])
			nsliceLedgeY[0] = tempPtYSS
			nsliceLedgeYtype[0] = 'tt'

		# add slip surface to slice edges - right
		edgeRinfo = newSliceN[firstLeft+1]
		edgeRinfo = edgeRinfo[1]

		newSliceN[firstLeft] = [sliceLedgeX, [nsliceLedgeY, nsliceLedgeYtype], sliceRedgeX, edgeRinfo]

	if not np.isnan(lastRight):

		sliceLedge = sliceN[lastRight]
		sliceLedgeX = sliceLedge[0]
		sliceLedgeY = sliceLedge[1][0]
		sliceLedgeYtype = sliceLedge[1][1]
		sliceRedge = sliceN[lastRight]
		sliceRedgeX = sliceRedge[2]
		sliceRedgeY = sliceRedge[3][0]
		sliceRedgeYtype = sliceRedge[3][1]

		# add slip surface to slice edges - left
		edgeLinfo = newSliceN[lastRight-1]
		edgeLinfo = edgeLinfo[3]

		# add slip surface to slice edges - right
		nsliceRedgeY = [0]
		nsliceRedgeYtype = [0]

		tt_sliceRedgeYtype_idx = sliceRedgeYtype.index('tt')

		if SSTypeList == 2:		# circular grid search
			# extract inputPara 
			pt0X = inputPara[0]
			pt0Y = inputPara[1]
			R = inputPara[2]

			tempPtXSS1 = pt0X + np.sqrt(R**2 - (sliceRedgeY[tt_sliceRedgeYtype_idx] - pt0Y)**2)
			tempPtXSS2 = pt0X - np.sqrt(R**2 - (sliceRedgeY[tt_sliceRedgeYtype_idx] - pt0Y)**2)
		
			if tempPtXSS1 > sliceLedgeX and tempPtXSS1 < sliceRedgeX:
				ss_csv.append([tempPtXSS1, sliceRedgeY[tt_sliceRedgeYtype_idx]])
				nsliceRedgeY[0] = sliceRedgeY[tt_sliceRedgeYtype_idx]
				nsliceRedgeYtype[0] = 'tt'

			elif tempPtXSS2 > sliceLedgeX and tempPtXSS2 < sliceRedgeX:
				ss_csv.append([tempPtXSS2, sliceRedgeY[tt_sliceRedgeYtype_idx]])
				nsliceRedgeY[0] = sliceRedgeY[tt_sliceRedgeYtype_idx]
				nsliceRedgeYtype[0] = 'tt'

		elif SSTypeList == 1:	# user defined
			DEMNameList = inputPara[0]
			interpolType = inputPara[1]

			tempPtYSS = interpolation2D(interpolType, (sliceLedgeX+sliceRedgeX)/2, DEMNameList, stdMax, exportOption=0)

			ss_csv.append([(sliceLedgeX+sliceRedgeX)/2, tempPtYSS])
			nsliceRedgeY[0] = tempPtYSS
			nsliceRedgeYtype[0] = 'tt'

		newSliceN[lastRight] = [sliceLedgeX, edgeLinfo, sliceRedgeX, [nsliceRedgeY, nsliceRedgeYtype]]

	#print(ss_csv)
	if len(newSliceN.keys()) == 0:
		return None
	else:
		exportList2CSV('interpolated_ss_type'+str(SSTypeList)+'.csv', ss_csv)
		return newSliceN, ss_csv

# find center of rotation and radius of user-defined slip surface
def findpt0nR_2D(ss_csv):
	import numpy as np

	# starting and ending index
	startID = 0
	endID = 0
	for loopID in range(len(ss_csv)):
		if np.isnan(ss_csv[loopID][1]):
			if startID != 0 and endID == 0 and not(np.isnan(ss_csv[loopID-1][1])):
				endID = loopID-1
				break
			else:
				continue
		else:
			if startID == 0:
				startID = loopID
			else:
				continue
	
	#print(ss_csv[startID])
	#print(ss_csv[startID+3])
	#print(ss_csv[endID])

	P1x = ss_csv[startID][0]
	P1y = ss_csv[startID][1]
	P2x = ss_csv[startID+3][0]
	P2y = ss_csv[startID+3][1]
	P3x = ss_csv[endID][0]
	P3y = ss_csv[endID][1]
	DistSq1 = P1x**2 + P1y**2
	DistSq2 = P2x**2 + P2y**2
	DistSq3 = P3x**2 + P3y**2

	M11 = np.array([[P1x, P1y, 1],[P2x, P2y, 1],[P3x, P3y, 1]])
	M12 = np.array([[DistSq1, P1y, 1],[DistSq2, P2y, 1],[DistSq3, P3y, 1]])
	M13 = np.array([[DistSq1, P1x, 1],[DistSq2, P2x, 1],[DistSq3, P3x, 1]])
	M14 = np.array([[DistSq1, P1x, P1y],[DistSq2, P2x, P2y],[DistSq3, P3x, P3y]])

	pt0X = 0.5*np.linalg.det(M12)/np.linalg.det(M11)
	pt0Y = -0.5*np.linalg.det(M13)/np.linalg.det(M11)

	R = np.sqrt(pt0X**2 + pt0Y**2 + (np.linalg.det(M14)/np.linalg.det(M11)))

	return [round(pt0X,3), round(pt0Y,3), round(R,3)]

'''calculate area using shoelace theorem'''
def area_np(x, y):   
	import numpy as np     
	x = np.asanyarray(x)
	y = np.asanyarray(y)
	n = len(x)
	shift_up = np.arange(-n+1, 1)
	shift_down = np.arange(-1, n-1)    
	return abs((x * (y.take(shift_up) - y.take(shift_down))).sum() / 2.0)

'''2D LEM Slope Analysis - compute R, f, x and e for moment arm for moment equilibrium '''
'''
Input - X,Y coordinate of each slices, central point of rotation 
Output - moment arm of R, f, x and e for corresponding Sm, N, W and k*W

## Input file column legend of inputSlice
0(A) - slice number
1(B) - central coordinate X
2(C) - central coordinate Y at top
3(D) - central coordinate Y at base
4(E) - slice horizontal width
5(F) - slice base angle from horizontal (alpha)

## output file column legend
0(A) - slice number
1(B) - radius (R) 
2(C) - perpendicular offset from center of rotation(f)
3(D) - horizontal distance from slice to the center of rotation (x)
4(E) - vertical distance from C.G. of slice to the center of rotation (e)
'''
def computeRfxe_2D(inputSlice, cenPt0):
	# import functions python libraries
	#import numpy as np
	import math
 
	# compute horizontal moment arm (x)
	xi = abs(inputSlice[1] - cenPt0[0])

	# compute vertical moment arm for horizontal seismic (e)
	ei = abs(cenPt0[1] - 0.5*(inputSlice[2]+inputSlice[3]))

	# compute moment arm of Ri and fi
	RY = abs(cenPt0[1] - inputSlice[3])
	RR = math.sqrt(xi**2 + RY**2)

	deltaRangleY = round(90 - (abs(inputSlice[5]) + abs(math.degrees(math.atan(RY/xi)))),2)

	if deltaRangleY == 0:
		Ri = RR
		fi = 0
	elif deltaRangleY != 0:
		Ri = RR*math.cos(math.radians(deltaRangleY))
		fi = RR*math.sin(math.radians(deltaRangleY))

	return [Ri, fi, xi, ei]

# GW level, pwp
def GW_2D(inputFile, baseZ, water_unitWeight):
	import math

	dictKeys = inputFile.keys()

	output = {}
	for loopSlice in dictKeys:

		# check presence of GW level
		checkGW_left = 0
		if 'gw' in inputFile[loopSlice][1][1]:
			gwLindex = inputFile[loopSlice][1][1]
			gwLindex = gwLindex.index('gw')
			gwyL = inputFile[loopSlice][1][0][gwLindex]
			xL = inputFile[loopSlice][0]
			ytL = inputFile[loopSlice][1][0][-1]
			ybL = inputFile[loopSlice][1][0][0]
			checkGW_left = 1

		checkGW_right = 0
		if 'gw' in inputFile[loopSlice][3][1]:
			gwRindex = inputFile[loopSlice][3][1]
			gwRindex = gwRindex.index('gw')
			gwyR = inputFile[loopSlice][3][0][gwLindex]
			xR = inputFile[loopSlice][2]
			ytR = inputFile[loopSlice][3][0][-1]
			ybR = inputFile[loopSlice][3][0][0]
			checkGW_right = 1
		
		if checkGW_left == 0 and checkGW_right == 0:
			continue
		elif checkGW_left != 0 and checkGW_right == 0: 
			gwyR = baseZ
			xR = inputFile[loopSlice][2]
			ytR = inputFile[loopSlice][3][0][-1]
			ybR = inputFile[loopSlice][3][0][0]
		elif checkGW_left == 0 and checkGW_right != 0: 
			gwyL = baseZ
			xL = inputFile[loopSlice][0]
			ytL = inputFile[loopSlice][1][0][-1]
			ybL = inputFile[loopSlice][1][0][0]

		# calculate angle of GW inclination
		GWangle = abs(math.atan((gwyR-gwyL)/(xR-xL)))

		# define l
		lb_l = gwyL - ybL
		lb_r = gwyR - ybR
		lt_l = max([gwyL - ytL, 0])
		lt_r = max([gwyR - ytR, 0])	
		# for top, take only positive hw

		# calculate hw for left, right, base, top
		hw_l = lb_l*(math.cos(GWangle))**2
		hw_r = lb_r*(math.cos(GWangle))**2
		hw_b = (hw_l+hw_r)/2
		hw_t = 0.5*(lt_l*((math.cos(GWangle))**2)+lt_r*((math.cos(GWangle))**2))

		# calculate net hw for side and base
		hw_b_net = hw_b - hw_t
		if hw_b_net < 0:
			hw_s_net = 0
		else:
			hw_s_net = abs(hw_l-hw_r)

		output[loopSlice] = [hw_l*water_unitWeight, hw_r*water_unitWeight, hw_t*water_unitWeight, hw_b*water_unitWeight, hw_s_net*water_unitWeight, hw_b_net*water_unitWeight]

	return output

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
	import making_list_with_floats as makelist
	import numpy as np

	# materialClass takes input 
	# class[name] = [modelType, inputPara]

	material = materialClass[materialName]
	modelType = material[0]
	inputPara = material[1][0]
	unsaturatedPara = material[1][1]

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
		
		userShearNormal = makelist.csv2list(inputPara[0])

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
		if matricSuction >= unsaturatedPara[1]:
			unsatShearStrength = min([matricSuction, unsaturatedPara[2]])*np.tan(np.radians(unsaturatedPara[0]))
			calcPhiC[1] += unsatShearStrength

	return calcPhiC

''' main function - 2D slope stability support forces calculations from class of support type '''
'''
0 = User Defined Support - [type, F(0), d1, d2, d]
1 = End Anchored - [type, T]
2 = GeoTextile - [type, A(%), phi, a, T, anchorage_setting]
3 = Grouted Tieback - [type, T, P, B]
4 = Grouted Tieback with Friction - [type, T, P, phi, a, D]
5 = Micro Pile - [type, T]
6 = Soil Nail - [type, T, P, B]
'''
'''
def support_analysis(supportClass, supportName, Ana2D3D=2, Li=None, Lo=None, spacing2D=1, eff_normal_stress=None):

	import math
	import making_list_with_floats as makelist

	# converting the input csv file into a list
	supportInput = supportClass[supportName]

	F_applied=[]

	# for 3D analysis, set S to 1
	if Ana2D3D==3:
		Spacing=1
	elif Ana2D3D==2:
		Spacing=spacing2D

	# End Anchored - 1 = End Anchored - [type, T]
	if supportInput[0] == 1:
		F=supportInput[1]/Spacing
		F_applied.append(F)
		
	# GeoTextile - #2 = GeoTextile - [type, A(%), phi, a, T, anchorage_setting]
	# For GeoTextile Anchorage, 0 = not applicable, 1 = None, 2 = slope face, 3 = embedded end, 4 = both ends
	elif supportInput[0] == 2:	
		# F=[F1:Pullout, F2:Tensile Failure, F3:Stripping]
		F=[2*Lo*supportInput[1]*(supportInput[3] + eff_normal_stress*math.tan(math.radians(supportInput[2])))/100, supportInput[4]*Lo/100, 2*Li*supportInput[1]*(supportInput[3] + eff_normal_stress*math.tan(math.radians(supportInput[2])))/100]
	
		if supportInput[5] == 1:
			F_applied.append(min(F))
		elif supportInput[5]== 2:
			F_applied.append(min(F[1],F[2]))
		elif supportInput[5] == 3:
			F_applied.append(min(F[2],F[3]))
		elif supportInput[5] == 4:
			F_applied.append(F[2])

	# Grouted Tieback - [type, T, P, B]
	elif supportInput[0] == 3:
		# F=[F1:Pullout, F2:Tensile Failure, F3:Stripping]
		F=[supportInput[3]*Lo/Spacing, supportInput[1]/Spacing, (supportInput[2]+supportInput[3]*Li)/Spacing]
		F_applied.append(min(F))

	# Grouted Tieback with Friction - [type, T, P, phi, a, D]
	elif supportInput[0] == 4:
		# F=[F1:Pullout, F2:Tensile Failure, F3:Stripping]
		strengthModel = supportInput[4] + eff_normal_stress*np.tan(np.radians(supportInput[3]))

		F=[math.pi*supportInput[5]*Lo*strengthModel/Spacing, supportInput[1]/Spacing, (supportInput[2] + math.pi*supportInput[5]*Li*strengthModel)/Spacing]
		F_applied.append(min(F))

	# Micro Pile - [type, T]
	elif supportInput[0] == 5:
		F=supportInput[1]/Spacing
		F_applied.append(F)

	# Soil Nail - [type, T, P, B]
	elif supportInput[0] == 6:
		F=[supportInput[3]*Lo/Spacing, supportInput[1]/Spacing, (supportInput[2]+supportInput[3]*Li)/Spacing]
		F_applied.append(min(F))

	# User Defined Support - [type, T, F(0), d1, d2] - 6,10,11,12,13
	elif supportInput[0] == 0:

		if Li < supportInput[d1]:
			slope1 = (supportInput[1]-supportInput[2])/supportInput[3]
			F = slope1*Li + supportInput[2]
			F_applied.append(F)

		elif Li >= supportInput[3] and Li <= (supportInput[3]+supportInput[4]):
			F=supportInput[1]
			F_applied.append(F)
		
		elif Li > (supportInput[3]+supportInput[4]):
			slope2=(0-supportInput[1])/((Lo+Li)-supportInput[3]-supportInput[4])
			F=slope2*(Li-supportInput[3]-supportInput[4])+supportInput[1]
			F_applied.append(F)

	return F_applied[0]
'''

# find geometric data (area, base, width) for each slice
def createInputfile4Analysis_2D_slices(newSliceN, L2R_or_R2L, seismicK, pt0, sliceNumberMax, materialClass, canvasRange, water_unitWeight, tensionCrackAngle=None):
	# import python libraries
	import numpy as np 
	#import scipy
	#import matplotlib.pyplot as plt
	
	analysisInputFile = [[2,0,1]]
	analysisInputFile.append([2, 0, 1, len(newSliceN), L2R_or_R2L, pt0[0], pt0[1], seismicK, 0, 0, 0])

	sliceKeyList = newSliceN.keys()

	# width, base length, inclination (base, top), side left length, side right length
	sliceInfo_bAlalphaBeta = {}
	sliceInfo_W = {}  # weight
	for sliceN in sliceKeyList:
		tempList = []

		# slice width (b) and base length (l)
		b = abs(newSliceN[sliceN][0] - newSliceN[sliceN][2])
		l = np.sqrt((newSliceN[sliceN][1][0][0] - newSliceN[sliceN][3][0][0])**2 + (newSliceN[sliceN][0] - newSliceN[sliceN][2])**2)
		tempList.append(b)
		tempList.append(l)

		# base angle and top angle (alpha and beta)
		alpha = abs((newSliceN[sliceN][1][0][0] - newSliceN[sliceN][3][0][0])/(newSliceN[sliceN][0] - newSliceN[sliceN][2]))
		beta = abs((newSliceN[sliceN][1][0][-1] - newSliceN[sliceN][3][0][-1])/(newSliceN[sliceN][0] - newSliceN[sliceN][2]))
		tempList.append(alpha)
		tempList.append(beta)

		# side left length, side right length
		sideL = abs((newSliceN[sliceN][1][0][0] - newSliceN[sliceN][1][0][-1]))
		sideR = abs((newSliceN[sliceN][3][0][0] - newSliceN[sliceN][3][0][-1]))
		tempList.append(sideL)
		tempList.append(sideR)

		# individual areas and their unit weights
		tempArea = []
		tempUnitWeight = []
		if sliceN == min(sliceKeyList):
			numDiffAreasL = 0
			numDiffAreasR = len(newSliceN[sliceN][3][1])-2

			for loopArea in range(numDiffAreasR):
				xList = [newSliceN[sliceN][0], newSliceN[sliceN][2], newSliceN[sliceN][2]]
				yList = [newSliceN[sliceN][1][0][0], newSliceN[sliceN][3][0][loopArea], newSliceN[sliceN][3][0][loopArea+1]]
				
				tempType=[newSliceN[sliceN][1][1][0], newSliceN[sliceN][3][1][loopArea], newSliceN[sliceN][3][1][loopArea+1]]
				tempArea.append(area_np(xList, yList))

				if 'gw' in tempType:
					for loopAtype in range(loopArea+1, len(newSliceN[sliceN][3][1])):
						if newSliceN[sliceN][3][1][loopAtype][0] in ['m']:
							tempUnitWeight.append(materialClass[newSliceN[sliceN][3][1][loopAtype]][3])
							break
						else:
							continue
				#elif tempType[0][0] in ['r','w']:
				#	tempAreaType.append(tempType[1])
				else:
					tempUnitWeight.append(materialClass[newSliceN[sliceN][3][1][loopAtype]][2])
		elif sliceN == max(sliceKeyList):
			numDiffAreasL = len(newSliceN[sliceN][1][1])-2
			numDiffAreasR = 0

			for loopArea in range(numDiffAreasL):
				xList = [newSliceN[sliceN][2], newSliceN[sliceN][2], newSliceN[sliceN][0]]
				yList = [newSliceN[sliceN][1][0][loopArea], newSliceN[sliceN][1][0][loopArea+1], newSliceN[sliceN][3][0][0]]
				
				tempType=[newSliceN[sliceN][1][1][loopArea], newSliceN[sliceN][1][1][loopArea+1], newSliceN[sliceN][3][1][0]]
				tempArea.append(area_np(xList, yList))

				if 'gw' in tempType:
					for loopAtype in range(loopArea+1, len(newSliceN[sliceN][1][1])):
						if newSliceN[sliceN][1][1][loopAtype][0] in ['m']:
							tempUnitWeight.append(materialClass[newSliceN[sliceN][1][1][loopAtype]][3])
							break
						else:
							continue
				#elif tempType[0][0] in ['r','w']:
				#	tempAreaType.append(tempType[1])
				else:
					tempUnitWeight.append(materialClass[newSliceN[sliceN][1][1][loopAtype]][2])
		else:
			numDiffAreasL = len(newSliceN[sliceN][1][1])-2
			numDiffAreasR = len(newSliceN[sliceN][3][1])-2

			for loopArea in range(min([numDiffAreasL, numDiffAreasR])):
				xList = [newSliceN[sliceN][0], newSliceN[sliceN][0], newSliceN[sliceN][2], newSliceN[sliceN][2]]
				yList = [newSliceN[sliceN][1][0][loopArea], newSliceN[sliceN][1][0][loopArea+1], newSliceN[sliceN][3][0][loopArea+1], newSliceN[sliceN][3][0][loopArea]]
				
				tempType=[newSliceN[sliceN][1][1][loopArea], newSliceN[sliceN][1][1][loopArea+1], newSliceN[sliceN][3][1][loopArea+1], newSliceN[sliceN][3][1][loopArea]]
				tempArea.append(area_np(xList, yList))

				if 'gw' in tempType:
					for loopAtype in range(loopArea+1, len(newSliceN[sliceN][1][1])):
						if newSliceN[sliceN][1][1][loopAtype][0] in ['m']:
							tempUnitWeight.append(materialClass[newSliceN[sliceN][1][1][loopAtype]][3])
						else:
							continue
				#elif tempType[0][0] in ['r','w']:
				#	tempAreaType.append(tempType[0])
				else:
					for loopAtype in range(loopArea+1, len(newSliceN[sliceN][1][1])):
						if newSliceN[sliceN][1][1][loopAtype][0] in ['m']:
							tempUnitWeight.append(materialClass[newSliceN[sliceN][1][1][loopAtype]][2])
						else:
							continue
					
		tempW = 0
		for loopW in range(len(tempArea)):
			tempW += tempArea[loopW]*tempUnitWeight[loopW]

		sliceInfo_bAlalphaBeta[sliceN] = tempList
		sliceInfo_W[sliceN] = tempW

	# R, f, x, e - moment arms
	# make input file
	'''
	0(A) - slice number
	1(B) - central coordinate X
	2(C) - central coordinate Y at top
	3(D) - central coordinate Y at base
	4(E) - slice horizontal width
	5(F) - slice base angle from horizontal (alpha)
	'''
	sliceInfo_Rfxe = {}
	for sliceN in sliceKeyList:
		inputSlice = [sliceN, 0.5*(newSliceN[sliceN][0]+newSliceN[sliceN][2]), 0.5*(newSliceN[sliceN][1][0][-1]+newSliceN[sliceN][3][0][-1]), 0.5*(newSliceN[sliceN][1][0][0]+newSliceN[sliceN][3][0][0]), sliceInfo_bAlalphaBeta[sliceN][0], sliceInfo_bAlalphaBeta[sliceN][2]]
		sliceInfo_Rfxe[sliceN] = computeRfxe_2D(inputSlice, pt0)

	# pore-water pressure 
	# output - [hw_l, hw_r, hw_t, hw_b, hw_s_net, hw_b_net]
	sliceInfo_GW = GW_2D(newSliceN, canvasRange[2], water_unitWeight)

	# external load - for future
	#sliceInfo_load = {}

	# support - for future
	#sliceInfo_supportT = {}

	# shear strength parameters
	sliceInfo_shear = {}
	for sliceN in sliceKeyList:

		# name of the material type
		# left side 
		if newSliceN[sliceN][1][1][0][0] in ['r','w']:
			materialNameL = newSliceN[sliceN][1][1][0]

		elif newSliceN[sliceN][1][1][0][0] in ['b','s'] and len(newSliceN[sliceN][1][1]) != 1:
			if newSliceN[sliceN][1][1][1] not in ['gw']:
				materialNameL = newSliceN[sliceN][1][1][1]
			else:
				materialNameL = newSliceN[sliceN][1][1][2]

		elif newSliceN[sliceN][1][1][0][0] in ['b','s'] and len(newSliceN[sliceN][1][1]) == 1:
			materialNameL = None

		# right side 
		if newSliceN[sliceN][3][1][0][0] in ['r','w']:
			materialNameR = newSliceN[sliceN][1][1][0]

		elif newSliceN[sliceN][3][1][0][0] in ['b','s'] and len(newSliceN[sliceN][3][1]) != 1:
			if newSliceN[sliceN][3][1][1] not in ['gw']:
				materialNameR = newSliceN[sliceN][3][1][1]
			else:
				materialNameR = newSliceN[sliceN][3][1][2]

		elif newSliceN[sliceN][3][1][0][0] in ['b','s'] and len(newSliceN[sliceN][3][1]) == 1:
			materialNameR = None

		if materialNameR != materialNameL and materialNameL == None:
			materialName = materialNameR
		elif materialNameR != materialNameL and materialNameR == None:
			materialName = materialNameL
		elif materialNameR == materialNameL and materialNameL != None and materialNameR != None:
			materialName = materialNameL
		else:
			materialName = newSliceN[sliceN+1][1][1][-2]

		# inputs
		if sliceInfo_GW[sliceN][-1] < 0: 
			matricSuction = abs(sliceInfo_GW[sliceN][-1])
		else:
			matricSuction = 0
		
		# width, base length, inclination (base, top), side left length, side right length
		eff_normal_stress = (sliceInfo_W[sliceN]/sliceInfo_bAlalphaBeta[sliceN][1]) - max(sliceInfo_GW[sliceN][-1], 0)

		Z = 0.5*(newSliceN[sliceN][1][0][0]+newSliceN[sliceN][3][0][0])

		if materialName in newSliceN[sliceN][1][1][1]:
			materialYLIDX = newSliceN[sliceN][1][1][1].index(materialName)
		else:
			materialYLIDX = -1 

		if materialName in newSliceN[sliceN][3][1][1]:
			materialYRIDX = newSliceN[sliceN][3][1][1].index(materialName)
		else:
			materialYRIDX = -1

		if materialYLIDX != None and materialYRIDX == None:
			Ztop = 0.5*(newSliceN[sliceN][1][0][materialYLIDX]+newSliceN[sliceN][3][0][0])
		elif materialYLIDX == None and materialYRIDX != None:
			Ztop = 0.5*(newSliceN[sliceN][1][0][0]+newSliceN[sliceN][3][0][materialYRIDX])
		elif materialYLIDX != None and materialYRIDX != None:
			Ztop = 0.5*(newSliceN[sliceN][1][0][materialYLIDX]+newSliceN[sliceN][3][0][materialYRIDX])

		sliceInfo_shear[sliceN] = shearModel2cphi(materialClass, materialName, Z, Ztop, eff_normal_stress, matricSuction)[0]

	# L and d factors for corrected Janbu
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

	analysisInputFile[1].append(L_factor)
	analysisInputFile[1].append(d_factor)

	# compile into a new csv file for 2D analysis
	newSliceID = 1
	for sliceN in sliceKeyList: 
		compileList = [newSliceID]

		compileList.append(sliceInfo_bAlalphaBeta[sliceN][0])	# width
		compileList.append(sliceInfo_bAlalphaBeta[sliceN][1])	# base length
		compileList.append(sliceInfo_bAlalphaBeta[sliceN][2])	# base angle incline
		compileList.append(sliceInfo_bAlalphaBeta[sliceN][3])	# top angle incline

		compileList.append(sliceInfo_Rfxe[sliceN][0])	# R
		compileList.append(sliceInfo_Rfxe[sliceN][1])	# f
		compileList.append(sliceInfo_Rfxe[sliceN][2])	# x
		compileList.append(sliceInfo_Rfxe[sliceN][3])	# e

		compileList.append(sliceInfo_W[sliceN])					# W
		compileList.append(sliceInfo_GW[sliceN][3]*sliceInfo_bAlalphaBeta[sliceN][0])				# U_t
		compileList.append(max([sliceInfo_GW[sliceN][4],0])*sliceInfo_bAlalphaBeta[sliceN][1])	# U_b
		compileList.append(sliceInfo_GW[sliceN][0]*sliceInfo_bAlalphaBeta[sliceN][4])				# U_l
		compileList.append(sliceInfo_GW[sliceN][1]*sliceInfo_bAlalphaBeta[sliceN][5])				# U_r

		compileList.append(0)	# L
		compileList.append(0)	# omega
		compileList.append(0)	# L-d

		compileList.append(0)	# T
		compileList.append(0)	# i

		if sliceInfo_shear[sliceN][0] == 0 and sliceInfo_shear[sliceN][1] != 0:
			analysisInputFile[1][10] = 2
			compileList.append(sliceInfo_shear[sliceN][1])	# Sm
			compileList.append(sliceInfo_shear[sliceN][0])	# phi'
			compileList.append(sliceInfo_shear[sliceN][1])	# c'

		elif sliceInfo_shear[sliceN][0] != 0:
			analysisInputFile[1][10] = 1
			compileList.append(0)	# Sm
			compileList.append(sliceInfo_shear[sliceN][0])	# phi'
			compileList.append(sliceInfo_shear[sliceN][1])	# c'

		if tensionCrackAngle==None:
			compileList.append(1)
		elif tensionCrackAngle!=None:
			if sliceInfo_bAlalphaBeta[sliceN][2] >= tensionCrackAngle:
				compileList.append(0)
			else:
				compileList.append(1)

		analysisInputFile.append(compileList)

	exportList2CSV('2DanalysisInputFile.csv', analysisInputFile)

	return analysisInputFile


# one main function to run 2D backend 
def overall_2D_backend(DEMNameList, DEMTypeList, DEMInterpolTypeList, materialClass, water_unitWeight, canvasRange, sliceNumberMax, SSTypeList, SSinputPara, L2R_or_R2L, seismicK):

	slicepoints = DEM_2D_slices(DEMNameList, DEMTypeList, DEMInterpolTypeList, canvasRange, sliceNumberMax)
	slicepointsv2, ss_csv = SS_2D_slices(SSTypeList, SSinputPara, canvasRange, sliceNumberMax, slicepoints)
	pt0 = findpt0nR_2D(ss_csv)
	#print(pt0)
	analysisInputFile = createInputfile4Analysis_2D_slices(slicepointsv2, L2R_or_R2L, seismicK, pt0, sliceNumberMax, materialClass, canvasRange, water_unitWeight, tensionCrackAngle=None)

return slicepointsv2, ss_csv, pt0, analysisInputFile


'''2D analysis'''
# Analysis_2D_Ordinary_V4_06_09_2018.py
def ordinary_method(filename):
	import math

	# converting the input csv file into a list
	analysisInput = csv2list(filename)

	# total number of slip surfaces
	totalSlipSurface = int(analysisInput[0][2])	

	# create an empty list for FS of all slip surfaces
	FS= []

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

		while difference>tol:

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
				u=slice[11]-slice[10]
				l=slice[1]/math.cos(math.radians(slice[3]))
				m_alpha=math.cos(math.radians(slice[3]))+(math.sin(math.radians(slice[3]))*math.tan(math.radians(slice[21])))/FSguess[surface_num]
				P=(slice[9]-(slice[20]*l*math.sin(math.radians(slice[3])))/FSguess[surface_num]+(u*l*math.tan(math.radians(slice[21]))*math.sin(math.radians(slice[3])))/FSguess[surface_num])/m_alpha
				numerator+=(slice[20]*l*slice[5]+(P-u)*slice[5]*math.tan(math.radians(slice[21]))) 
				sum_Wx+=slice[9]*slice[7]
				sum_Pf+=slice[6]*P
				sum_kWe+=analysisInfo[7]*slice[9]*slice[8]
				sum_Ld+=slice[14]*slice[16]
				sum_T+=slice[17]

			F=numerator/(sum_Ld+sum_Pf+sum_Wx+sum_Aa+sum_kWe)
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
		if statistics.mode(c_list) != 0 and statistics.mode(phi_list) !=0:
			b1=0.5
		elif statistics.mode(c_list) ==0 and statistics.mode(phi_list) !=0:
			b1=0.31
		elif statistics.mode(c_list) !=0 and statistics.mode(phi_list) ==0:
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
				u=slice[11]-slice[10]
				l=slice[1]/math.cos(math.radians(slice[3]))
				m_alpha=math.cos(math.radians(slice[3]))+(math.sin(math.radians(slice[3]))*math.tan(math.radians(slice[21])))/FSguess[surface_num]
				P=(slice[9]-(slice[20]*l*math.sin(math.radians(slice[3])))/FSguess[surface_num]+(u*l*math.tan(math.radians(slice[21]))*math.sin(math.radians(slice[3])))/FSguess[surface_num])/m_alpha
				numerator += slice[20]*l*math.cos(math.radians(slice[3]))+(P-u)*math.tan(math.radians(slice[21]))*math.cos(math.radians(slice[3]))
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
	for loop2 in range(decimalPoint):
		if absDiffFS <= tolaranceFS:
			valueChange = valueChangeList[-1]
			break
		elif absDiffFS <= dFSLimList[loop2] and absDiffFS > dFSLimList[loop2+1]:
			valueChange = valueChangeList[loop2]
			break
			
	thetaInter += valueChange*UorD

	return thetaInter

'''main function - Spencer Method 2D LEM slope stability analysis'''
def analysis2DSpencer(inputFileName, tolaranceFS=0.0005, iterationNMax=200):
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
				FS_f_de = analysisInfo[8] 	# A 

				# FS for moment calculated
				FS_moment = 0
				FS_m_nom = 0
				FS_m_de = analysisInfo[8]*analysisInfo[9] # A*a

				# iterate trough slice
				for loopSlice in range(len(sliceInfo)):			
					# net pore-water pressure
					u_net_base = sliceInfo[loopSlice][11] - sliceInfo[loopSlice][10]
					u_net_side = abs(sliceInfo[loopSlice][12] - sliceInfo[loopSlice][13])

					# interslice assumption for first analysis
					if iterationN == 1:
						dX_f = math.tan(math.radians(thetaInter))*u_net_side
						dX_m = math.tan(math.radians(thetaInter))*u_net_side	   # change in vertical interslice force (dX = X_L-X_R)
					# using FS from previous calculation dX is calculated
					else:
						dX_f = math.tan(math.radians(thetaInter))*dE_list[loopSlice][0]
						dX_m = math.tan(math.radians(thetaInter))*dE_list[loopSlice][1]

					# actual resisting base length = base length * tension crack coefficient
					b_len_r = sliceInfo[loopSlice][2]*sliceInfo[loopSlice][22]
					
					if analysisInfo[10] == 1:
						# calculate normal force (P) for force equilibrium
						ma_force = math.cos(math.radians(sliceInfo[loopSlice][3])) + math.sin(math.radians(sliceInfo[loopSlice][3]))*math.tan(math.radians(sliceInfo[loopSlice][21]))/FS_force_i
						P_force = (sliceInfo[loopSlice][9] + sliceInfo[loopSlice][10] - dX_f - (sliceInfo[loopSlice][20])*b_len_r*math.sin(math.radians(sliceInfo[loopSlice][3]))/FS_force_i + u_net_base*math.tan(math.radians(sliceInfo[loopSlice][21]))*math.sin(math.radians(sliceInfo[loopSlice][3]))/FS_force_i)/ma_force
						
						# calculate normal force (P) for moment equilibrium
						ma_moment = math.cos(math.radians(sliceInfo[loopSlice][3])) + math.sin(math.radians(sliceInfo[loopSlice][3]))*math.tan(math.radians(sliceInfo[loopSlice][21]))/FS_moment_i
						P_moment = (sliceInfo[loopSlice][9] + sliceInfo[loopSlice][10] - dX_m - (sliceInfo[loopSlice][20])*b_len_r*math.sin(math.radians(sliceInfo[loopSlice][3]))/FS_moment_i + u_net_base*math.tan(math.radians(sliceInfo[loopSlice][21]))*math.sin(math.radians(sliceInfo[loopSlice][3]))/FS_moment_i)/ma_moment
					
						# calculate shear strength
						shear_strength_f = (sliceInfo[loopSlice][21]*b_len_r + (P_force - u_net_base)*math.tan(math.radians(sliceInfo[loopSlice][21])))
						shear_strength_m = (sliceInfo[loopSlice][21]*b_len_r + (P_moment - u_net_base)*math.tan(math.radians(sliceInfo[loopSlice][21])))

					elif analysisInfo[10] != 1:
						# calculate normal force (P) for force equilibrium
						P_force = (sliceInfo[loopSlice][9] + sliceInfo[loopSlice][10] - dX_f - sliceInfo[loopSlice][19]*math.sin(math.radians(sliceInfo[loopSlice][3]))/FS_force_i)/math.cos(math.radians(sliceInfo[loopSlice][3]))
						
						# calculate normal force (P) for moment equilibrium
						P_moment = (sliceInfo[loopSlice][9] + sliceInfo[loopSlice][10] - dX_m - sliceInfo[loopSlice][19]*math.sin(math.radians(sliceInfo[loopSlice][3]))/FS_moment_i)/math.cos(math.radians(sliceInfo[loopSlice][3]))

						# calculate shear strength
						shear_strength_f = sliceInfo[loopSlice][19]
						shear_strength_m = sliceInfo[loopSlice][19]

					# calcualte FS for force 
					FS_f_nom += math.cos(math.radians(sliceInfo[loopSlice][3]))*shear_strength_f
					FS_f_de += P_force*math.sin(math.radians(sliceInfo[loopSlice][3])) + analysisInfo[7]*(sliceInfo[loopSlice][9]+sliceInfo[loopSlice][10]) - sliceInfo[loopSlice][14]*math.cos(math.radians(sliceInfo[loopSlice][15]))

					# calcualte FS for moment 
					FS_m_nom += sliceInfo[loopSlice][5]*shear_strength_m
					FS_m_de += (sliceInfo[loopSlice][9]+sliceInfo[loopSlice][10])*sliceInfo[loopSlice][7] - P_moment*sliceInfo[loopSlice][6] + analysisInfo[5]*sliceInfo[loopSlice][9]*sliceInfo[loopSlice][8] + sliceInfo[loopSlice][14]*sliceInfo[loopSlice][16]
					
					# calculate dE for next iteration
					# dE = change in horizontal interslice force (dE = E_L-E_R)
					dE_f = u_net_side + P_force*math.sin(math.radians(sliceInfo[loopSlice][3])) - (math.cos(math.radians(sliceInfo[loopSlice][3]))*shear_strength_f/FS_force_i) + analysisInfo[7]*sliceInfo[loopSlice][9] 
					dE_m = u_net_side + P_moment*math.sin(math.radians(sliceInfo[loopSlice][3])) - (math.cos(math.radians(sliceInfo[loopSlice][3]))*shear_strength_m/FS_moment_i) + analysisInfo[7]*sliceInfo[loopSlice][9] 

					if iterationN == 1:
						dE_list.append([dE_f, dE_m])
					else:
						dE_list[loopSlice] = [dE_f, dE_m]

				# calculated FS
				FS_force = FS_f_nom/FS_f_de
				FS_moment = FS_m_nom/FS_m_de
				
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
			
			if iterationN >= iterationNMax or iterationNN >= iterationNMax:
				print('too many iterations - check code or increase maximum iteration number')
				iteration1 = False
				FS_final = 'None'
			elif abs(FS_moment_f - FS_force_f) > tolaranceFS:
				iterationNN += 1
				thetaInter = changeIntersliceTheta_Spencer(thetaInter, FS_moment_f, FS_force_f, tolaranceFS)
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
def analysis2DMorgensternPrice(inputFileName, tolaranceFS, FxType, inputFx, iterationNMax):
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
		scaleLambda = 0.5
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
				FS_f_de = analysisInfo[8] 	# A 

				# FS for moment calculated
				FS_moment = 0
				FS_m_nom = 0
				FS_m_de = analysisInfo[8]*analysisInfo[9] # A*a

				# iterate trough slice
				for loopSlice in range(len(sliceInfo)):	
					# net pore-water pressure
					u_net_base = sliceInfo[loopSlice][11] - sliceInfo[loopSlice][10]
					u_net_side = abs(sliceInfo[loopSlice][12] - sliceInfo[loopSlice][13])
					
					# interslice assumption for first analysis
					if iterationN == 1:
						dX_f = scaleLambda*intersliceFxList[loopSlice][2]*u_net_side
						dX_m = scaleLambda*intersliceFxList[loopSlice][2]*u_net_side	   # change in vertical interslice force (dX = X_L-X_R)
					# using FS from previous calculation dX is calculated
					else:
						dX_f = scaleLambda*intersliceFxList[loopSlice][2]*dE_list[loopSlice][0]
						dX_m = scaleLambda*intersliceFxList[loopSlice][2]*dE_list[loopSlice][1]

					# actual resisting base length
					b_len_r = sliceInfo[loopSlice][2]*sliceInfo[loopSlice][22]

					if analysisInfo[10] == 1:
						# calculate normal force (P) for force equilibrium
						ma_force = math.cos(math.radians(sliceInfo[loopSlice][3])) + math.sin(math.radians(sliceInfo[loopSlice][3]))*math.tan(math.radians(sliceInfo[loopSlice][21]))/FS_force_i
						P_force = (sliceInfo[loopSlice][9] + sliceInfo[loopSlice][10] - dX_f - (sliceInfo[loopSlice][20])*b_len_r*math.sin(math.radians(sliceInfo[loopSlice][3]))/FS_force_i + u_net_base*math.tan(math.radians(sliceInfo[loopSlice][21]))*math.sin(math.radians(sliceInfo[loopSlice][3]))/FS_force_i)/ma_force
						
						# calculate normal force (P) for moment equilibrium
						ma_moment = math.cos(math.radians(sliceInfo[loopSlice][3])) + math.sin(math.radians(sliceInfo[loopSlice][3]))*math.tan(math.radians(sliceInfo[loopSlice][21]))/FS_moment_i
						P_moment = (sliceInfo[loopSlice][9] + sliceInfo[loopSlice][10] - dX_m - (sliceInfo[loopSlice][20])*b_len_r*math.sin(math.radians(sliceInfo[loopSlice][3]))/FS_moment_i + u_net_base*math.tan(math.radians(sliceInfo[loopSlice][21]))*math.sin(math.radians(sliceInfo[loopSlice][3]))/FS_moment_i)/ma_moment
					
						# calculate shear strength
						shear_strength_f = (sliceInfo[loopSlice][21]*b_len_r + (P_force - u_net_base)*math.tan(math.radians(sliceInfo[loopSlice][21])))
						shear_strength_m = (sliceInfo[loopSlice][21]*b_len_r + (P_moment - u_net_base)*math.tan(math.radians(sliceInfo[loopSlice][21])))

					elif analysisInfo[10] != 1:
						# calculate normal force (P) for force equilibrium
						P_force = (sliceInfo[loopSlice][9] + sliceInfo[loopSlice][10] - dX_f - sliceInfo[loopSlice][19]*math.sin(math.radians(sliceInfo[loopSlice][3]))/FS_force_i)/math.cos(math.radians(sliceInfo[loopSlice][3]))
						
						# calculate normal force (P) for moment equilibrium
						P_moment = (sliceInfo[loopSlice][9] + sliceInfo[loopSlice][10] - dX_m - sliceInfo[loopSlice][19]*math.sin(math.radians(sliceInfo[loopSlice][3]))/FS_moment_i)/math.cos(math.radians(sliceInfo[loopSlice][3]))

						# calculate shear strength
						shear_strength_f = sliceInfo[loopSlice][19]
						shear_strength_m = sliceInfo[loopSlice][19]

					# calcualte FS for force 
					FS_f_nom += math.cos(math.radians(sliceInfo[loopSlice][3]))*shear_strength_f
					FS_f_de += P_force*math.sin(math.radians(sliceInfo[loopSlice][3])) + analysisInfo[7]*(sliceInfo[loopSlice][9]+sliceInfo[loopSlice][10]) - sliceInfo[loopSlice][14]*math.cos(math.radians(sliceInfo[loopSlice][15]))

					# calcualte FS for moment 
					FS_m_nom += sliceInfo[loopSlice][5]*shear_strength_m
					FS_m_de += (sliceInfo[loopSlice][9]+sliceInfo[loopSlice][10])*sliceInfo[loopSlice][7] - P_moment*sliceInfo[loopSlice][6] + analysisInfo[5]*sliceInfo[loopSlice][9]*sliceInfo[loopSlice][8] + sliceInfo[loopSlice][14]*sliceInfo[loopSlice][16]
					
					# calculate dE for next iteration
					# dE = change in horizontal interslice force (dE = E_L-E_R)
					dE_f = u_net_side + P_force*math.sin(math.radians(sliceInfo[loopSlice][3])) - (math.cos(math.radians(sliceInfo[loopSlice][3]))*shear_strength_f/FS_force_i) + analysisInfo[7]*sliceInfo[loopSlice][9] 
					dE_m = u_net_side + P_moment*math.sin(math.radians(sliceInfo[loopSlice][3])) - (math.cos(math.radians(sliceInfo[loopSlice][3]))*shear_strength_m/FS_moment_i) + analysisInfo[7]*sliceInfo[loopSlice][9] 

					if iterationN == 1:
						dE_list.append([dE_f, dE_m])
					else:
						dE_list[loopSlice] = [dE_f, dE_m]

				# calculated FS
				FS_force = FS_f_nom/FS_f_de
				FS_moment = FS_m_nom/FS_m_de
				
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
			
			#print(scaleLambda)
			#print(FS_force_f)
			#print(FS_moment_f)
			#print(iterationNN)

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
				scaleLambda = changeIntersliceLambda_MP(scaleLambda, FS_moment_f, FS_force_f, tolaranceFS)
			
			else:
				FS_final = FS_force_f
				iteration1 = False
		
		results2DMP.append([analysisInfo[0:3], [FS_final, scaleLambda, FxType]])
		#print(iterationNN)

	return results2DMP

# for input method: ordinary=1, modified bishop=2, simplified Janbu=3, corected Janbu=4, spencer=5, morgenstern price=6
def selectMethod(inputFileName,method,tolaranceFS,iterationNMax,FxType,inputFx):
	if method == 1:
		return ordinary_method(inputFileName)
	elif method == 2:
		return modified_bishop(inputFileName,tolaranceFS,iterationNMax)
	elif method == 3:
		return janbu_simplified(inputFileName,tolaranceFS,iterationNMax,False)
	elif method == 4:
		return janbu_simplified(inputFileName,tolaranceFS,iterationNMax,True)
	elif method == 5:
		return analysis2DSpencer(inputFileName, tolaranceFS, iterationNMax)
	elif method == 6:
		return analysis2DMorgensternPrice(inputFileName, tolaranceFS, FxType, inputFx, iterationNMax)
	else:
		return 'Invalid Input Method'

# everything combined
def process2DDEM(DEMNameList, DEMTypeList, DEMInterpolTypeList, materialClass, water_unitWeight, canvasRange, sliceNumberMax, SSTypeList, SSinputPara, L2R_or_R2L, seismicK, method, inputFileName='2DanalysisInputFile.csv',tolaranceFS=0.001,iterationNMax=100,FxType=0,inputFx=0):

	slicepointsv2, ss_csv, pt0, analysisInputFile = overall_2D_backend(DEMNameList, DEMTypeList, DEMInterpolTypeList, materialClass, water_unitWeight, canvasRange, sliceNumberMax, SSTypeList, SSinputPara, L2R_or_R2L, seismicK)
	FS2D = selectMethod(inputFileName,method,tolaranceFS,iterationNMax,FxType,inputFx)

	return FS2D

'''
#naming convension of DEMTypeList
'gw' = groundwater - phreatic layer (uw = 0 at the layer)
'mx' = top surface boundary of material x (x is integer)
'wx' = weak layer x - does not apply material properties below the weak layer x (x is integer)
'rr' = rock layer - base of the slice goes along the surface of rock boundary
'tt' = slope face - above this layer there is no material
'''
## Input file column legend
'''
[modelType, inputPara, unit weight, saturated unit weight]
1 = Mohr-Coulomb - [phi', c']
2 = undrained - depth - [shear_max, shear_min, Su_top, diff_Su]
3 = undrained - datum - [shear_max, shear_min, Su_datum, diff_Su, z_datum]
4 = power curve - [P_atm, a, b]
5 = shear-normal user defined function - [fileName]
unsaturated - [phi_b, aev, us_max]

Interpolation methods used:
1. scipy interpolation interp1d 
	> linear 			- a1

2. kriging ordinary 	
	> linear			- b1
	> power				- b2
	> gaussian			- b3
	> spherical			- b4
	> exponentail		- b5
	> hole-effect		- b6
3. kriging universal
	> linear			- c1
	> power				- c2
	> gaussian			- c3
	> spherical			- c4
	> exponentail		- c5
	> hole-effect		- c6


'''
'''Output Check'''
import time
time_start = time.clock()

DEMNameList = ['gwLevel2D.csv','bedrockLayer2D.csv','weakLayer2D.csv','groundSurface2D.csv','groundSurface2D.csv']
DEMTypeList = ['gw','rr','w1','m1','tt'] 
materialClass = {'rr':[1,[[45,10000],[0,0,0]],150,150,1],'w1':[1,[[10,0],[0,0,0]],120,120,2],'m1':[1,[[20,600],[0,0,0]],120,120,3]}
water_unitWeight = 62.4

DEMInterpolTypeList = ['b1', 'b1', 'b1', 'b1', 'b1']
canvasRange = [0, 155, 0, 80] # min X, max X, min Y, max Y
sliceNumberMax = 100

SSTypeList = 2
SSinputPara = [60, 90, 80]
L2R_or_R2L = 1
seismicK = 0

# for input method: ordinary=1, modified bishop=2, simplified Janbu=3, corected Janbu=4, spencer=5, morgenstern price=6
method = 3

#selectMethod(inputFileName,method,tolaranceFS,iterationNMax,FxType,inputFx)
print(process2DDEM(DEMNameList, DEMTypeList, DEMInterpolTypeList, materialClass, water_unitWeight, canvasRange, sliceNumberMax, SSTypeList, SSinputPara, L2R_or_R2L, seismicK, method, inputFileName='2DanalysisInputFile.csv',tolaranceFS=0.001,iterationNMax=100,FxType=0,inputFx=0))

time_elapsed = (time.clock() - time_start)
print(time_elapsed)  # tells us the computation time in seconds