'''3D back end'''

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
	#with open(fileName, 'w', encoding='UTF-8', newline='') as f:
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
1. scipy interpolation interp2d 
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
def interpolation3D(interpolType, edgeXCoords, edgeYCoords, DEMname, stdMax, exportOption=0):
	import numpy as np
	#print(DEMname)
	inputCSV = csv2list(DEMname)
	tempArrayX, tempArrayY, tempArrayZ = np.array(inputCSV).T
	csvXYZ = []

	''' interpolation method '''
	#library import
	if interpolType[0] == 'a':
		from scipy.interpolate import interp2d
	elif interpolType[0] == 'b':
		from pykrige.ok import OrdinaryKriging
	elif interpolType[0] == 'c':
		from pykrige.uk import UniversalKriging

	# scipy interpol1d
	if interpolType == 'a1': 
		tempInterpolated = interp2d(tempArrayX, tempArrayY, tempArrayZ, bounds_error=False)
		interpolZ = tempInterpolated(edgeXCoords, edgeYCoords)
		#print(interpolZ)
		#interpolZ = interpolZ[0,:]

	# pykrige ordinary kriging
	elif interpolType == 'b1':
		tempInterpolated = OrdinaryKriging(tempArrayX, tempArrayY, tempArrayZ, variogram_model='linear')
		interpolZ, stdZ = tempInterpolated.execute('grid', edgeXCoords, edgeYCoords)
	elif interpolType == 'b2':
		tempInterpolated = OrdinaryKriging(tempArrayX, tempArrayY, tempArrayZ, variogram_model='power')
		interpolZ,stdZ = tempInterpolated.execute('grid', edgeXCoords, edgeYCoords)
	elif interpolType == 'b3':
		tempInterpolated = OrdinaryKriging(tempArrayX, tempArrayY, tempArrayZ, variogram_model='gaussian')
		interpolZ,stdZ = tempInterpolated.execute('grid', edgeXCoords, edgeYCoords)
	elif interpolType == 'b4':
		tempInterpolated = OrdinaryKriging(tempArrayX, tempArrayY, tempArrayZ, variogram_model='spherical')
		interpolZ,stdZ = tempInterpolated.execute('grid', edgeXCoords, edgeYCoords)
	elif interpolType == 'b5':
		tempInterpolated = OrdinaryKriging(tempArrayX, tempArrayY, tempArrayZ, variogram_model='exponential')
		interpolZ,stdZ = tempInterpolated.execute('grid', edgeXCoords, edgeYCoords)
	elif interpolType == 'b6':
		tempInterpolated = OrdinaryKriging(tempArrayX, tempArrayY, tempArrayZ, variogram_model='hole-effect')
		interpolZ,stdZ = tempInterpolated.execute('grid', edgeXCoords, edgeYCoords)

	# pykrige universal kriging
	elif interpolType == 'c1':
		tempInterpolated = UniversalKriging(tempArrayX, tempArrayY, tempArrayZ, variogram_model='linear')
		interpolZ,stdZ = tempInterpolated.execute('grid', edgeXCoords, edgeYCoords)
	elif interpolType == 'c2':
		tempInterpolated = UniversalKriging(tempArrayX, tempArrayY, tempArrayZ, variogram_model='power')
		interpolZ,stdZ = tempInterpolated.execute('grid', edgeXCoords, edgeYCoords)
	elif interpolType == 'c3':
		tempInterpolated = UniversalKriging(tempArrayX, tempArrayY, tempArrayZ, variogram_model='gaussian')
		interpolZ,stdZ = tempInterpolated.execute('grid', edgeXCoords, edgeYCoords)
	elif interpolType == 'c4':
		tempInterpolated = UniversalKriging(tempArrayX, tempArrayY, tempArrayZ, variogram_model='spherical')
		interpolZ,stdZ = tempInterpolated.execute('grid', edgeXCoords, edgeYCoords)
	elif interpolType == 'c5':
		tempInterpolated = UniversalKriging(tempArrayX, tempArrayY, tempArrayZ, variogram_model='exponential')
		interpolZ,stdZ = tempInterpolated.execute('grid', edgeXCoords, edgeYCoords)
	elif interpolType == 'c6':
		tempInterpolated = UniversalKriging(tempArrayX, tempArrayY, tempArrayZ, variogram_model='hole-effect')
		interpolZ,stdZ = tempInterpolated.execute('grid', edgeXCoords, edgeYCoords)
	
	# for pykrige, eliminate points that has a large standard deviation
	#print(interpolZ)
	#print(len(interpolZ[0])*len(interpolZ))
	'''
	if interpolType[0] in ['b','c']:
		for loopZPredrow in range(len(interpolZ)):
			for loopZPredcol in range(len(interpolZ[0])):
				#print(stdZ[loopZPredrow][loopZPredcol])
				#print(stdZ[loopZPredrow][loopZPredcol] > 1000)
				#print(interpolZ[loopZPredrow][loopZPredcol])
				if stdZ[loopZPredrow][loopZPredcol] > stdMax:
					interpolZ[loopZPredrow][loopZPredcol] = np.nan
				else:
					continue
	'''
	#print(interpolZ)
	#print(stdZ)

	#print(len(edgeXCoords))
	#print(len(edgeYCoords))
	#print(len(interpolZ))
	#print(len(interpolZ[0]))

	if len(interpolZ[0]) < len(interpolZ):
		interpolZ = interpolZ.tolist()
	elif len(interpolZ[0]) >= len(interpolZ):
		interpolZ = np.transpose(interpolZ)
		interpolZ = interpolZ.tolist()

	#print(len(interpolZ))
	#print(len(interpolZ[0]))
	
	for loopXYrow in range(len(interpolZ)):
		for loopXYcol in range(len(interpolZ[0])):
			if (len(interpolZ)) <= (len(interpolZ[0])):
				csvXYZ.append([edgeXCoords[loopXYrow], edgeYCoords[loopXYcol], interpolZ[loopXYrow][loopXYcol]])
			elif (len(interpolZ)) > (len(interpolZ[0])):
				csvXYZ.append([edgeXCoords[loopXYcol], edgeYCoords[loopXYrow], interpolZ[loopXYrow][loopXYcol]])

	#print(csvXYZ)

	# export the interpolated data into csv file
	if exportOption==0:
		exportList2CSV('interpolated_'+DEMname, csvXYZ)

	return interpolZ, csvXYZ

# function that will calculate z-coordinates (from csv files) for given XY for each layer 
# at column edge
def DEM_3D_columnEdge(DEMNameList, DEMTypeList, DEMInterpolTypeList, canvasRange, colXmax, colYmax, stdMax=150):
	# import python libraries
	import numpy as np 
	#import scipy
	#import matplotlib.pyplot as plt

	# column X and Y coordinates edge
	initialSpaceX = abs(np.linspace(canvasRange[0], canvasRange[1], colXmax+1)[1] - np.linspace(canvasRange[0], canvasRange[1], colXmax+1)[0])
	initialSpaceY = abs(np.linspace(canvasRange[2], canvasRange[3], colYmax+1)[1] - np.linspace(canvasRange[2], canvasRange[3], colYmax+1)[0])

	if initialSpaceX == initialSpaceY:
		edgeXCoords = np.linspace(canvasRange[0], canvasRange[1], colXmax+1)
		edgeYCoords = np.linspace(canvasRange[2], canvasRange[3], colYmax+1)
	elif initialSpaceX < initialSpaceY:
		edgeXCoords = np.linspace(canvasRange[0], canvasRange[1], colXmax+1)
		edgeYCoords = np.arange(canvasRange[2], canvasRange[3], initialSpaceX)
	elif initialSpaceX > initialSpaceY:
		edgeXCoords = np.arange(canvasRange[0], canvasRange[1], initialSpaceY)
		edgeYCoords = np.linspace(canvasRange[2], canvasRange[3], colYmax+1)

	#print(initialSpaceX)
	#print(initialSpaceY)

	#print(edgeXCoords)
	
	#Input file sorting
	Z_pred = []
	for loopFile in range(len(DEMNameList)):
		interpolZ,csvXYZ = interpolation3D(DEMInterpolTypeList[loopFile], edgeXCoords, edgeYCoords, DEMNameList[loopFile], stdMax, exportOption=1)
		Z_pred.append(interpolZ)

	#print(Z_pred)

	edgesXY = {}
	# find XYZ for each material layer for given XY
	for loopX in range(len(edgeXCoords)):
		for loopY in range(len(edgeYCoords)):
			tempPtZ = [float(canvasRange[4])]
			tempPtM = ['bb']

			# input information to dictionary of edgesX
			for loopFile in range(len(DEMNameList)):
				#print(len(Z_pred[loopFile]))
				#print(len(Z_pred[loopFile][0]))
				#print(loopX)
				#print(loopY)
				interpolZ_edge = Z_pred[loopFile][loopX][loopY]
				#print(interpolY_edge)
				if not(np.isnan(interpolZ_edge)):
					if DEMTypeList[loopFile] == 'tt' and interpolZ_edge > canvasRange[5]:
						tempPtZ.append(float(canvasRange[5]))
						tempPtM.append('tt')
					elif DEMTypeList[loopFile] == 'tt' and interpolZ_edge <= canvasRange[5]:
						tempPtZ.append(interpolZ_edge)
						tempPtM.append('tt')
					elif DEMTypeList[loopFile] == 'rr' and interpolZ_edge > canvasRange[4]:
						tempPtZ[0] = interpolZ_edge
						tempPtM[0] = 'rr'
					else:
						tempPtZ.append(interpolZ_edge)
						tempPtM.append(DEMTypeList[loopFile])

			edgesXY[edgeXCoords[loopX],edgeYCoords[loopY]] = [tempPtZ, tempPtM]
			
	# find XY for each column edge
	colNedge = {}
	totalColNum = colXmax*colYmax
	count = 1
	rowCount = 0
	colCount = 0
	rowCountMax = len(edgeXCoords)-1
	colCountMax = len(edgeYCoords)-1
	while count <= totalColNum:
		tempList = []

		#print('rowCount=%f'%rowCount)
		#print('colCount=%f'%colCount)

		edge1 = edgesXY[edgeXCoords[rowCount],edgeYCoords[colCount]]
		tempList.append([edgeXCoords[rowCount],edgeYCoords[colCount]])
		tempList.append(edge1)

		edge2 = edgesXY[edgeXCoords[rowCount+1],edgeYCoords[colCount]]
		tempList.append([edgeXCoords[rowCount+1],edgeYCoords[colCount]])
		tempList.append(edge2)

		edge3 = edgesXY[edgeXCoords[rowCount+1],edgeYCoords[colCount+1]]
		tempList.append([edgeXCoords[rowCount+1],edgeYCoords[colCount+1]])
		tempList.append(edge3)

		edge4 = edgesXY[edgeXCoords[rowCount],edgeYCoords[colCount+1]]
		tempList.append([edgeXCoords[rowCount],edgeYCoords[colCount+1]])
		tempList.append(edge4)

		colNedge[count] = tempList

		if (rowCount+1) == rowCountMax and (colCount+1) == colCountMax:	# last column
			break
		elif (rowCount+1) == rowCountMax and (colCount+1) != colCountMax: # move to next row
			rowCount = 0
			colCount += 1
		elif (rowCount+1) != rowCountMax:	# move along the row
			rowCount += 1

		count += 1 

	return colNedge

# function that will calculate z-coordinates (from csv files) for given XY for each layer 
# at column center
def DEM_3D_columnCenter(colNedge, DEMNameList, DEMTypeList, DEMInterpolTypeList, canvasRange, colXmax, colYmax, stdMax=150):
	# import python libraries
	import numpy as np 
	#import scipy
	#import matplotlib.pyplot as plt

	# column X and Y coordinates edge
	initialSpaceX = abs(np.linspace(canvasRange[0], canvasRange[1], colXmax+1)[1] - np.linspace(canvasRange[0], canvasRange[1], colXmax+1)[0])
	initialSpaceY = abs(np.linspace(canvasRange[2], canvasRange[3], colYmax+1)[1] - np.linspace(canvasRange[2], canvasRange[3], colYmax+1)[0])
	#print(initialSpaceX)
	#print(initialSpaceY)

	if initialSpaceX == initialSpaceY:
		edgeXCoords = np.linspace(canvasRange[0]+initialSpaceX, canvasRange[1]-initialSpaceX, colXmax)
		edgeYCoords = np.linspace(canvasRange[2]+initialSpaceY, canvasRange[3]-initialSpaceY, colYmax)
	elif initialSpaceX < initialSpaceY:
		edgeXCoords = np.linspace(canvasRange[0]+initialSpaceX, canvasRange[1]-initialSpaceX, colXmax)
		edgeYCoords = np.arange(canvasRange[2]+initialSpaceX, canvasRange[3]-initialSpaceX, initialSpaceX)
	elif initialSpaceX > initialSpaceY:
		edgeXCoords = np.arange(canvasRange[0]+initialSpaceY, canvasRange[1]-initialSpaceY, initialSpaceY)
		edgeYCoords = np.linspace(canvasRange[2]+initialSpaceY, canvasRange[3]-initialSpaceY, colYmax)
	
	'''
	if len(edgeXCoords) < len(edgeYCoords):
		edgeXCoords = np.arange(canvasRange[0]+initialSpaceY, canvasRange[1]-initialSpaceY, initialSpaceY)
	elif len(edgeXCoords) > len(edgeYCoords):
		edgeYCoords = np.arange(canvasRange[0]+initialSpaceY, canvasRange[1]-initialSpaceY, initialSpaceY)
	'''
	#print(edgeXCoords)
	#print(len(edgeXCoords))
	#print(edgeYCoords)
	#print(len(edgeYCoords))
	
	#Input file sorting
	Z_pred = []
	for loopFile in range(len(DEMNameList)):
		interpolZ, csvXYZ = interpolation3D(DEMInterpolTypeList[loopFile], edgeXCoords, edgeYCoords, DEMNameList[loopFile], stdMax, exportOption=0)
		Z_pred.append(interpolZ)

	#print(Z_pred)

	edgesXY = {}
	# find XYZ for each material layer for given XY
	for loopX in range(len(edgeXCoords)):
		for loopY in range(len(edgeYCoords)):
			tempPtZ = [float(canvasRange[4])]
			tempPtM = ['bb']

			# input information to dictionary of edgesX
			for loopFile in range(len(DEMNameList)):
				#print(Z_pred[loopFile][loopX][loopY])
				interpolZ_edge = Z_pred[loopFile][loopX][loopY]
				#print(interpolY_edge)
				if not(np.isnan(interpolZ_edge)):
					if DEMTypeList[loopFile] == 'tt' and interpolZ_edge > canvasRange[5]:
						tempPtZ.append(float(canvasRange[5]))
						tempPtM.append('tt')
					elif DEMTypeList[loopFile] == 'tt' and interpolZ_edge <= canvasRange[5]:
						tempPtZ.append(interpolZ_edge)
						tempPtM.append('tt')
					elif DEMTypeList[loopFile] == 'rr' and interpolZ_edge > canvasRange[4]:
						tempPtZ[0] = interpolZ_edge
						tempPtM[0] = 'rr'
					else:
						tempPtZ.append(interpolZ_edge)
						tempPtM.append(DEMTypeList[loopFile])

			edgesXY[edgeXCoords[loopX],edgeYCoords[loopY]] = [tempPtZ, tempPtM]

	# find XY for each column edge
	rowCount = 0
	colCount = 0
	rowCountMax = colXmax
	colCountMax = colYmax
	for colN in colNedge.keys():
		tempList = colNedge[colN]

		center = edgesXY[edgeXCoords[rowCount],edgeYCoords[colCount]]
		tempList.append([edgeXCoords[rowCount],edgeYCoords[colCount]])
		tempList.append(center)

		colNedge[colN] = tempList

		if rowCount+1 == rowCountMax and colCount+1 == colCountMax:	# last column
			break
		elif rowCount+1 == rowCountMax and colCount+1 != colCountMax: # move to next row
			rowCount = 0
			colCount += 1
		elif rowCount+1 != rowCountMax:	# move along the row
			rowCount += 1

	return colNedge

''' DEM points of 3D slip surface '''
'''
SSTypeList = 1	-> user-defined surface [DEMname, interpolType]
SSTypeList = 2	-> grid eplitical search [pt0x, pt0y, pt0z, xr, yr, zr]

output
#colNedge format = [[X1, Y1], [[Z..],[type...]...], an exmaple below
#1: [[1795700.0, 498790.0], [[700.0, 758.2936221107052, 758.2936221107052], ['bb', 'w1', 'tt']], [1795726.5, 498790.0], [[700.0, 758.3716389809305, 758.3716389809305], ['bb', 'w1', 'tt']], [1795726.5, 498791.0], [[700.0, 708.1436968600685, 708.1436968600685], ['bb', 'w1', 'tt']], [1795700.0, 498791.0], [[700.0, 706.7219523927594, 706.7219523927594], ['bb', 'w1', 'tt']]]
'''
# find base y-coordinates for a given slip surface
def SS_3D_columnsEdge(SSTypeList, inputPara, canvasRange, colXmax, colYmax, colNedge, stdMax=150):
	# import python libraries
	import numpy as np 
	#import scipy
	#from scipy.interpolate import interp1d
	#from pykrige.ok import OrdinaryKriging
	#from pykrige.uk import UniversalKriging
	#import matplotlib.pyplot as plt

	newColEdge = {}
	ss_csv = []

	# column X and Y coordinates edge
	initialSpaceX = abs(np.linspace(canvasRange[0], canvasRange[1], colXmax+1)[1] - np.linspace(canvasRange[0], canvasRange[1], colXmax+1)[0])
	initialSpaceY = abs(np.linspace(canvasRange[2], canvasRange[3], colYmax+1)[1] - np.linspace(canvasRange[2], canvasRange[3], colYmax+1)[0])

	if initialSpaceX == initialSpaceY:
		edgeXCoords = np.linspace(canvasRange[0], canvasRange[1], colXmax+1)
		edgeYCoords = np.linspace(canvasRange[2], canvasRange[3], colYmax+1)
	elif initialSpaceX < initialSpaceY:
		edgeXCoords = np.linspace(canvasRange[0], canvasRange[1], colXmax+1)
		edgeYCoords = np.arange(canvasRange[2], canvasRange[3], initialSpaceX)
	elif initialSpaceX > initialSpaceY:
		edgeXCoords = np.arange(canvasRange[0], canvasRange[1], initialSpaceY)
		edgeYCoords = np.linspace(canvasRange[2], canvasRange[3], colYmax+1)

	#print(edgeXCoords)
	
	'''Input SS for each XY coordiantes'''
	ss_Z_pred = {}		#keys = XY, value = Z

	if SSTypeList == 2:		# ellipsoid grid search
		# extract inputPara 
		pt0X = inputPara[0]
		pt0Y = inputPara[1]
		pt0Z = inputPara[2]
		xr = inputPara[3]
		yr = inputPara[4]
		zr = inputPara[5]

		for loopX in range(len(edgeXCoords)):
			for loopY in range(len(edgeYCoords)):
				if (1 - ((edgeXCoords[loopX] - pt0X)**2)/(xr**2) - ((edgeYCoords[loopY] - pt0Y)**2)/(yr**2)) >= 0:

					tempPtZSS = pt0Z - zr*np.sqrt(1 - ((edgeXCoords[loopX] - pt0X)**2)/(xr**2) - ((edgeYCoords[loopY] - pt0Y)**2)/(yr**2))
				else:
					tempPtZSS = np.nan
			
				ss_Z_pred[edgeXCoords[loopX], edgeYCoords[loopY]] = tempPtZSS

	elif SSTypeList == 1:		#  user-defined surface
		# extract inputPara 
		DEMname = inputPara[0]
		interpolType = inputPara[1]

		tempPtZSS,csvFile = interpolation3D(interpolType, edgeXCoords, edgeYCoords, DEMname, stdMax, exportOption=1)

		for loopX in range(len(edgeXCoords)):
			for loopY in range(len(edgeYCoords)):
				ss_Z_pred[edgeXCoords[loopX], edgeYCoords[loopY]] = tempPtZSS[loopX][loopY]

	colNumberMax = colXmax*colYmax
	for loopCol in range(1,colNumberMax+1):

		# extract XY of each corner of the columns 
		corner1 = colNedge[loopCol][0]
		corner2 = colNedge[loopCol][2]
		corner3 = colNedge[loopCol][4]
		corner4 = colNedge[loopCol][6]

		# slip surface z coordinates
		corner1z_ss = ss_Z_pred[corner1[0],corner1[1]]
		corner2z_ss = ss_Z_pred[corner2[0],corner2[1]]
		corner3z_ss = ss_Z_pred[corner3[0],corner3[1]]
		corner4z_ss = ss_Z_pred[corner4[0],corner4[1]]

		# check if isnan
		corner1_check = 0
		if not(np.isnan(corner1z_ss)):
			if corner1z_ss <= colNedge[loopCol][1][0][-1]:
				corner1_check = 1
		
		corner2_check = 0
		if not(np.isnan(corner2z_ss)):
			if corner2z_ss <= colNedge[loopCol][3][0][-1]:
				corner2_check = 1
		
		corner3_check = 0
		if not(np.isnan(corner3z_ss)):
			if corner3z_ss <= colNedge[loopCol][5][0][-1]:
				corner3_check = 1
			
		corner4_check = 0
		if not(np.isnan(corner4z_ss)):
			if corner4z_ss <= colNedge[loopCol][7][0][-1]:
				corner4_check = 1

		if corner1_check!=1 or corner2_check!=1 or corner3_check!=1 or corner4_check!=1:
			continue

		elif corner1_check==1 and corner2_check==1 and corner3_check==1 and corner4_check==1:
			# list of Z coordinates
			corner1z = colNedge[loopCol][1][0]
			corner2z = colNedge[loopCol][3][0]
			corner3z = colNedge[loopCol][5][0]
			corner4z = colNedge[loopCol][7][0]

			# list of Z coorindate types
			corner1z_type = colNedge[loopCol][1][1]
			corner2z_type = colNedge[loopCol][3][1]
			corner3z_type = colNedge[loopCol][5][1]
			corner4z_type = colNedge[loopCol][7][1]

			# corner 1
			nslice1 = [0]
			nslice1_type = [0]
			tempZss = []
			tempZssType = []
			for loopLayer in range(len(corner1z_type)):
				if corner1z_type[loopLayer][0] in ['t','m','g']:
					nslice1.append(corner1z[loopLayer])
					nslice1_type.append(corner1z_type[loopLayer])

				elif corner1z_type[loopLayer][0] in ['r','w','b']:
					if loopLayer == 0:
						tempZss.append(corner1z_ss) 
						tempZssType.append('ss') 

					if corner1z_ss <= corner1z[loopLayer]:
						tempZss.append(corner1z[loopLayer]) 
						tempZssType.append(corner1z_type[loopLayer]) 
			
			maxEdgeBottom = max(tempZss)
			maxEdgeBottomIDX = tempZss.index(maxEdgeBottom)
			maxEdgeBottomType = tempZssType[maxEdgeBottomIDX]
			
			nslice1[0] = maxEdgeBottom
			nslice1_type[0] = maxEdgeBottomType

			# corner 2
			nslice2 = [0]
			nslice2_type = [0]
			tempZss = []
			tempZssType = []
			for loopLayer in range(len(corner2z_type)):
				if corner2z_type[loopLayer][0] in ['t','m','g']:
					nslice2.append(corner2z[loopLayer])
					nslice2_type.append(corner2z_type[loopLayer])

				elif corner2z_type[loopLayer][0] in ['r','w','b']:
					if loopLayer == 0:
						tempZss.append(corner2z_ss) 
						tempZssType.append('ss') 

					if corner2z_ss <= corner2z[loopLayer]:
						tempZss.append(corner2z[loopLayer]) 
						tempZssType.append(corner2z_type[loopLayer]) 
			
			maxEdgeBottom = max(tempZss)
			maxEdgeBottomIDX = tempZss.index(maxEdgeBottom)
			maxEdgeBottomType = tempZssType[maxEdgeBottomIDX]
			
			nslice2[0] = maxEdgeBottom
			nslice2_type[0] = maxEdgeBottomType


			# corner 3
			nslice3 = [0]
			nslice3_type = [0]
			tempZss = []
			tempZssType = []
			for loopLayer in range(len(corner3z_type)):
				if corner3z_type[loopLayer][0] in ['t','m','g']:
					nslice3.append(corner3z[loopLayer])
					nslice3_type.append(corner3z_type[loopLayer])

				elif corner3z_type[loopLayer][0] in ['r','w','b']:
					if loopLayer == 0:
						tempZss.append(corner3z_ss) 
						tempZssType.append('ss') 

					if corner3z_ss <= corner3z[loopLayer]:
						tempZss.append(corner3z[loopLayer]) 
						tempZssType.append(corner3z_type[loopLayer]) 
			
			maxEdgeBottom = max(tempZss)
			maxEdgeBottomIDX = tempZss.index(maxEdgeBottom)
			maxEdgeBottomType = tempZssType[maxEdgeBottomIDX]
			
			nslice3[0] = maxEdgeBottom
			nslice3_type[0] = maxEdgeBottomType

			# corner 4
			nslice4 = [0]
			nslice4_type = [0]
			tempZss = []
			tempZssType = []
			for loopLayer in range(len(corner4z_type)):
				if corner4z_type[loopLayer][0] in ['t','m','g']:
					nslice4.append(corner4z[loopLayer])
					nslice4_type.append(corner4z_type[loopLayer])

				elif corner4z_type[loopLayer][0] in ['r','w','b']:
					if loopLayer == 0:
						tempZss.append(corner4z_ss) 
						tempZssType.append('ss') 

					if corner4z_ss <= corner4z[loopLayer]:
						tempZss.append(corner4z[loopLayer]) 
						tempZssType.append(corner4z_type[loopLayer]) 
			
			maxEdgeBottom = max(tempZss)
			maxEdgeBottomIDX = tempZss.index(maxEdgeBottom)
			maxEdgeBottomType = tempZssType[maxEdgeBottomIDX]
			
			nslice4[0] = maxEdgeBottom
			nslice4_type[0] = maxEdgeBottomType

			# slip surface XYZ csv file
			ss_csv.append([corner1[0], corner1[1], nslice1[0]])
			ss_csv.append([corner2[0], corner2[1], nslice2[0]])
			ss_csv.append([corner3[0], corner3[1], nslice3[0]])
			ss_csv.append([corner4[0], corner4[1], nslice4[0]])

			newColEdge[loopCol] = [corner1, [nslice1, nslice1_type], corner2, [nslice2, nslice2_type], corner3, [nslice3, nslice3_type], corner4, [nslice4, nslice4_type], colNedge[loopCol][8], colNedge[loopCol][9]]

	#print(ss_csv)
	if len(newColEdge.keys()) == 0:
		return None
	else:
		#exportList2CSV('interpolated_ss_type'+str(SSTypeList)+'.csv', ss_csv)
		return newColEdge, ss_csv

''' DEM points of 3D slip surface '''
'''
SSTypeList = 1	-> user-defined surface [DEMname, interpolType]
SSTypeList = 2	-> grid eplitical search [pt0x, pt0y, pt0z, xr, yr, zr]

output
#colNedge format = [[X1, Y1], [[Z..],[type...]..., [Xc, Yc],[[Z...],[type...]]]
'''
# find base z-coordinates for a given slip surface
def SS_3D_columnsCenter(SSTypeList, inputPara, canvasRange, colXmax, colYmax, newColEdge, stdMax=150):
	# import python libraries
	import numpy as np 
	#import scipy
	#from scipy.interpolate import interp1d
	#from pykrige.ok import OrdinaryKriging
	#from pykrige.uk import UniversalKriging
	#import matplotlib.pyplot as plt

	ss_csv = []

	# column X and Y coordinates center
	initialSpaceX = abs(np.linspace(canvasRange[0], canvasRange[1], colXmax+1)[1] - np.linspace(canvasRange[0], canvasRange[1], colXmax+1)[0])
	initialSpaceY = abs(np.linspace(canvasRange[2], canvasRange[3], colYmax+1)[1] - np.linspace(canvasRange[2], canvasRange[3], colYmax+1)[0])

	if initialSpaceX == initialSpaceY:
		edgeXCoords = np.linspace(canvasRange[0]+initialSpaceX, canvasRange[1]-initialSpaceX, colXmax)
		edgeYCoords = np.linspace(canvasRange[2]+initialSpaceY, canvasRange[3]-initialSpaceY, colYmax)
	elif initialSpaceX < initialSpaceY:
		edgeXCoords = np.linspace(canvasRange[0]+initialSpaceX, canvasRange[1]-initialSpaceX, colXmax)
		edgeYCoords = np.arange(canvasRange[2]+initialSpaceX, canvasRange[3]-initialSpaceX, initialSpaceX)
	elif initialSpaceX > initialSpaceY:
		edgeXCoords = np.arange(canvasRange[0]+initialSpaceY, canvasRange[1]-initialSpaceY, initialSpaceY)
		edgeYCoords = np.linspace(canvasRange[2]+initialSpaceY, canvasRange[3]-initialSpaceY, colYmax)

	#colNumberMax = colXmax*colYmax
	#print(edgeXCoords)
	
	'''Input SS for each XY coordiantes'''
	ss_Z_pred = {}		#keys = XY, value = Z

	if SSTypeList == 2:		# ellipsoid grid search
		# extract inputPara 
		pt0X = inputPara[0]
		pt0Y = inputPara[1]
		pt0Z = inputPara[2]
		xr = inputPara[3]
		yr = inputPara[4]
		zr = inputPara[5]

		for loopX in range(len(edgeXCoords)):
			for loopY in range(len(edgeYCoords)):
				if (1 - ((edgeXCoords[loopX] - pt0X)**2)/(xr**2) - ((edgeYCoords[loopY] - pt0Y)**2)/(yr**2)) >= 0:

					tempPtZSS = pt0Z - zr*np.sqrt(1 - ((edgeXCoords[loopX] - pt0X)**2)/(xr**2) - ((edgeYCoords[loopY] - pt0Y)**2)/(yr**2))
				else:
					tempPtZSS = np.nan
			
				ss_Z_pred[edgeXCoords[loopX], edgeYCoords[loopY]] = tempPtZSS

	elif SSTypeList == 1:		#  user-defined surface
		# extract inputPara 
		DEMname = inputPara[0]
		interpolType = inputPara[1]

		tempPtZSS,csvFile = interpolation3D(interpolType, edgeXCoords, edgeYCoords, DEMname, stdMax, exportOption=0)

		for loopX in range(len(edgeXCoords)):
			for loopY in range(len(edgeYCoords)):
				ss_Z_pred[edgeXCoords[loopX], edgeYCoords[loopY]] = tempPtZSS[loopX][loopY]

	for loopCol in newColEdge.keys():

		# extract XY of each corner of the columns 
		#print(loopCol)
		#print((newColEdge[loopCol]))
		center = newColEdge[loopCol][8]

		# slip surface z coordinates
		center_ss = ss_Z_pred[center[0],center[1]]

		# list of Z coordinates
		centerz = newColEdge[loopCol][9][0]

		# list of Z coorindate types
		centerz_type = newColEdge[loopCol][9][1]

		# corner 1
		ncenterz = [0]
		ncenterz_type = [0]
		tempZss = []
		tempZssType = []
		for loopLayer in range(len(centerz_type)):
			#print(centerz_type[loopLayer])
			if centerz_type[loopLayer][0] in ['t','m','g']:
				ncenterz.append(centerz[loopLayer])
				ncenterz_type.append(centerz_type[loopLayer])

			elif centerz_type[loopLayer][0] in ['r','w','b']:
				if loopLayer == 0:
					tempZss.append(center_ss) 
					tempZssType.append('ss') 

				if center_ss <= centerz[loopLayer]:
					tempZss.append(centerz[loopLayer]) 
					tempZssType.append(centerz_type[loopLayer]) 
		
		maxEdgeBottom = max(tempZss)
		maxEdgeBottomIDX = tempZss.index(maxEdgeBottom)
		maxEdgeBottomType = tempZssType[maxEdgeBottomIDX]
		
		ncenterz[0] = maxEdgeBottom
		ncenterz_type[0] = maxEdgeBottomType

		# slip surface XYZ csv file
		ss_csv.append([center[0], center[1], ncenterz[0]])

		newColEdge[loopCol][9] = [ncenterz, ncenterz_type]

	#print(ss_csv)
	if len(newColEdge.keys()) == 0:
		return None
	else:
		exportList2CSV('interpolated_ss_type'+str(SSTypeList)+'.csv', ss_csv)
		return newColEdge, ss_csv

# find center of rotation and radius of user-defined slip surface
def findpt0nR_approxSphere_3D(ss_csv):
	import numpy as np

	midpt = np.random.choice(np.arange(1,len(ss_csv)-1),2)
	if midpt[0] == midpt[1]:
		midpt[1] += 1

	P1x = ss_csv[0][0]
	P1y = ss_csv[0][1]
	P1z = ss_csv[0][2]
	
	P2x = ss_csv[midpt[0]][0]
	P2y = ss_csv[midpt[0]][1]
	P2z = ss_csv[midpt[0]][2]
	
	P3x = ss_csv[midpt[1]][0]
	P3y = ss_csv[midpt[1]][1]
	P3z = ss_csv[midpt[1]][2]
	
	P4x = ss_csv[-1][0]
	P4y = ss_csv[-1][1]
	P4z = ss_csv[-1][2]

	DistSq1 = -(P1x**2 + P1y**2 + P1z**2)
	DistSq2 = -(P2x**2 + P2y**2 + P2z**2)
	DistSq3 = -(P3x**2 + P3y**2 + P3z**2)
	DistSq4 = -(P4x**2 + P4y**2 + P4z**2)

	Mx = np.array([[DistSq1, P1y, P1z, 1], [DistSq2, P2y, P2z, 1], [DistSq1, P3y, P3z, 1], [DistSq4, P4y, P4z, 1]])
	My = np.array([[P1x, DistSq1, P1z, 1], [P2x, DistSq2, P2z, 1], [P3x, DistSq3, P3z, 1], [P4x, DistSq4, P4z, 1]])
	Mz = np.array([[P1x, P1y, DistSq1, 1], [P2x, P2y, DistSq2, 1], [P3x, P3y, DistSq3, 1], [P4x, P4y, DistSq4, 1]])
	Mr = np.array([[P1x, P1y, P1z, DistSq1], [P2x, P2y, P2z, DistSq2], [P3x, P3y, P3z, DistSq3], [P4x, P4y, P4z, DistSq4]])
	T = np.array([[P1x, P1y, P1z, 1], [P2x, P2y, P2z, 1], [P3x, P3y, P3z, 1], [P4x, P4y, P4z, 1]])

	pt0X = -0.5*np.linalg.det(Mx)/np.linalg.det(T)
	pt0Y = -0.5*np.linalg.det(My)/np.linalg.det(T)
	pt0Z = -0.5*np.linalg.det(Mz)/np.linalg.det(T)

	R = 0.5*np.sqrt(pt0X**2 + pt0Y**2 + pt0Z**2 - 4*(np.linalg.det(Mr)/np.linalg.det(T)))

	return [round(pt0X,3), round(pt0Y,3), round(pt0Z,3), round(R,3)]


# The appropriate input for this function is a list of tuples in the format
# [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3)]
# output = dip, dipDirection, strike
def dip_dipDirection_from3pts(pts):
	import math

	#print(pts)

	ptA, ptB, ptC = pts[0], pts[1], pts[2]
	x1, y1, z1 = float(ptA[0]), float(ptA[1]), float(ptA[2])
	x2, y2, z2 = float(ptB[0]), float(ptB[1]), float(ptB[2])
	x3, y3, z3 = float(ptC[0]), float(ptC[1]), float(ptC[2])

	u1 = float(((y1 - y2) * (z3 - z2) - (y3 - y2) * (z1 - z2)))
	u2 = float((-((x1 - x2) * (z3 - z2) - (x3 - x2) * (z1 - z2))))
	u3 = float(((x1 - x2) * (y3 - y2) - (x3 - x2) * (y1 - y2)))

	# determine dip
	#print(strike, 'strike')
	if abs(z3-z1) < 0.01 and abs(z2-z1)<0.01 and abs(z2-z3)<0.01:
		dip = 0
	else:
		part1_dip = math.sqrt(u2**2 + u1**2)
		part2_dip = math.sqrt(u1**2 + u2**2 + u3**2)
		dip = math.degrees(math.asin(part1_dip / part2_dip))
	
	'''
	Calculate pseudo eastings and northings from origin
	these are actually coordinates of a new point that represents
	the normal from the plane's origin defined as (0,0,0).
	
	If the z value (u3) is above the plane we first reverse the easting
	then we check if the z value (u3) is below the plane, if so
	we reverse the northing. 
	
	This is to satisfy the right hand rule in geology where dip is always
	to the right if looking down strike. 
	'''
	if dip == 0:
		dipDirection = 999
		strike = 999
	else:
		'''
		if u3 < 0:
			easting = u2
		else:
			easting = -u2

		if u3 > 0:
			northing = u1
		else:
			northing = -u1
		'''
		easting = u2
		northing = -u1

		if easting >= 0:
			partA_strike = (easting**2) + (northing**2)
			strike = math.degrees(math.acos(northing / math.sqrt(partA_strike)))
		else:
			partA_strike = northing / math.sqrt((easting**2) + (northing**2))
			strike = math.degrees(2 * math.pi - math.acos(partA_strike))

		dipDirection = (strike+90)%360

	return round(dip,2), round(dipDirection,2), round(strike,2)

'''calculate area in 3D plane'''
# pts = [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x4, y4, z4)]
def area_3d(pts):   
	import numpy as np     
	
	p1 = pts[0]
	p2 = pts[1]
	p3 = pts[2]
	p4 = pts[3] 
	
	vector21 = [p2[0]-p1[0], p2[1]-p1[1], p2[2]-p1[2]]
	vector41 = [p4[0]-p1[0], p4[1]-p1[1], p4[2]-p1[2]]

	vector23 = [p2[0]-p3[0], p2[1]-p3[1], p2[2]-p3[2]]
	vector43 = [p4[0]-p3[0], p4[1]-p3[1], p4[2]-p3[2]]

	area = 0.5*np.linalg.norm(np.cross(vector21,vector41)) + 0.5*np.linalg.norm(np.cross(vector23,vector43))

	return area

'''calculate tetrahedron volume'''
def vol_tetra(pt1, pt2, pt3, pt4):
	# based on vector points method
	# import modules
	import numpy as np
	
	row1 = pt1[:]
	row1.append(1)
	row2 = pt2[:]
	row2.append(1)
	row3 = pt3[:]
	row3.append(1)
	row4 = pt4[:]
	row4.append(1)
	
	volM = np.array([row1, row2, row3, row4])
	v = abs(np.linalg.det(volM))/6

	return v

'''calculate a volume of box (can be irregular) from 8 tetrahedron volumes'''
def vol_box_from_tetras(base4Pts, top4Pts):
	# calculate center of gravity point of base point 4 base points and 4 top points
	baseCenPt = [0.5*(base4Pts[0][0]+base4Pts[1][0]), 0.5*(base4Pts[0][1]+base4Pts[3][1]), 0]
	baseCenPt[2] = 0.5*(0.5*(base4Pts[2][2]+base4Pts[3][2])+0.5*(base4Pts[0][2]+base4Pts[1][2]))

	topCenPt = [0.5*(top4Pts[0][0]+top4Pts[1][0]), 0.5*(top4Pts[0][1]+top4Pts[3][1]), 0]
	topCenPt[2] = 0.5*(0.5*(top4Pts[2][2]+top4Pts[3][2])+0.5*(top4Pts[0][2]+top4Pts[1][2]))
	
	v1 = vol_tetra(base4Pts[0], base4Pts[1], base4Pts[3], top4Pts[0])
	v2 = vol_tetra(base4Pts[2], base4Pts[1], base4Pts[3], top4Pts[2])
	v3 = vol_tetra(top4Pts[1], top4Pts[0], top4Pts[2], base4Pts[1])
	v4 = vol_tetra(top4Pts[3], top4Pts[0], top4Pts[2], base4Pts[3])
	v5 = vol_tetra(baseCenPt, topCenPt, top4Pts[0], base4Pts[1])
	v6 = vol_tetra(baseCenPt, topCenPt, top4Pts[0], base4Pts[3])
	v7 = vol_tetra(baseCenPt, topCenPt, top4Pts[2], base4Pts[1])
	v8 = vol_tetra(baseCenPt, topCenPt, top4Pts[2], base4Pts[3])
	
	return v1+v2+v3+v4+v5+v6+v7+v8

# GW level, pwp
def GW_3D(inputFile, water_unitWeight):
	import math

	dictKeys = inputFile.keys()

	output = {}
	for loopCol in dictKeys:

		# check presence of GW level
		checkGW_1 = 0
		if 'gw' in inputFile[loopCol][1][1]:
			gw1index = inputFile[loopCol][1][1]
			gw1index = gw1index.index('gw')
			gwZ1 = inputFile[loopCol][1][0][gw1index]
			x1 = inputFile[loopCol][0][0]
			y1 = inputFile[loopCol][0][1]
			zt1 = inputFile[loopCol][1][0][-1]
			zb1 = inputFile[loopCol][1][0][0]
			checkGW_1 = 1

		checkGW_2 = 0
		if 'gw' in inputFile[loopCol][3][1]:
			gw2index = inputFile[loopCol][3][1]
			gw2index = gw2index.index('gw')
			gwZ2 = inputFile[loopCol][3][0][gw2index]
			x2 = inputFile[loopCol][2][0]
			y2 = inputFile[loopCol][2][1]
			zt2 = inputFile[loopCol][3][0][-1]
			zb2 = inputFile[loopCol][3][0][0]
			checkGW_2 = 1

		checkGW_3 = 0
		if 'gw' in inputFile[loopCol][5][1]:
			gw3index = inputFile[loopCol][5][1]
			gw3index = gw3index.index('gw')
			gwZ3 = inputFile[loopCol][5][0][gw3index]
			x3 = inputFile[loopCol][4][0]
			y3 = inputFile[loopCol][4][1]
			zt3 = inputFile[loopCol][5][0][-1]
			zb3 = inputFile[loopCol][5][0][0]
			checkGW_3 = 1

		checkGW_4 = 0
		if 'gw' in inputFile[loopCol][7][1]:
			gw4index = inputFile[loopCol][7][1]
			gw4index = gw4index.index('gw')
			gwZ4 = inputFile[loopCol][7][0][gw4index]
			x4 = inputFile[loopCol][6][0]
			y4 = inputFile[loopCol][6][1]
			zt4 = inputFile[loopCol][7][0][-1]
			zb4 = inputFile[loopCol][7][0][0]
			checkGW_4 = 1
		
		if checkGW_1 == 0 or checkGW_2 == 0 or checkGW_3 == 0 or checkGW_4 == 0:
			continue

		elif checkGW_1 == 1 and checkGW_2 == 1 and checkGW_3 == 1 and checkGW_4 == 1:
			
			#centralX = inputFile[loopCol][8][0]
			#centralY = inputFile[loopCol][8][1]
			#print(inputFile[loopCol][9][0])
			#print(inputFile[loopCol][9][1])
			centralZb = inputFile[loopCol][9][0][0]
			centralZt = inputFile[loopCol][9][0][-1]

			if 'gw' in inputFile[loopCol][9][1]:
				gwIDX = inputFile[loopCol][9][1].index('gw')
				gw_centralZ = inputFile[loopCol][9][0][gwIDX]

			pts = [(x1, y1, gwZ1), (x2, y2, gwZ2), (x3, y3, gwZ3)]
			dip, dipDirection, strike = dip_dipDirection_from3pts(pts)

			# calculate hw for left, right, base, top
			if gw_centralZ > centralZt:
				hw_t = abs(gw_centralZ-centralZt)*(math.cos(dip))**2
				hw_b = abs(gw_centralZ-centralZb)*(math.cos(dip))**2
				hw_b_net = hw_b - hw_t
			else:
				hw_b_net = (gw_centralZ-centralZb)*(math.cos(dip))**2

			output[loopCol] = [hw_b_net*water_unitWeight]

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
		if matricSuction >= inputPara[1]:
			unsatShearStrength = min([matricSuction, inputPara[2]])*np.tan(np.radians(inputPara[0]))
			calcPhiC[1] += unsatShearStrength

	return calcPhiC

''' main function - 3D slope stability support forces calculations from class of support type '''
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

'''
output
#colNedge format = [[X1, Y1], [[Z..],[type...]...], an exmaple below
#1: [[1795700.0, 498790.0], [[700.0, 758.2936221107052, 758.2936221107052], ['bb', 'w1', 'tt']], [1795726.5, 498790.0], [[700.0, 758.3716389809305, 758.3716389809305], ['bb', 'w1', 'tt']], [1795726.5, 498791.0], [[700.0, 708.1436968600685, 708.1436968600685], ['bb', 'w1', 'tt']], [1795700.0, 498791.0], [[700.0, 706.7219523927594, 706.7219523927594], ['bb', 'w1', 'tt']]]
'''
# find geometric data (area, base, width) for each slice
def createInputfile4Analysis_3D_columns(newColData, materialClass, canvasRange, water_unitWeight, tensionCrackAngle=None):
	# import python libraries
	import numpy as np 
	#import scipy
	#import matplotlib.pyplot as plt
	
	analysisInputFile = []

	colKeyList = newColData.keys()

	# base area, base length, inclination (base, top), side left length, side right length
	colInfo_dip_dipDirection = {}
	colInfo_area = {}
	colInfo_vol = {}
	colInfo_W = {}  # weight
	for colN in colKeyList:

		# dip and dip direction
		# [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3)]
		# output = dip, dipDirection, strike
		#print(colN)
		#pts = [(newColData[colN][0][0], newColData[colN][0][1], newColData[colN][1][0][0]), (newColData[colN][2][0], newColData[colN][2][1], newColData[colN][3][0][0]), (newColData[colN][4][0], newColData[colN][4][1], newColData[colN][5][0][0])]
		pts = [(newColData[colN][0][0], newColData[colN][0][1], newColData[colN][1][0][0]), (newColData[colN][2][0], newColData[colN][2][1], newColData[colN][3][0][0]), (newColData[colN][6][0], newColData[colN][6][1], newColData[colN][7][0][0])]
		dip, dipDirection, strike = dip_dipDirection_from3pts(pts)

		colInfo_dip_dipDirection[colN] = [dip, dipDirection]

		# area - base and side total
		# pts = [(x1, y1, z1), (x2, y2, z2), (x3, y3, z3), (x4, y4, z4)]
		baseApts = [(newColData[colN][0][0], newColData[colN][0][1], newColData[colN][1][0][0]), (newColData[colN][2][0], newColData[colN][2][1], newColData[colN][3][0][0]), (newColData[colN][4][0], newColData[colN][4][1], newColData[colN][5][0][0]), (newColData[colN][6][0], newColData[colN][6][1], newColData[colN][7][0][0])]
		baseA = area_3d(baseApts)

		side12pts = [(newColData[colN][0][0], newColData[colN][0][1], newColData[colN][1][0][0]), (newColData[colN][0][0], newColData[colN][0][1], newColData[colN][1][0][-1]), (newColData[colN][2][0], newColData[colN][2][1], newColData[colN][3][0][-1]), (newColData[colN][2][0], newColData[colN][2][1], newColData[colN][3][0][0])]
		side12A = area_3d(side12pts)

		side23pts = [(newColData[colN][2][0], newColData[colN][2][1], newColData[colN][3][0][0]), (newColData[colN][2][0], newColData[colN][2][1], newColData[colN][3][0][-1]), (newColData[colN][4][0], newColData[colN][4][1], newColData[colN][5][0][-1]), (newColData[colN][4][0], newColData[colN][4][1], newColData[colN][5][0][0])]
		side23A = area_3d(side23pts)

		side34pts = [(newColData[colN][4][0], newColData[colN][4][1], newColData[colN][5][0][0]), (newColData[colN][4][0], newColData[colN][4][1], newColData[colN][5][0][-1]), (newColData[colN][6][0], newColData[colN][6][1], newColData[colN][7][0][-1]), (newColData[colN][6][0], newColData[colN][6][1], newColData[colN][7][0][0])]
		side34A = area_3d(side34pts)

		side14pts = [(newColData[colN][6][0], newColData[colN][6][1], newColData[colN][7][0][0]), (newColData[colN][6][0], newColData[colN][6][1], newColData[colN][7][0][-1]), (newColData[colN][0][0], newColData[colN][0][1], newColData[colN][1][0][-1]), (newColData[colN][0][0], newColData[colN][0][1], newColData[colN][1][0][0])]
		side14A = area_3d(side14pts)

		colInfo_area[colN] = [baseA, side12A, side23A, side34A, side14A]

		# individual volumes and their unit weights
		tempVol = []
		tempUnitWeight = []

		centralZLists = newColData[colN][9][0][:]
		centralTYPElists = newColData[colN][9][1][:]
		if 'gw' in centralTYPElists:
			gwIDX = centralTYPElists.index('gw')
			gwCentZ = centralZLists[gwIDX]

			if gwCentZ < centralZLists[0]:
				centralZLists.pop(gwIDX)
				centralTYPElists.pop(gwIDX)
		
		numDiffvol = len(centralTYPElists)-2

		for loopVol in range(numDiffvol):
			
			#print(loopVol)
			#print(newColData[colN][9][0])
			#print(newColData[colN][9][1])
			planArea = abs(newColData[colN][0][0]-newColData[colN][4][0])*abs(newColData[colN][0][1]-newColData[colN][4][1])
			dZ = abs(centralZLists[loopVol+1] - centralZLists[loopVol])
			#print(colN, loopVol, planArea, dZ)
			tempV = planArea*dZ
			'''
			base4Pts = []
			top4Pts = []
			for loopVol2 in [0, 2, 4, 6]:
				base4Pts.append([newColData[colN][loopVol2][0], newColData[colN][loopVol2][1], newColData[colN][loopVol2+1][0][loopVol]])
				top4Pts.append([newColData[colN][loopVol2][0], newColData[colN][loopVol2][1], newColData[colN][loopVol2+1][0][loopVol+1]])
			tempV = vol_box_from_tetras(base4Pts, top4Pts)
			'''
			tempVol.append(tempV)

			tempVType = [centralTYPElists[loopVol], centralTYPElists[loopVol+1]]

			if tempVType[1] == 'gw':
				for loopVtype in range(loopVol+1, len(centralTYPElists)):
					if centralTYPElists[loopVtype][0] in ['m']:
						tempUnitWeight.append(materialClass[centralTYPElists[loopVtype]][3])
					else:
						continue
			#elif tempType[0][0] in ['r','w']:
			#	tempAreaType.append(tempType[0])
			else:
				for loopVtype in range(loopVol+1, len(centralTYPElists)):
					if centralTYPElists[loopVtype][0] in ['m']:
						tempUnitWeight.append(materialClass[centralTYPElists[loopVtype]][2])
					else:
						continue

		#print(tempUnitWeight)
		#print(tempVol)
		tempW = 0
		tempVtotal = 0
		for loopW in range(len(tempVol)):
			tempVtotal += tempVol[loopW]
			tempW += tempVol[loopW]*tempUnitWeight[loopW]

		tempVol.append(tempVtotal)

		colInfo_vol[colN] = tempVol
		colInfo_W[colN] = tempW
	
	#print(colInfo_vol)
	#print(colInfo_W)

	# pore-water pressure 
	# output - [u_b_net]
	#print(newColData)
	colInfo_GW = GW_3D(newColData, water_unitWeight)
	#print(colInfo_GW)

	# external load - for future
	#colInfo_load = {}

	# support - for future
	#colInfo_supportT = {}

	# shear strength parameters
	colInfo_shear = {}
	for colN in colKeyList:

		# name of the material type
		# center
		if newColData[colN][9][1][0][0] in ['r','w']:
			materialName = newColData[colN][9][1][0]

		elif newColData[colN][9][1][0][0] in ['b','s']:
			if newColData[colN][9][1][1] not in ['gw']:
				materialName = newColData[colN][9][1][1]
			else:
				materialName = newColData[colN][9][1][2]

		# inputs
		pwp = 0
		if len(colInfo_GW.keys()) == 0:
			matricSuction = 0
			pwp = 0
		else:
			pwp = colInfo_GW[colN][0]
			if colInfo_GW[colN][0] < 0: 
				matricSuction = abs(colInfo_GW[colN][0])
			else:
				matricSuction = 0
		
		eff_normal_stress = (colInfo_W[colN]/colInfo_area[colN][0]) - max(pwp, 0)

		Z = newColData[colN][9][0][0]
		Ztop = newColData[colN][9][0][-1]

		phiC = shearModel2cphi(materialClass, materialName, Z, Ztop, eff_normal_stress, matricSuction)[0][0]
		colInfo_shear[colN] = [materialClass[materialName][0], [phiC[0], phiC[1]], materialClass[materialName][-1]]

	# compile into a new csv file for 3D analysis
	for colN in colKeyList: 
		compileList = np.zeros(30) 

		compileList[0] = colN

		compileList[1] = newColData[colN][8][0]	# X
		compileList[2] = newColData[colN][8][1]	# Y
		compileList[20] = newColData[colN][9][0][-1]	# Z top
		compileList[21] = newColData[colN][9][0][0]		# Z bottom

		if len(colInfo_GW.keys()) != 0:
			if colInfo_GW[colN][0] < 0:
				compileList[19] = abs(colInfo_GW[colN][0])	# matric suction
			elif colInfo_GW[colN][0] > 0:
				compileList[5] = colInfo_GW[colN][0]	# Initial Pore Pressure

		compileList[14] = colInfo_W[colN]		# W
		compileList[15] = colInfo_vol[colN][-1]		# volume total
		compileList[16] = compileList[14]/colInfo_area[colN][0]	 # W/baseA

		compileList[22] = colInfo_area[colN][1]	# area of side 1
		compileList[23] = colInfo_area[colN][2] # area of side 2
		compileList[24] = colInfo_area[colN][3] # area of side 3
		compileList[25] = colInfo_area[colN][4] # area of side 4

		compileList[17] = colInfo_dip_dipDirection[colN][0]	# dip
		compileList[18] = colInfo_dip_dipDirection[colN][1]	# dip direction

		compileList[26] = colInfo_shear[colN][0]		# shear failure model
		compileList[27] = colInfo_shear[colN][-1]		# material name
		compileList[13] = colInfo_shear[colN][1][0]		# phi'
		compileList[12] = colInfo_shear[colN][1][1]		# c'

		if tensionCrackAngle==None:
			compileList[29] = 1
		elif tensionCrackAngle!=None:
			if compileList[17] >= tensionCrackAngle:
				compileList[29] = 0
			else:
				compileList[29] = 1

		analysisInputFile.append(compileList)

	exportList2CSV('3DanalysisInputFile.csv', analysisInputFile)

	return analysisInputFile

def overall_3D_backend(DEMNameList, DEMTypeList, DEMInterpolTypeList, materialClass, water_unitWeight, canvasRange, colXmax, colYmax, SSTypeList, inputPara):
	colNedge = DEM_3D_columnEdge(DEMNameList, DEMTypeList, DEMInterpolTypeList, canvasRange, colXmax, colYmax, stdMax=150)
	colNedge1 = DEM_3D_columnCenter(colNedge, DEMNameList, DEMTypeList, DEMInterpolTypeList, canvasRange, colXmax, colYmax, stdMax=150)

	newColEdge,ss_csv = SS_3D_columnsEdge(SSTypeList, inputPara, canvasRange, colXmax, colYmax, colNedge1, stdMax=150)
	newColData,ss_csv = SS_3D_columnsCenter(SSTypeList, inputPara, canvasRange, colXmax, colYmax, newColEdge, stdMax=150)
	analysisInputFile = createInputfile4Analysis_3D_columns(newColData, materialClass, canvasRange, water_unitWeight, tensionCrackAngle=None)

	return newColData, ss_csv, analysisInputFile

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


''' 3D analysis '''
# Analysis_3D_HungrBishop1989_v4_06_28_2018.py
def analysis3DHungrBishop1989(fileName, seismicK, centerPt0, materialClass, iterationNMax=200, tolFS=0.0005, occuranceFactor=0.5, tolDirection_user=None, spacingDirection=0.5, avDipDirectionB_user=None):
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
			FSmDe = 0

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
				RmidZ = abs(centerPt0[2] - 0.5*(analysisInput[loopCol][20] + analysisInput[loopCol][21]))

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
				deltaRangleXY = round(90 - (np.degrees(dipSlidingDirection) + abs(np.degrees(np.arctan(RZ/(max([xi,0.01])))))),2)

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
				FSmDe += Wi*xi - Ni*fi*(cosBaseAngleGamma/np.cos(dipSlidingDirection)) + seismicKxy*Wi*RmidZ + Ei*E_d

			# computed FS	
			FS_c = FSmNu/FSmDe

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
def analysis3DHungrJanbu1989(fileName, seismicK, materialClass, correctFS=None, iterationNMax=200, tolFS=0.0005, occuranceFactor=0.5, tolDirection_user=None, spacingDirection=0.5, avDipDirectionB_user=None):
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

	directionOfSlidingBList =  making_float_list(avDipDirectionB-tolDirection, avDipDirectionB+tolDirection, spacingDirection)
	#directionOfSlidingPList =  making_float_list(avDipDirectionP-tolDirection, avDipDirectionP+tolDirection, spacingDirection)

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
		correctionFactor = 1 + b1*((d/L) - 1.4*((d/L)**2))
	else:
		correctionFactor = correctFS

	''' FS computation '''
	FS_results = []

	for dirLoop in range(len(directionOfSlidingBList)):

		iterationN = 1
		FS_i = 3			#inital guess of FS
		sumW = 0
		sumV = 0 
		
		# iterate through to find global 3D FS
		iterationFS = True
		while iterationFS:

			FSNu = 0
			FSDe = 0

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
				cosBaseAngleGamma = ((np.tan(baseAngleX))**2 + (np.tan(baseAngleY))**2 + 1)**(-0.5)

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
				ui = analysisInput[loopCol][5]							# pore-water pressure force
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
					newShear = shearModel2cphi(materialClass, analysisInput[loopCol][27], 0, 0, (Ni/baseA - ui), matricSuction)

					ci = newShear[0][1]
					phiRad = np.radians(newShear[0][0])

				# force equilibrium
				FSNu += (ci*baseA + (Ni - ui*baseA)*np.tan(phiRad))*np.cos(dipSlidingDirection)
				FSDe += Ni*cosBaseAngleGamma*np.tan(dipSlidingDirection) + seismicKxy*Wi + Ei
		
			# computed FS	
			FS_c = FSNu/FSDe

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
def analysis3DChengnYip2007_correctFSfactor(correctFS, analysisInput):
	import numpy as np 

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
		#xxx =  listAtColNum(analysisInput,1)
		#yyy =  listAtColNum(analysisInput,2)
		#print([xxx, yyy])

		Lx = abs(max( listAtColNum(analysisInput,1)) - min( listAtColNum(analysisInput,1)))
		Ly = abs(max( listAtColNum(analysisInput,2)) - min( listAtColNum(analysisInput,2)))
		Lxy = np.sqrt(Lx**2 + Ly**2)
		Llist = [Lx, Ly, Lxy]
		Lmin = min(Llist)
		Lindex = Llist.index(Lmin)
		Lz = abs(max( listAtColNum(analysisInput,20)) - min( listAtColNum(analysisInput,21)))
		LangleRad = np.arctan(Lz/Lmin)
		#LangleDeg = np.degrees(np.arctan(Lz/Lmin))
		L = np.sqrt(Lmin**2 + Lz**2)

		#print([Lx, Ly, Lxy, Lmin, Lz, LangleRad, LangleDeg, L])
		#print(b1)

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

	changeFS = 0.005
	#changeFS = 0.01
	changeLambda = 0.2

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
		#slidingDirectionP = directionOfSlidingPList[loopSDir]

		# intercolumn function
		if method == 4: # simplified Janbu method
			Fxy = 0		# intercoloum function
			lambdaXY = 0 	# horizontal direction intercoloum force scaling factor
			scaleLambdaXList = [0] # making_float_list(-1, 3, 0.05)
			scaleLambdaYList = [0] # making_float_list(-1, 3, 0.05)
			correctionFactor = analysis3DChengnYip2007_correctFSfactor(correctFS, analysisInput)
		elif method == 3: # simplified bishop method
			Fxy = 0		# intercoloum function
			lambdaXY = 0 	# horizontal direction intercoloum force scaling factor
			scaleLambdaXList = [0] # making_float_list(-1, 3, 0.05)
			scaleLambdaYList = [0] # making_float_list(-1, 3, 0.05)
		elif method == 2: # spencer method
			Fxy = 1		# intercoloum function
			lambdaXY = 0 	# horizontal direction intercoloum force scaling factor
			if lambdaIteration == None:
				scaleLambdaXList = [-1]
				scaleLambdaYList = [-1]
			elif lambdaIteration != None:
				scaleLambdaXList = [lambdaIteration]
				scaleLambdaYList = [lambdaIteration]
		elif method == 1:  # Morgenstern-Price method
			lambdaXY = 0 	# horizontal direction intercoloum force scaling factor
			if lambdaIteration == None:
				scaleLambdaXList = [-1]
				scaleLambdaYList = [-1]
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
				FSxDe = 0
				FSyNu = 0
				FSyDe = 0
				FSmxNu = 0
				FSmxDe = 0
				FSmyNu = 0
				FSmyDe = 0
			
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
						FSxDe += Ai*g1 - Qxi + seismicKx*Wi - dHx + dEx
						#print([Axi,  Axi*(f1-Bi*g1), Ai*g1 - Qxi + seismicKx*Wi - dHx + dEx])

						# Force equilibrium - Y direction
						Ayi = (Ci + (Ai - Ui)*np.tan(phiRad))/(1-(Bi*np.tan(phiRad)/FSy_i))
						FSyNu += Ayi*(f2-(Bi*g2))
						FSyDe += Ai*g2 - Qyi + seismicKy*Wi - dHy + dEy
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
						FSmxDe += Wi*(RY + seismicKy*RZ) + Ni_mx*(g2*RZ - g3*RY) + Qmx

						# Moment equilibrium - YY
						Kmyi = (Ci + (Ai - Ui)*np.tan(phiRad))/(1-(Bi*np.tan(phiRad)/FSmy_i))
						FSmyNu += Kmyi*(f1*RZ+f3*RX)
						FSmyDe += Wi*(RX + seismicKx*RZ) + Ni_my*(g2*RZ - g3*RX) + Qmy
						
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
					FSx = FSxNu/FSxDe
					FSy = FSyNu/FSyDe

				if method in [1,2,3]:
					FSmx = FSmxNu/FSmxDe
					FSmy = FSmyNu/FSmyDe

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

							if round(slidingDirection,2) in [90, 270]:
								tempFSforceList.append(FSy_f)
								tempFSmomentList.append(FSmx_f)

							elif round(slidingDirection,2) in [0, 180, 360]:
								tempFSforceList.append(FSx_f)
								tempFSmomentList.append(FSmy_f)
							
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

							if abs(tempFSforceList[-1]-tempFSmomentList[-1]) > tolFS:
								if scaleLambdaXList[-1] > scaleLambdaYList[-1]:
									scaleLambdaYList.append(scaleLambdaYList[-1]+changeLambda)
								else:
									scaleLambdaXList.append(scaleLambdaXList[-1]+changeLambda)

								#scaleLambdaXList, scaleLambdaYList = changeLambda3D(scaleLambdaXList, scaleLambdaYList, FS_results_Temp, tolFS)
							else:
								FS_results.append([method, iterationN_a, iterationN_FS, slidingDirection, lambdaX, lambdaY, lambdaXY, FSmx_f, FSmy_f, FSx_f, FSy_f, np.nan])
								iterationLambda = False
							
							iterationN_FS = 0
							counts = 0
							iterationFS = False

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
			print(FS_results)

			FSmx_f = FS_results[loopFS][7]
			FSmy_f = FS_results[loopFS][8]
			FSx_f = FS_results[loopFS][9]
			FSy_f = FS_results[loopFS][10]

			# FSs_x - FSs_y
			'''
			checkFSsM = 0 
			checkFSsF = 0 
			checkFSsFM = 0
			'''
			if round(slidingDirection,2) in [0, 90, 180, 270, 360]:
				checkFSsM = 0 
				checkFSsF = 0 
				checkFSsFM = 0
			elif round(FSmx_f, 2) == 0 or round(FSmy_f, 2) == 0 or round(FSx_f, 2) == 0 or round(FSy_f, 2) == 0:
				checkFSsM = 0 
				checkFSsF = 0 
				checkFSsFM = 0
			else:
				checkFSsM = abs(FSmx_f - FSmy_f)
				checkFSsF = abs(FSx_f - FSy_f)
				checkFSsFM = abs(0.5*(FSmx_f+FSmy_f) - 0.5*(FSx_f+FSy_f))

			#print('checkFSsM=%f'%checkFSsM)
			#print('checkFSsF=%f'%checkFSsF)
			#print('checkFSsFM=%f'%checkFSsFM)

			if (checkFSsM < tolFS) and (checkFSsF < tolFS) and checkFSsFM < tolFS:
				'''
				if round(slidingDirection,2) in [90, 270]:
					FS_results[loopFS][11] = (FSmx_f+FSx_f)/2

				elif round(slidingDirection,2) in [0, 180, 360]:
					FS_results[loopFS][11] = (FSmy_f+FSy_f)/2
				'''
				if round(slidingDirection,2) in [0, 90, 180, 270, 360]:
					FS_results[loopFS][11] = max([(FSmx_f+FSx_f)/2, (FSmy_f+FSy_f)/2])

				elif round(FSx_f, 2) == 0 and round(FSy_f, 2) != 0:
					if round(FSmx_f, 2) == 0 and round(FSmy_f, 2) != 0:
						FS_results[loopFS][11] = (FSy_f+FSmy_f)/2

					elif round(FSmx_f, 2) != 0 and round(FSmy_f, 2) == 0:
						FS_results[loopFS][11] = (FSy_f+FSmx_f)/2

				elif round(FSx_f, 2) != 0 and round(FSy_f, 2) == 0:
					if round(FSmx_f, 2) == 0 and round(FSmy_f, 2) != 0:
						FS_results[loopFS][11] = (FSx_f+FSmy_f)/2

					elif round(FSmx_f, 2) != 0 and round(FSmy_f, 2) == 0:
						FS_results[loopFS][11] = (FSx_f+FSmx_f)/2

				else:
					FS_results[loopFS][11] = (FSmx_f+FSx_f+FSmy_f+FSy_f)/4

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
def select3DMethod(fileName, method, seismicK, materialClass, centerPt0, useDirectionB=None, useriterationNMax=200, usertolFS=0.001):
	if method == 1:
		FS = analysis3DHungrBishop1989(fileName, seismicK, centerPt0, materialClass, iterationNMax=useriterationNMax, tolFS=usertolFS, occuranceFactor=0.5, tolDirection_user=None, spacingDirection=0.5, avDipDirectionB_user=useDirectionB)
	elif method == 2:
		FS = analysis3DHungrJanbu1989(fileName, seismicK, materialClass, correctFS=1, iterationNMax=useriterationNMax, tolFS=usertolFS, occuranceFactor=0.5, tolDirection_user=None, spacingDirection=0.5, avDipDirectionB_user=useDirectionB)
	elif method == 3:
		FS = analysis3DHungrJanbu1989(fileName, seismicK, materialClass, correctFS=None, iterationNMax=useriterationNMax, tolFS=usertolFS, occuranceFactor=0.5, tolDirection_user=None, spacingDirection=0.5, avDipDirectionB_user=useDirectionB)
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
	
	return FS#[0][-1]


# everything combined
def process3DDEM(DEMNameList, DEMTypeList, DEMInterpolTypeList, materialClass, water_unitWeight, canvasRange, colXmax, colYmax, SSTypeList, inputPara, inputFileName='3DanalysisInputFile.csv', method, seismicK, centerPt0, useriterationNMax=100, useDirectionB=180):

	overall_3D_backend(DEMNameList, DEMTypeList, DEMInterpolTypeList, materialClass, water_unitWeight, canvasRange, colXmax, colYmax, SSTypeList, inputPara)
	FS3D = select3DMethod(fileName, method, seismicK, materialClass, centerPt0, useriterationNMax=100, useDirectionB=180)

	return FS2D


'''Output Check'''
import time
time_start = time.clock()

DEMNameList = ['gwLevel3D.csv','bedrockLayer3D.csv','weakLayer3D.csv','groundSurface3D.csv','groundSurface3D.csv']
#DEMNameList = ['weakLayer3D.csv','groundSurface3D.csv','groundSurface3D.csv']

DEMTypeList = ['gw','rr','w1','m1','tt'] 
#DEMTypeList = ['w1','m1','tt'] 

materialClass = {'rr':[1,[[45,10000],[0,0,0]],150,150,1],'w1':[1,[[10,0],[0,0,0]],120,120,2],'m1':[1,[[20,600],[0,0,0]],120,120,3]}
water_unitWeight = 62.4

#print(DEM_3D_column(['weak layer.csv','weak layer.csv'], ['w1','tt'], ['a1','a1'], [1795700, 1795965, 498790, 498800, 700, 760], 50, 50, stdMax=10))

#DEMInterpolTypeList = ['a1', 'a1', 'a1', 'a1', 'a1']
DEMInterpolTypeList = ['c1', 'c1', 'c1', 'c1', 'c1']
#DEMInterpolTypeList = ['b1', 'b1', 'b1', 'b1']
canvasRange = [0, 160, 0, 160, 0, 80] #[1795700, 1795965, 498790, 498800, 650, 1000] # min X, max X, min Y, max Y, min Z, max Z
colXmax = 75
colYmax = 75

SSTypeList = 2
inputPara = [80, 60, 90, 60, 80, 80]
#inputPara = [50, 60, 90, 59.5, 79.3, 79.3]
#inputPara = ['overall_ss3Dv2.csv','a1']

#fileName = 'ColumnResults_circular_Bishop.csv'				# 180
#fileName = 'ColumnResults_planar.csv'						# 180
#fileName = 'ColumnResults_plane_270.csv'					# 270
#fileName = 'yeager_noGW_noGeogrid.csv'						# 170.5
#fileName = 'ChengNYip2007_example2_ColumnResults.csv'		# 225
#fileName = 'ChengNYip2007_example3_ColumnResults.csv'		# 225
#fileName = 'ChengNYip2007_example4_ColumnResults.csv'		# 153.4
fileName = '3DanalysisInputFile.csv'						# 180
#fileName = 'slide3columns.csv'						# 180

materialClass = {'rr':[1,[[45,10000],[0,0,0]],150,150,1],'w1':[1,[[10,0],[0,0,0]],120,120,2],'m1':[1,[[20,600],[0,0,0]],120,120,3]}

seismicK = [0, 0]

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
method = 4

#centerPt0 = [5, 0, 5] 
#centerPt0 = [5, 0, 8.03973] 
#centerPt0 = [0, 5, 8.03973] 
#centerPt0 = [1795820, 498800, 1008.41]
#centerPt0 = [0, 0, 5]
#centerPt0 = [0, 0, 9.46981]
#centerPt0 = [4, 0, 6.53953]
centerPt0 = [80, 60, 90]  # [80, 50, 120] #

print (select3DMethod(fileName, method, seismicK, materialClass, centerPt0, useriterationNMax=100, useDirectionB=180))

#print(analysisInputFile)
newColData, ss_csv, analysisInputFile = overall_3D_backend(DEMNameList, DEMTypeList, DEMInterpolTypeList, materialClass, water_unitWeight, canvasRange, colXmax, colYmax, SSTypeList, inputPara)

time_elapsed = (time.clock() - time_start)
print(time_elapsed)  # tells us the computation time in seconds