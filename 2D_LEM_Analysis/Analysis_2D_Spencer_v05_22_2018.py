'''
Created on 05/21/2018
v1 Finished on 05/22/2018

@author: Enok C.

2D LEM Slope Analysis - Spencer (1967)
'''

'''
purpose: Compute 2D FS using Spencer's method (1967) 
Input: slice geometry, shear strength,  external load, seismic load, water, tension crack, center point, 
Output: FS, Theta (interslice force angle)

Spencer's 2D LEM slope stability analysis method (1967) assumes:
1. equilibrium in vertical, horizontal and moment
2. relationship between interslice shear and normal forces are constant
'''

'''Input'''
# import packages
'''
import math
import making_list_with_floats as makeList  # functions from making_list_with_floats.py
import numpy as np
from sympy import Symbol, solve
import matplotlib.pyplot as plt
from csv
import scipy as sp
'''

# import slice
'''
description of the input file - subjected to change

## Input file column legend for row 1 
0(A) - 2D or 3D analysis (2 = 2D; 3 = 3D)
1(B) - id of analysis (id links to a specific model)
2(C) - id of slip surface (id links to a particular slip surface)
3(D) - X coordinate of center of rotation (Pt0x)
4(E) - Y coordinate of center of rotation (Pt0y)
5(F) - seismic coefficient for horizontal force (k)
6(G) - external load - line load (L)
7(H) - line load orientation from horizontal (omega)
8(I) - perpendicular distance from the line load to the center of rotation (L-d)
9(J) - resultant water forces - from tension crack or submersion (A)
10(K) - perpendicular distance from the resultant water force to the center of rotation (a)
11(L) - shear strength id (each id links to certain shear strength; e.g. 1 = Mohr-Coulomb)

## Input file column legend for row 2 and below
0(A) - slice number
1(B) - slice horizontal width (b)
2(C) - slice total area (A)
3(D) - slice base angle from horizontal (alpha)
4(E) - radius (R)
5(F) - perpendicular offset from center of rotation(f)
6(G) - horizontal distance from slice to the center of rotation (x)
7(H) - vertical distance from C.G. of slice to the center of rotation (e)
8(I) - water head (hw)
9(J) - slice weight (W) 
10(K) - tensile force from support (T) 
11(L) - Mohr-Coulomb shear strength - cohesion (c')
12(M) - Mohr-Coulomb shear strength - angle for friction (phi')
'''

'''check input file'''
def checkAnalysis2DSpencer(analysisInput, requiredInfoNum, requiredSliceNum):
	# check1: check that input is all number
	for row in range(len(analysisInput)):
		if all([str(x).isnumeric() for x in analysisInput[row]]):
			raise(NameError('InputError: check that inputs are all number - check the code'))
			return None

	# check2: check first row has all the required info
	if len(analysisInput[0]) != requiredInfoNum:
		raise(NameError('InputError: insufficient information provided - check the code'))
		return None

	# check3: row and col match for slice info section
	for row in range(1,len(analysisInput)):
		if len(analysisInput[row]) != requiredSliceNum:
			raise(NameError('InputError: insufficient information of slice on row %i' %row))
			return None

'''change the interslice angle theta based on the difference of FS'''
def changeIntersliceTheta_Tol_1e_3(thetaInter, FS_moment_f, FS_force_f):
	# change the interslice force angle
	if FS_moment_f-FS_force_f > 0:
		if abs(FS_moment_f-FS_force_f) >= 0.1:
			thetaInter += 10
		elif abs(FS_moment_f-FS_force_f) >= 0.01 and abs(FS_moment_f-FS_force_f) < 0.1:
			thetaInter += 5	
		elif abs(FS_moment_f-FS_force_f) >= 0.005 and abs(FS_moment_f-FS_force_f) < 0.01:
			thetaInter += 0.5
		elif abs(FS_moment_f-FS_force_f) > 0.0001 and abs(FS_moment_f-FS_force_f) < 0.005:
			thetaInter += 0.05

	elif FS_moment_f-FS_force_f < 0:
		if abs(FS_moment_f-FS_force_f) >= 0.1:
			thetaInter -= 10
		elif abs(FS_moment_f-FS_force_f) >= 0.01 and abs(FS_moment_f-FS_force_f) < 0.1:
			thetaInter -= 5
		elif abs(FS_moment_f-FS_force_f) >= 0.005 and abs(FS_moment_f-FS_force_f) < 0.01:
			thetaInter -= 0.5
		elif abs(FS_moment_f-FS_force_f) > 0.0001 and abs(FS_moment_f-FS_force_f) < 0.005:
			thetaInter -= 0.05	

	return thetaInter

'''main function - Spencer Method 2D LEM slope stability analysis'''
def analysis2DSpencer(inputFileName, waterUnitWeight, tolaranceFS, requiredInfoNum, requiredSliceNum):
	# import function from Math Library
	#import numpy as np
	import math
	import making_list_with_floats as makeList  # functions from making_list_with_floats.py
 
	# take the inputfile and convert it into list
	analysisInput = makeList.csv2list(inputFileName)

	# check for input file errors
	checkAnalysis2DSpencer(analysisInput, requiredInfoNum, requiredSliceNum)

	# cut into separate files
	analysisInfo = analysisInput[0]
	sliceInfo = analysisInput[1:]
	
	# trial interslice angle (theta)
	thetaInter = 0
	iterationNN = 1
	iteration1 = True

	while iteration1:

		FS_force_i = 1		# inital trial value of FS for force equilibrium
		FS_moment_i = 1		# inital trial value of FS for moment equilibrium
		iterationN = 1
		dE_list = []
		iteration2 = True
		
		while iteration2:
			
			# FS for force calculated
			FS_force = 0
			FS_f_nom = 0
			FS_f_de = analysisInfo[9] - analysisInfo[6]*math.cos(math.radians(analysisInfo[7]))

			# FS for moment calculated
			FS_moment = 0
			FS_m_nom = 0
			FS_m_de = analysisInfo[9]*analysisInfo[10] + analysisInfo[6]*analysisInfo[8]

			# iterate trough slice
			numRow = len(sliceInfo)
			for loopSlice in range(numRow):			
				# interslice assumption for first analysis
				if iterationN == 1:
					dX_f = 0
					dX_m = 0	   # change in vertical interslice force (dX = X_L-X_R)
				# using FS from previous calculation dX is calculated
				else:
					dX_f = math.tan(math.radians(thetaInter))*dE_list[loopSlice][0]
					dX_m = math.tan(math.radians(thetaInter))*dE_list[loopSlice][1]

				# pore-water pressure and base slice length
				u_pwp = waterUnitWeight*sliceInfo[loopSlice][8]
				base_len = sliceInfo[loopSlice][1]/math.cos(math.radians(sliceInfo[loopSlice][3]))
				
				# calculate normal force (P) for force equilibrium
				ma_force = math.cos(math.radians(sliceInfo[loopSlice][3])) + math.sin(math.radians(sliceInfo[loopSlice][3]))*math.tan(math.radians(sliceInfo[loopSlice][12]))/FS_force_i
				P_force = (sliceInfo[loopSlice][9] - dX_f - (sliceInfo[loopSlice][11])*base_len*math.sin(math.radians(sliceInfo[loopSlice][3]))/FS_force_i + u_pwp*base_len*math.tan(math.radians(sliceInfo[loopSlice][12]))*math.sin(math.radians(sliceInfo[loopSlice][3]))/FS_force_i)/ma_force
				
				# calculate normal force (P) for moment equilibrium
				ma_moment = math.cos(math.radians(sliceInfo[loopSlice][3])) + math.sin(math.radians(sliceInfo[loopSlice][3]))*math.tan(math.radians(sliceInfo[loopSlice][12]))/FS_moment_i
				P_moment = (sliceInfo[loopSlice][9] - dX_m - (sliceInfo[loopSlice][11])*base_len*math.sin(math.radians(sliceInfo[loopSlice][3]))/FS_moment_i + u_pwp*base_len*math.tan(math.radians(sliceInfo[loopSlice][12]))*math.sin(math.radians(sliceInfo[loopSlice][3]))/FS_moment_i)/ma_moment
				
				# calcualte FS for force
				FS_f_nom += math.cos(math.radians(sliceInfo[loopSlice][3]))*(sliceInfo[loopSlice][11]*base_len + (P_force - u_pwp*base_len)*math.tan(math.radians(sliceInfo[loopSlice][12])))
				FS_f_de += P_force*math.sin(math.radians(sliceInfo[loopSlice][3])) + analysisInfo[5]*sliceInfo[loopSlice][9] 

				# calcualte FS for moment
				FS_m_nom += sliceInfo[loopSlice][4]*(sliceInfo[loopSlice][11]*base_len + (P_moment - u_pwp*base_len)*math.tan(math.radians(sliceInfo[loopSlice][12])))
				FS_m_de += sliceInfo[loopSlice][9]*sliceInfo[loopSlice][6] - P_moment*sliceInfo[loopSlice][5] + analysisInfo[5]*sliceInfo[loopSlice][9]*sliceInfo[loopSlice][7]

				# calculate dE for next iteration
				# dE = change in horizontal interslice force (dE = E_L-E_R)
				dE_f = P_force*math.sin(math.radians(sliceInfo[loopSlice][3])) - (math.cos(math.radians(sliceInfo[loopSlice][3]))/FS_force_i)*(sliceInfo[loopSlice][11]*base_len + (P_force - u_pwp*base_len)*math.tan(math.radians(sliceInfo[loopSlice][12])))
				dE_m = P_moment*math.sin(math.radians(sliceInfo[loopSlice][3])) - (math.cos(math.radians(sliceInfo[loopSlice][3]))/FS_moment_i)*(sliceInfo[loopSlice][11]*base_len + (P_moment - u_pwp*base_len)*math.tan(math.radians(sliceInfo[loopSlice][12])))

				if iterationN == 1:
					dE_list.append([dE_f, dE_m])
				else:
					dE_list[loopSlice] = [dE_f, dE_m]

			# calculated FS
			FS_force = FS_f_nom/FS_f_de
			FS_moment = FS_m_nom/FS_m_de
			
			'''
			print(iterationN)
			print(FS_force_i)
			print(FS_force)
			print(FS_moment_i)
			print(FS_moment)
			'''
			
			if abs(FS_force_i-FS_force) > tolaranceFS or abs(FS_moment_i-FS_moment) > tolaranceFS:
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
		
		print(iterationNN)
		print(iterationN)
		print(FS_force_f)
		print(FS_moment_f)
		print(FS_moment_f-FS_force_f)
		print(thetaInter)

		if abs(FS_moment_f-FS_force_f) > tolaranceFS:
			iterationNN += 1
			thetaInter = changeIntersliceTheta_Tol_1e_3(thetaInter, FS_moment_f, FS_force_f)
		else:
			FS_final = FS_force_f
			iteration1 = False
	
	results2DLEMSpencer = [analysisInfo[0:3], [FS_final, thetaInter]]

	return results2DLEMSpencer
	
'''Output Check'''
import time
time_start = time.clock()

waterUnitWeight = 62.4		# unit: pcf
inputFileName = 'test inputs for Spencer method.csv'	# test sample of csv file used for the analysis
requiredInfoNum = 12
requiredSliceNum = 13
tolaranceFS = 0.0001
FS = analysis2DSpencer(inputFileName, waterUnitWeight, tolaranceFS, requiredInfoNum, requiredSliceNum)
print(FS)

time_elapsed = (time.clock() - time_start)
print(time_elapsed)  # tells us the computation time in seconds
