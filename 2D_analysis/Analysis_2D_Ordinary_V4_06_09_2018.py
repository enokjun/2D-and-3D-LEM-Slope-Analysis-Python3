'''
Ordinary Method of Slices

purpose: compute 2D FS using Ordinary Method of Slices
Input: seismic load, water forces, weight, external load
Output: FS

description of the input file - subjected to change

## Input file column legend for row 0
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
10(K) - shear strength id (each id links to certain shear strength; e.g. 1 = Mohr-Coulomb)

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
10(K) - pore-water force at top (U_t)
11(L) - pore-water force at base (U_b)
12(M) - pore-water force at left (U_l)
13(N) - pore-water force at right (U_r)
14(O) - external load - line load (L)
15(P) - line load orientation from horizontal (omega)
16(Q) - perpendicular distance from the line load to the center of rotation (L-d)
17(R) - resultant maximum tensile force from support (T) 
18(S) - angle of support force from horizontal (i) 
19(T) - Soil Shear Strength force (Sm)
20(U) - Mohr-Coulomb shear strength - cohesion (c')
21(V) - Mohr-Coulomb shear strength - angle for friction (phi')
'''
'''
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
'''

'''main function - 2D slope stability analysis with Ordinary Method of Slices'''
def ordinary_method(filename):
	import math
	import making_list_with_floats as makelist # functions from making_lists_with_floats.py created by Enok

	# converting the input csv file into a list
	analysisInput = makelist.csv2list(filename)

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

'''Output Check'''
import time
time_start = time.clock()

print(ordinary_method('test inputs for analysis.csv'))

time_elapsed = (time.clock() - time_start)
print(time_elapsed)  # tells us the computation time in seconds
