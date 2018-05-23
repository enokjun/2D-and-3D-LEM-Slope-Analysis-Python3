# -*- coding: utf-8 -*-
"""
Created on Tue May  1 2018

Completed on Tue May  1 2018

@author: Enok C.

making list with floats
"""
'''
def txt2numInList(listName):
	totalRow = len(listName)
'''

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

def arrayAtColNum(arrayName,colNum):
	import numpy as np
	result = []
	for i in range(len(arrayName)):
		result.append(float(arrayName[i][colNum]))
	result = np.array(result)
	return result