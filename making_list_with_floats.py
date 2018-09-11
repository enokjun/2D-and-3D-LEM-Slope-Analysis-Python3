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

#example of exportList2CSV function input files
#csv_columns = ['Row','Name','Country']
#csv_data_list = [['1', 'Alex', 'India'], ['2', 'Ben', 'USA'], ['3', 'Shri Ram', 'India'], ['4', 'Smith', 'USA'], ['5', 'Yuva Raj', 'India'], ['6', 'Suresh', 'India']]
#csv_file = "test.csv"

#exportList2CSV('slide3columns.csv',csv2txtlistSlide3('ColumnResults.csv'))
