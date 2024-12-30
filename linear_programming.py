import numpy as np
np.set_printoptions(threshold=np.inf)
import math as math
from scipy import linalg
from scipy.optimize import linprog
import random
from fractions import Fraction
from qpsolvers import solve_qp

def get_transition_value(x, y):
  small_matrix = np.array([
      [0, 1, 2],
      [3, 0, 4],
      [5, 6, 0]
  ])
  first_digit = small_matrix[x // 3, y // 3]
  second_digit = small_matrix[x % 3, y % 3]
  transition_value = f"{first_digit}{second_digit}"
  return transition_value

def weighted_average(values):
  vector1 = [4/7,2/7,1/7]
  a = []
  for i in range(len(vector1)):
      a.append(np.dot(cyclic_shift(vector1, i), values))
  med = np.median(a)
  for i in range(len(vector1)):
      if (a[i] == med):
          return cyclic_shift(vector1, i)

def get_column_number(transition):
  mapping = {
      # 10, 20, ..., 60 -> 00 (Columns 0-5)
      "10->00": 0, "20->00": 1, "30->00": 2, "40->00": 3, "50->00": 4, "60->00": 5,
      # 01, 02, ..., 06 -> 00 (Columns 6-11)
      "01->00": 6, "02->00": 7, "03->00": 8, "04->00": 9, "05->00": 10, "06->00": 11,
      # 11, 12, ..., 16 -> 00 (Columns 12-17)
      "11->00": 12, "12->00": 13, "13->00": 14, "14->00": 15, "15->00": 16, "16->00": 17,
      # 21, 22, ..., 26 -> 00 (Columns 18-23)
      "21->00": 18, "22->00": 19, "23->00": 20, "24->00": 21, "25->00": 22, "26->00": 23,
      # 31, 32, ..., 36 -> 00 (Columns 24-29)
      "31->00": 24, "32->00": 25, "33->00": 26, "34->00": 27, "35->00": 28, "36->00": 29,
      # 41, 42, ..., 46 -> 00 (Columns 30-35)
      "41->00": 30, "42->00": 31, "43->00": 32, "44->00": 33, "45->00": 34, "46->00": 35,
      # 51, 52, ..., 56 -> 00 (Columns 36-41)
      "51->00": 36, "52->00": 37, "53->00": 38, "54->00": 39, "55->00": 40, "56->00": 41,
      # 61, 62, ..., 66 -> 00 (Columns 42-47)
      "61->00": 42, "62->00": 43, "63->00": 44, "64->00": 45, "65->00": 46, "66->00": 47,
      # 11, 12, ..., 16 -> 10 (Columns 48-53)
      "11->10": 48, "12->10": 49, "13->10": 50, "14->10": 51, "15->10": 52, "16->10": 53,
      # 21, 22, ..., 26 -> 20 (Columns 54-59)
      "21->20": 54, "22->20": 55, "23->20": 56, "24->20": 57, "25->20": 58, "26->20": 59,
      # 31, 32, ..., 36 -> 30 (Columns 60-65)
      "31->30": 60, "32->30": 61, "33->30": 62, "34->30": 63, "35->30": 64, "36->30": 65,
      # 41, 42, ..., 46 -> 40 (Columns 66-71)
      "41->40": 66, "42->40": 67, "43->40": 68, "44->40": 69, "45->40": 70, "46->40": 71,
      # 51, 52, ..., 56 -> 50 (Columns 72-77)
      "51->50": 72, "52->50": 73, "53->50": 74, "54->50": 75, "55->50": 76, "56->50": 77,
      # 61, 62, ..., 66 -> 60 (Columns 78-83)
      "61->60": 78, "62->60": 79, "63->60": 80, "64->60": 81, "65->60": 82, "66->60": 83,
      # 11, 12, ..., 16 -> 01, 02, ..., 06 (Columns 84-89)
      "11->01": 84, "12->02": 85, "13->03": 86, "14->04": 87, "15->05": 88, "16->06": 89,
      # 21, 22, ..., 26 -> 01, 02, ..., 06 (Columns 90-95)
      "21->01": 90, "22->02": 91, "23->03": 92, "24->04": 93, "25->05": 94, "26->06": 95,
      # 31, 32, ..., 36 -> 01, 02, ..., 06 (Columns 96-101)
      "31->01": 96, "32->02": 97, "33->03": 98, "34->04": 99, "35->05": 100, "36->06": 101,
      # 41, 42, ..., 46 -> 01, 02, ..., 06 (Columns 102-107)
      "41->01": 102, "42->02": 103, "43->03": 104, "44->04": 105, "45->05": 106, "46->06": 107,
      # 51, 52, ..., 56 -> 01, 02, ..., 06 (Columns 108-113)
      "51->01": 108, "52->02": 109, "53->03": 110, "54->04": 111, "55->05": 112, "56->06": 113,
      # 61, 62, ..., 66 -> 01, 02, ..., 06 (Columns 114-119)
      "61->01": 114, "62->02": 115, "63->03": 116, "64->04": 117, "65->05": 118, "66->06": 119
  }

  return mapping.get(transition, None)
def get_transition_from_column_number(column_number):
  reverse_mapping = {
      # 10, 20, ..., 60 -> 00 (Columns 0-5)
      0: "10->00", 1: "20->00", 2: "30->00", 3: "40->00", 4: "50->00", 5: "60->00",
      # 01, 02, ..., 06 -> 00 (Columns 6-11)
      6: "01->00", 7: "02->00", 8: "03->00", 9: "04->00", 10: "05->00", 11: "06->00",
      # 11, 12, ..., 16 -> 00 (Columns 12-17)
      12: "11->00", 13: "12->00", 14: "13->00", 15: "14->00", 16: "15->00", 17: "16->00",
      # 21, 22, ..., 26 -> 00 (Columns 18-23)
      18: "21->00", 19: "22->00", 20: "23->00", 21: "24->00", 22: "25->00", 23: "26->00",
      # 31, 32, ..., 36 -> 00 (Columns 24-29)
      24: "31->00", 25: "32->00", 26: "33->00", 27: "34->00", 28: "35->00", 29: "36->00",
      # 41, 42, ..., 46 -> 00 (Columns 30-35)
      30: "41->00", 31: "42->00", 32: "43->00", 33: "44->00", 34: "45->00", 35: "46->00",
      # 51, 52, ..., 56 -> 00 (Columns 36-41)
      36: "51->00", 37: "52->00", 38: "53->00", 39: "54->00", 40: "55->00", 41: "56->00",
      # 61, 62, ..., 66 -> 00 (Columns 42-47)
      42: "61->00", 43: "62->00", 44: "63->00", 45: "64->00", 46: "65->00", 47: "66->00",
      # 11, 12, ..., 16 -> 10 (Columns 48-53)
      48: "11->10", 49: "12->10", 50: "13->10", 51: "14->10", 52: "15->10", 53: "16->10",
      # 21, 22, ..., 26 -> 20 (Columns 54-59)
      54: "21->20", 55: "22->20", 56: "23->20", 57: "24->20", 58: "25->20", 59: "26->20",
      # 31, 32, ..., 36 -> 30 (Columns 60-65)
      60: "31->30", 61: "32->30", 62: "33->30", 63: "34->30", 64: "35->30", 65: "36->30",
      # 41, 42, ..., 46 -> 40 (Columns 66-71)
      66: "41->40", 67: "42->40", 68: "43->40", 69: "44->40", 70: "45->40", 71: "46->40",
      # 51, 52, ..., 56 -> 50 (Columns 72-77)
      72: "51->50", 73: "52->50", 74: "53->50", 75: "54->50", 76: "55->50", 77: "56->50",
      # 61, 62, ..., 66 -> 60 (Columns 78-83)
      78: "61->60", 79: "62->60", 80: "63->60", 81: "64->60", 82: "65->60", 83: "66->60",
      # 11, 12, ..., 16 -> 01, 02, ..., 06 (Columns 84-89)
      84: "11->01", 85: "12->02", 86: "13->03", 87: "14->04", 88: "15->05", 89: "16->06",
      # 21, 22, ..., 26 -> 01, 02, ..., 06 (Columns 90-95)
      90: "21->01", 91: "22->02", 92: "23->03", 93: "24->04", 94: "25->05", 95: "26->06",
      # 31, 32, ..., 36 -> 01, 02, ..., 06 (Columns 96-101)
      96: "31->01", 97: "32->02", 98: "33->03", 99: "34->04", 100: "35->05", 101: "36->06",
      # 41, 42, ..., 46 -> 01, 02, ..., 06 (Columns 102-107)
      102: "41->01", 103: "42->02", 104: "43->03", 105: "44->04", 106: "45->05", 107: "46->06",
      # 51, 52, ..., 56 -> 01, 02, ..., 06 (Columns 108-113)
      108: "51->01", 109: "52->02", 110: "53->03", 111: "54->04", 112: "55->05", 113: "56->06",
      # 61, 62, ..., 66 -> 01, 02, ..., 06 (Columns 114-119)
      114: "61->01", 115: "62->02", 116: "63->03", 117: "64->04", 118: "65->05", 119: "66->06"
  }

  return reverse_mapping.get(column_number, None)

def cyclic_shift(arr, index):
  arr = np.array(arr)
  return np.roll(arr, -index).tolist()

def coeff_matrix(matrix, s, weighted_average_function,axis):
  rows, cols = matrix.shape
  x = rows*cols
  coeff_matrix = np.zeros((x,x),dtype=float) 
  if axis:
      for i in range(rows):
          y = weighted_average_function(matrix[i])
          for j in range(cols):
              index = i *cols+j
              for k in range(cols):
                  if index == i*cols+k:
                      coeff_matrix[index][i*cols+k] = (s-1)/s + y[k]/s
                  else:
                      coeff_matrix[index][i*cols+k] = y[k]/s
  else:
      for i in range(cols):
          y = weighted_average_function(matrix[:, i])
          for j in range(rows):
              index = j * cols + i
              for k in range(rows):
                  if j == k:
                      coeff_matrix[index][k*cols+i] = (s-1)/s + y[k]/s
                  else:
                      coeff_matrix[index][k*cols+i] = y[k]/s

  return coeff_matrix


def expfunction(matrix, s, weighted_average_function):
  X = coeff_matrix(matrix, s, weighted_average_function, True)
  Y = coeff_matrix(matrix, s, weighted_average_function, False)
  result = 1000000 * (linalg.logm(X) + linalg.logm(Y))
  exp_result = linalg.expm(result)
  exp_result1 = exp_result[0]
  return exp_result1.flatten().reshape(len(matrix), len(matrix[0]))


#TAU FUNCTIONS
def tau1d(values):
  vector1 = [4/7, 2/7, 1/7]
  a = [np.dot(np.roll(vector1, -i), values) for i in range(len(vector1))]
  return np.median(a)

def tau2d(x,bool):
  b = tau1d([tau1d(i) for i in (x if bool else np.transpose(x))])
  return (b)


def tau2(matrix,s,weighted_average_function):
  m = matrix.flatten()
  n = expfunction(matrix,s,weighted_average_function).flatten()
  x = np.dot(m,n)
  return (x)

def tau2dnew(matrix):
  a = [0,0,0]
  matrix10 = np.array([
      [2.0, 1.0, 0.0],
      [0.0, 2.0, 1.0],
      [1.0, 0.0, 2.0]
  ])
  matrix1000000 = np.array([
      [8.0, 4.0, 1.0],
      [1.0, 8.0, 4.0],
      [4.0, 1.0, 8.0]
  ])
  matrix1 = np.array([
      [2.0, 4.0, 1.0],
      [1.0, 0.0, 2.0],
      [0.0, 2.0, 1.0]
  ])
  matrix100000 = np.array([
      [4.0, 2.0, 1.0],
      [1.0, 4.0, 2.0],
      [2.0, 1.0, 4.0]
  ])
  matrix2 = np.roll(matrix1, shift=-1, axis=1)  
  matrix3 = np.roll(matrix1, shift=-2, axis=1)
  matrix4 = matrix.flatten()
  a[0] = np.dot(matrix1.flatten(),matrix4)
  a[1] = np.dot(matrix2.flatten(),matrix4)
  a[2] = np.dot(matrix3.flatten(),matrix4)
  return (np.median(a)/39)

#INTERTAU FUNCTIONS
def intertau(matrix,s,weighted_average_function):
  b = []
  a = expfunction(matrix, s, weighted_average_function)
  for i in range(len(matrix)):
      x = np.dot(matrix[i], a[i])/np.sum(a[i])
      b.append(float(x))

  for j in range(len(matrix[0])):
      col = [matrix[i][j] for i in range(len(matrix))]
      col1 = [a[i][j] for i in range(len(a))]
      x = np.dot(col, col1)/np.sum(col1)
      b.append(float(x))
  print((np.max(b) - np.min(b))/(np.max(matrix) - np.min(matrix)))
  return b

def intertau1(matrix):
  a = []
  for i in range(len(matrix)):
      a.append(tau1d(matrix[i]))
  for i in range(matrix.shape[1]): 
      a.append(tau1d(matrix[:, i]))

  return a

def intertau2(matrix):
  a = np.zeros((2, 6), dtype=float)  

  a[0][0] = tau1d(matrix[0])
  a[0][1] = tau1d(matrix[0])
  a[0][2] = tau1d(matrix[1])
  a[0][3] = tau1d(matrix[1])
  a[0][4] = tau1d(matrix[2])
  a[0][5] = tau1d(matrix[2])

  a[1][0] = tau1d(matrix[:, 0])
  a[1][1] = tau1d(matrix[:, 0])
  a[1][2] = tau1d(matrix[:, 1])
  a[1][3] = tau1d(matrix[:, 1])
  a[1][4] = tau1d(matrix[:, 2])
  a[1][5] = tau1d(matrix[:, 2])


  return a
#3 LINES MATRIX WORKING SOLUTON USING FRACTIONS
def easy_matrix_solns(r,s,t):
  def tau1d(values):
      vector1 = [4/7, 2/7, 1/7]
      a = [np.dot(np.roll(vector1, -i), values) for i in range(len(vector1))]
      return np.median(a)

  def equation1(a, b, r, s, t, x):
      return (2*r+a*s+4*b*t)/(2+a+4*b)-x

  def equation2(c, d, r, s, t, x):
      return (4*c*r+2*s+d*t)/(4*c+2+d)-x

  def equation3(e, f, r, s, t, x):
      return (e*r+4*f*s+2*t)/(e+4*f+2)-x

  def solns(r, s, t, x, eq_type):
      tolerance = 1e-6
      solutions = []

      if eq_type == 1:
          a_values = np.linspace(0, 1, 100)
          b_values = np.linspace(0, 1, 100)
          for a in a_values:
              for b in b_values:
                  if abs(equation1(a, b, r, s, t, x)) < tolerance:
                      solutions.append((a, b))

      elif eq_type == 2:
          c_values = np.linspace(0, 1, 100)
          d_values = np.linspace(0, 1, 100)
          for c in c_values:
              for d in d_values:
                  if abs(equation2(c, d, r, s, t, x)) < tolerance:
                      solutions.append((c, d))

      elif eq_type == 3:
          e_values = np.linspace(0, 1, 100)
          f_values = np.linspace(0, 1, 100)
          for e in e_values:
              for f in f_values:
                  if abs(equation3(e, f, r, s, t, x)) < tolerance:
                      solutions.append((e, f))

      if solutions:
          return solutions[0]
      else:
          print("No solutions found within the range.")
          return None

  a = [r, s, t]
  x = tau1d(a)

  results_map = {}

  # Solve for the equations and map to the specific ordering
  for i in range(1, 4):
      a = solns(r, s, t, x, i)
      if a is not None:
          if i == 1:
              results_map["30->00"] = a[0]
              results_map["50->00"] = a[1]
          elif i == 2:
              results_map["10->00"] = a[0]
              results_map["60->00"] = a[1]
          elif i == 3:
              results_map["20->00"] = a[0]
              results_map["40->00"] = a[1]

  for i in range(1, 4):
      a = solns(r, s, t, x, i)
      if a is not None:
          if i == 1:
              results_map["03->00"] = a[0]
              results_map["05->00"] = a[1]
          elif i == 2:
              results_map["01->00"] = a[0]
              results_map["06->00"] = a[1]
          elif i == 3:
              results_map["02->00"] = a[0]
              results_map["04->00"] = a[1]
  count = 0
  matrix1 = np.zeros((120,))
  for i in range(1, 7):
      matrix1[count] = (results_map.get(f"{i}0->00", 0))
      count+=1

  for i in range(1, 7):
      matrix1[count] = (results_map.get(f"0{i}->00", 0))
      count+=1
  for i in range(1, 7):
      for j in range(1, 7):
          p = results_map.get(f"{i}0->00", 0)
          q = results_map.get(f"0{j}->00", 0)
          matrix1[count] = (p*q)
          count+=1

  for i in range(1, 7):
      for j in range(1, 7):
          p = results_map.get(f"{i}0->00", 0)
          q = results_map.get(f"0{j}->00", 0)
          matrix1[count] = ((1-p)*q)
          count+=1

  for i in range(1, 7):
      for j in range(1, 7):
          p = results_map.get(f"{i}0->00", 0)
          q = results_map.get(f"0{j}->00", 0)
          matrix1[count] = ((1-q)*p)
          count+=1


  return(matrix1)


def transition_matrix(matrix):
  tau_2 = tau2d(matrix,True)
  #tau_2 = tau_2(matrix,2,weighted_average)
  #tau_2 = tau2dnew(matrix)
  matrix1 = np.array([
      [2.0, 4.0, 1.0],
      [1.0, 2.0, 4.0],
      [4.0, 1.0, 2.0]
  ])
  matrix2 = np.array([
      [2.0, 4.0, 1.0],
      [4.0, 1.0, 2.0],
      [1.0, 2.0, 4.0]
  ])
  matrix2 = matrix1.flatten()

  matrix3 = [4.0,1,0,1.0,4.0,4.0,1.0]
  nmatrix = matrix.flatten()
  coef_matrix = np.zeros((45,120),dtype=float) 

  beq = np.zeros((45,),dtype=float)
  intertau = intertau2(matrix)

  count = 0

  for i in range(1,7):
      #THE IF STATEMENTS BELOW ARE FOR THE 6 INTERTAU IN THE FORM (X,0)
      tau = intertau[0][(i-1)]
      for j in range(3):
          b1,b2,b3 = 0,0,0
          if (i == 1 or i == 2):
              b1 = nmatrix[0]
              b2 = nmatrix[1]
              b3 = nmatrix[2]
          if (i == 3 or i == 4):
              b1 = nmatrix[3]
              b2 = nmatrix[4]
              b3 = nmatrix[5]
          if (i == 5 or i == 6):
              b1 = nmatrix[6]
              b2 = nmatrix[7]
              b3 = nmatrix[8]
          if (j == 0):
              #i0 -> 00
              beq[count] = matrix1[0][j]*matrix3[i-1]*(tau-b1)
              coef_matrix[count][i-1] = matrix1[0][j]*matrix3[i-1]*(tau-b1)
              #i3 -> i0
              coef_matrix[count][50+6*i-6]=matrix1[1][j]*matrix3[i-1]*(b2-tau)
              #i5 ->  i0
              coef_matrix[count][52+6*i-6]=matrix1[2][j]*matrix3[i-1]*(b3-tau)
              count=count+1


          if (j == 1):
              #i1 -> i0
              coef_matrix[count][48+6*i-6] = matrix1[0][j]*matrix3[i-1]*(b1-tau)
              #i0 -> 00
              beq[count] = matrix1[1][j]*matrix3[i-1]*(tau-b2)
              coef_matrix[count][i-1] = matrix1[1][j]*matrix3[i-1]*(tau-b2)
              #i6 -> i0
              coef_matrix[count][53+6*i-6] = matrix1[2][j]*matrix3[i-1]*(b3-tau)
              count=count+1


          if (j == 2):
              #i2 -> i0
              coef_matrix[count][49+6*i-6] = matrix1[0][j]*matrix3[i-1]*(b1-tau)
              #i4 -> i0
              coef_matrix[count][51+6*i-6] = matrix1[1][j]*matrix3[i-1]*(b2-tau)
              #i0 -> 00
              beq[count] = matrix1[2][j]*matrix3[i-1]*(tau-b3)
              coef_matrix[count][i-1] = matrix1[2][j]*matrix3[i-1]*(tau-b3)
              count=count+1
  #THE NEXT 6 INTERTAU FOR (0,X)   
  for i in range(1,7):
      tau = intertau[1][(i-1)]
      for j in range(3):
          b1,b2,b3 = 0,0,0
          if (i == 1 or i == 2):
              b1 = nmatrix[0]
              b2 = nmatrix[3]
              b3 = nmatrix[6]
          if (i == 3 or i == 4):
              b1 = nmatrix[1]
              b2 = nmatrix[4]
              b3 = nmatrix[7]
          if (i == 5 or i == 6):
              b1 = nmatrix[2]
              b2 = nmatrix[5]
              b3 = nmatrix[8]

          if (j == 0):
              #3i -> 0i
              beq[count] = 2*matrix3[i-1]*(tau-b1)
              coef_matrix[count][5+i] = 2*matrix3[i-1]*(tau-b1)
              #0i -> 00
              coef_matrix[count][95+i] = 1*matrix3[i-1]*(b2-tau)
              #5i -> 0i
              coef_matrix[count][107+i] = 4*matrix3[i-1]*(b3-tau)
              count=count+1
          if (j == 1):
              #1i -> 0i
              coef_matrix[count][83+i] = 4*matrix3[i-1]*(b1-tau)
              beq[count] = 2*matrix3[i-1]*(tau-b2)
              #0i -> 00
              coef_matrix[count][5+i] = 2*matrix3[i-1]*(tau-b2)
              #6i -> 0i
              coef_matrix[count][113+i] = 1*matrix3[i-1]*(b3-tau)
              count=count+1
          if (j == 2):
              #2i -> 0i
              coef_matrix[count][89+i] = 1*matrix3[i-1]*(b1-tau)
              #4i -> 0i
              coef_matrix[count][101+i] = 4*matrix3[i-1]*(b2-tau)
              beq[count] = 2*matrix3[i-1]*(tau-b3)
              #0i -> 00
              coef_matrix[count][5+i] = 2*matrix3[i-1]*(tau-b3)
              count=count+1

  #THE 9 COLUMN EQUATIONS
  nested_matrix = np.kron(matrix1,matrix1)
  for i in range(9):
      for j in range(9):
          if (i == j):
              #make sure order for tau-nmatrix is correct and it isn't the other way. 
              #i changed it based on previous changes
              beq[count] = nested_matrix[i][j]*(tau_2-nmatrix[j])
          else:
              a = get_transition_value(j,i)
              a = a+"->00"



              b = get_column_number(a)
              coef_matrix[count][b] = nested_matrix[j][i]*(nmatrix[j] - tau_2)
      count=count+1

  bub = np.ones((36,),dtype=float)
  aub = np.zeros((36,120), dtype=float)
  count1 = 0
  for i in range(1,7):
      for j in range(1,7):
          a = str(i) + str(j) + "->" + "0" + str(j)
          b = str(i) + str(j) + "->" + str(i) + "0"
          c = str(i) + str(j) + "->00"
          one = get_column_number(a)
          two = get_column_number(b)
          three = get_column_number(c)
          aub[count1][one] = 1
          aub[count1][two] = 1
          aub[count1][three] = 1
          count1=count1+1
  c = np.ones((120), dtype=float) 
  bound1 = [0, 1+10**(-6)]
  res = linprog(c,A_ub=aub,b_ub=bub, A_eq=coef_matrix, b_eq=beq, bounds=bound1)
  return res.x







def tau2(matrix,s,weighted_average_function):
  m = matrix.flatten()
  n = expfunction(matrix,s,weighted_average_function).flatten()
  x = np.dot(m,n)
  return (x)




matrix_3 = np.array([
  [6.0, 0.0, 0.0],
  [0.0, 1.0, 0.0],
  [0.0, 0.0, 0.0]
])
print(tau2d(matrix_3,True))
#print(tau2(matrix_3,1.1,weighted_average))
for i in range(-50,10):
    matrix_3[0][0] = 0.025*i
    #print("(", str(matrix_3[0][0]), ",", str(tau2(matrix_3,1.1,weighted_average)), ")") 
    #print(str(matrix_3[0][0]))
    #print(str(tau2d(matrix_3,True)), ",") 
#print(transition_matrix(matrix_3))
#count = 0
#for i in range(500):
#  a = random_matrix = np.random.rand(3, 3)
#  if(transition_matrix(a)):
#      count+=1
#print(count/500)
#for i in range(120):
#    print(get_transition_from_column_number(i) + " " +str(b[i]))
#a = np. random. randint(size=(3,3))
#print(a)
#for i in range(120):
#  transition_key = get_transition_from_column_number(i)  # e.g., "10->00"
#  transitions_map[transition_key] = b[i]
#a = [4,1,1,4,4,1]
#for i in range(1,7):
#   print(a[i-1]*2*(1-transitions_map[f"{i}0->00"])


#random_matrix = np.random.rand(3, 3)
#print(random_matrix)

#print(easy_matrix_solns(1,0,0))
#print(transition_matrix(matrix_3) @ easy_matrix_solns(1,0,0))




# col1-6: 10, 20, ..., 60 → 00

# col7-12: 01, 02, ..., 06 → 00

# col13-18: 11, 12, ..., 16 → 00
# col19-24: 21, 22, ..., 26 → 00
# (Skipping 3, 4, 5 series)
# col43-48: 61, 62, ..., 66 → 00

# col49-54: 11, 12,13 ..., 16 → 10
# col55-60: 21, 22,23 ..., 26 → 20
# (Skipping 3, 4, 5 series)
# col70-84: 61, 62, ..., 66 → 60

# col85-90: 11, 12, ..., 16 → 01, 02, ..., 06
# col91-96: 21, 22, ..., 26 → 01, 02, ..., 06
# (Skipping 3, 4, 5 series)
# col115-120: 61, 62, ..., 66 → 01, 02, ..., 06