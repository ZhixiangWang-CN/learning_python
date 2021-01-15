#! coding:utf8
import sys
import math
import numpy as np

def sigmoid(x, w, b):
  return 1.0/(1.0+np.e**-(np.dot(x, w.T)+b))

# output: mx1
def p_y_equal_1(x, w, b):
  e = math.e**(np.dot(x, np.transpose(w))+b)
  return e/(1.0+e)

def log_likelihood_first_second_derivatives(w, b, x, y):
  p1 = p_y_equal_1(x, w, b)
  y_minus_p1 = y-p1
  # print("p1:{} y_minus_p1:{}".format(p1, y_minus_p1))
  ## [1 x n, scalar]
  first_derivative = [-np.sum(y_minus_p1*x, axis=0), -np.sum(y_minus_p1)]
  # print("first_derivative:{}".format(first_derivative))
  # print((np.dot(x,x.T)*p1*(1.0-p1)).shape)
  ## [n x n, scalar]
  ### 高维下的二阶矩阵就是海森矩阵，维度是nxn的
  second_derivative = [np.dot(x.T, x*p1*(1.0-p1)), np.sum(p1*(1.0-p1), axis=0)]
  return first_derivative, second_derivative

def do_predict(x, y, w, b):
  return np.array([1 if x>=0.5 else 0 for x in sigmoid(x,w,b)]).reshape(y.shape)

def cal_accuracy(predictions, true_ys):
  correct = np.equal(true_ys, predictions)
  accuracy = np.sum([1 if x else 0 for x in correct])*1.0/len(correct)
  return accuracy

if __name__ == "__main__":
  ## load data from csv
  data = np.loadtxt("water_melon.csv", dtype=float, delimiter=",")
  x = data[1:,1:-1]
  y = data[1:,-1]
  y = y.reshape([y.shape[0],1])
  print("x.shape:{} y.shape:{}".format(x.shape, y.shape))

  ## initialize w and b
  w = np.random.normal(0, 0.1, [1, x.shape[1]])
  b = np.random.normal(0, 0.1)
  print("initialize w:{} b:{}".format(w, b))

  ## update w and b
  delta = lambda x1,x2:np.linalg.norm(x1-x2) 
  epsilon = 1e-10
  epoch = 0
  while True:
    first_derivative, second_derivative = log_likelihood_first_second_derivatives(w, b, x, y)
    tmp_w, tmp_b = w, b
    # w, b = [tmp_w, tmp_b] - first_derivative/second_derivative
    ### 高维下，first_derivative/second_derivative = second_derivative的逆 dot first_derivative
    w = w - np.dot(np.linalg.inv(second_derivative[0]), first_derivative[0])
    b = b - first_derivative[1]/second_derivative[1]
    if epoch%100==0:
      error = delta(w, tmp_w)
      predictions = do_predict(x, y, w, b)
      accuracy = cal_accuracy(predictions, y)
      print("epoch:{} error:{} accuracy:{}".format(epoch, error, accuracy))
    epoch += 1
    if error < epsilon:
      break
  print("Training End. w:{} b:{}".format(w, b))

  predictions = do_predict(x, y, w, b)
  # print("Predictions:{}".format(predictions))
  accuracy = cal_accuracy(predictions, y)
  print("accuracy:{}".format(accuracy))
