import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression
class Newton(object):
	def __init__(self,epochs=50):
		self.W = None
		self.epochs = epochs
		
		
	def get_loss(self, X, y, W,b):
		"""
		计算损失
		input: X(2 dim np.array):特征
				y(1 dim np.array):标签
				W(2 dim np.array):线性回归模型权重矩阵
		output：损失函数值
		"""
		#print(np.dot(X,W))
		loss = 0.5*np.sum((y - np.dot(X,W)-b)**2)
		return loss
		
	def first_derivative(self,X,y):
		"""
		计算一阶导数g = (y_pred - y)*x
		input: X(2 dim np.array):特征
				y(1 dim np.array):标签
				W(2 dim np.array):线性回归模型权重矩阵
		output：损失函数值
		"""
		y_pred = np.dot(X,self.W) + self.b
		g = np.dot(X.T, np.array(y_pred - y))
		g_b = np.mean(y_pred-y)
		return g,g_b
		 
	def second_derivative(self,X,y):
		"""
		计算二阶导数 Hij = sum(X[i]*X[j])
		input: X(2 dim np.array):特征
				y(1 dim np.array):标签
		output：损失函数值
		"""
		H = np.zeros(shape=(X.shape[1],X.shape[1]))
		# for i in range(X.shape[1]):
			# for j in range(X.shape[1]):
				# H[i][j] = np.sum(X.T[i]*X.T[j])
		H = np.dot(X.T, X)
		H_b = 1
		return H, H_b
		
	def fit(self, X, y):
		"""
		线性回归 y = WX + b拟合，牛顿法求解
		input: X(2 dim np.array):特征
				y(1 dim np.array):标签
		output：拟合的线性回归
		"""
		np.random.seed(10)
		# W(2 dim np.array):线性回归模型权重矩阵
		self.W = np.random.normal(size=(X.shape[1]))
		self.b = 0
		for epoch in range(self.epochs):
			g,g_b = self.first_derivative(X,y)  # 一阶导数
			H,H_b = self.second_derivative(X,y)  # 二阶导数
			self.W = self.W - np.dot(np.linalg.pinv(H),g)
			self.b = self.b - 1/H_b*g_b
			print("itration:{} ".format(epoch), "loss:{:.6f}".format(
			self.get_loss(X, y , self.W,self.b)))
		
	def predict(self, X, y):
		y_pred = np.dot(X,self.W)+self.b
		loss = self.get_loss(X,y,self.W,self.b)
		print("predict_loss", loss)
		return y_pred
		
def normalize(x):
    return (x - np.min(x))/(np.max(x) - np.min(x))
if __name__ == "__main__":
	# input datasets 
	X, y = datasets.load_boston(return_X_y=True)
	np.random.seed(2)
	X = np.random.rand(100,5)
	y = np.sum(X**3 + X**2,axis=1)
	print(X.shape, y.shape)
	# 归一化
	X_norm = normalize(X)
	X_train = X_norm[:int(len(X_norm)*0.8)]
	X_test = X_norm[int(len(X_norm)*0.8):]
	y_train = y[:int(len(X_norm)*0.8)]
	y_test = y[int(len(X_norm)*0.8):]

	# model 1
	newton=Newton()
	newton.fit(X_train, y_train)
	newton.predict(X_test,y_test)
	
	reg = LinearRegression().fit(X_train, y_train)
	y_pred = reg.predict(X_test)
	print(0.5*np.sum((y_test - y_pred)**2))
		
