from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


#病人样本：性别，年龄大于50，吸烟，有无子女,是否运动  类别：肺癌

data = [[1,1,1,0,1],[1,1,1,1,1],[0,1,1,0,1],[0,0,0,1,0],[1,0,0,1,1],[0,0,0,1,1]]
label = [1,1,1,0,0,0]

std = StandardScaler()

x_train = std.fit_transform(data)


# 逻辑回归预测
lg = LogisticRegression(C=1.0)

lg.fit(x_train, label)


test_data = [[0,1,1,0,0],[1,0,0,1,1],[0,1,1,1,1]]

y_predict = lg.predict(test_data)

test_label =[1,0,1]
print("预测结果：",y_predict)

right_number = 0
for i in range(len(test_label)):
    if test_label[i]==y_predict[i]:
        right_number+=1
total_number = len(test_label)
print('准确率: {:.2%}'.format(right_number/total_number))
