from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
let = LabelEncoder()
let.fit(xtt['大标签'])
list(let.classes_)
y_true_b=let.transform(xtt['大标签'])
y_pred_b=let.transform(xtt['big'])
print (classification_report(y_true_b,y_pred_b,target_names=list(let.classes_)))

#平均划分测试集，保证分布一致，‘y’的分布保证一致
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import numpy as np
X = np.array(df)
y = np.array(df["y"])
#分成2组，测试比例为0.25，训练比例是0.75
ss=StratifiedShuffleSplit(n_splits=2,test_size=0.01,train_size=0.99,random_state=0)

for train_index, test_index in ss.split(X, y):
    X_train, X_test = pd.DataFrame(X[train_index]), pd.DataFrame(X[test_index])
    
#另一种划分测试集的方式   
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df.drop('is_attributed', axis=1), 
                                                    df['is_attributed'], test_size=0.2)
