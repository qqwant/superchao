from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
let = LabelEncoder()
let.fit(xtt['大标签'])
list(let.classes_)
y_true_b=let.transform(xtt['大标签'])
y_pred_b=let.transform(xtt['big'])
print (classification_report(y_true_b,y_pred_b,target_names=list(let.classes_)))
