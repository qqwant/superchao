# fasttext
```python
import fastText as fasttext
Inputdata_Path = './train1600.txt'
#txt文件每行为分好词打好label
如： xx xx xx xx xx	__label__xx
test_path='./test1600.txt'

def build_classification(Inputdata_Path):
    print("开始训练模型：....")
    classifier = fasttext.train_supervised(Inputdata_Path,lr=0.1,bucket=500000,label='__label__')
    print("训练完成。")
    return classifier

#训练模型	
classf=build_classification(Inputdata_Path)

#验证模型
classf.test(test_path)

#预测	
texts = ['xx xx xxxx', 'xx xxx xx']
classf.predict(texts)

#保存加载
classf.save_model('Q_fasttext.model')

f=fasttext.load_model('Q_fasttext.model')

```
