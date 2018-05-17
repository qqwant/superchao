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



#制作fasttext格式要求的txt文件
def file_processing(file_input, file_output='./tmp.txt', func=lambda x: x.lower(), overwrite=True):
    """
    :param file_input:
    :param file_output:
    :param func: return str
    :param overwrite:
    :return:
    """
    import os
    from tqdm import tqdm
    assert isinstance(func('str'), str)

    if overwrite and os.path.isfile(file_output):
        os.remove(file_output)
    with open(file_input) as ifile:
        with open(file_output, 'a') as ofile:  # append
            for i in tqdm(ifile, desc='File Processing'):
                ofile.writelines(func(i))
    print('Before Processing:\n', os.popen('wc -l %s && head -n 5 %s' % (file_input, file_input)).read())
    print('After Processing:\n', os.popen('wc -l %s && head -n 5 %s' % (file_output, file_output)).read())
```
