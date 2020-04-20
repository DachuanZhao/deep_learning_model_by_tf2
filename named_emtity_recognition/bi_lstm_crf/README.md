# bi-lstm-crf tf2 ner模型
### bilstmcrf tensorflow2版
## 版本说明 
### 1. tensorflow >= 2.0.0

## 目录说明
### 1. ```./data/``` : 数据类文件保存路径
### 2. ```./model_callback/``` : checkpoint和tensorboard文件
### 3. ```word2vec``` : 预训练word2vec代码和结果文件


## 使用说明：
### 1. 首先```nohup python 2_main.py &```生成word2vec ，文件格式参见```./data/word2vec_input.txt```
### 2. 然后运行```nohup python train_bi_lstm_crf.py & ``` 训练模型，文件格式参见```./data/ner_input.txt``` ;在此期间可以 ```sh run_tensorboard.sh``` 在tensorboard查看训练结果
### 3. ```nohup python predict.py &``` 为预测，文件格式参见```./data/ner_predict.txt```
