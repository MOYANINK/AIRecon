# AIRecon
基于机器学习的恶意请求检测
选择使用了XGBoost算法进行恶意请求检测模型的训练，代替原先的逻辑回归算法，提升了检测恶意请求的准确率。
good_file = 'goodqueries2.txt'正常请求文件
bad_file = 'badqueries.txt'恶意请求文件
model_filename = 'xgboost_model.pkl' 保存模型地址
vectorizer_filename = 'tfidf_vectorizer.pkl' 保存向量地址
data.txt 检测日志文件地址
