import os
import urllib
import pickle
import warnings
import re
import random
import numpy as np
import xgboost as xgb
from datetime import datetime
from collections import defaultdict
from imblearn.over_sampling import SMOTE
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import urllib.parse
import requests
import os
import requests
import matplotlib.pyplot as plt
import geopandas as gpd
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score  # 添加在文件顶部
# 在代码开头添加
import matplotlib
matplotlib.use("Agg")  # 无GUI环境下必加
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve
from re import search
from collections import defaultdict

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# ANSI 颜色代码
RED = '\033[31m'
GREEN = '\033[32m'
RESET = '\033[0m'

# 高风险路径
HIGH_RISK_PATHS = [
    '/env', '/cf_scripts', '/.env', '/config', '/admin', '/login', '/wp-admin', '/manager', '/backup',
    '/localstart.aspx', '/+CSCOE+/logon.html', '/inicio.cgi', '/cgi-bin/info.cgi', '/admin.php', '/admin.jsp',
    '/magento_version', '/xml/info.xml', '/main.shtml', '/home.html', '/geoserver/web/', '/form.html', '/upl.php',
    '/boaform/admin/formLogin', '/manage/account/login', '/cgi-bin/login.cgi', '/login.jsp','/password.php'
]
# 定义 URL 白名单
ALLOWED_URLS = [
    '/',
    '/index.html',
    '/home.html',
    '/robots.txt'
]

def is_high_risk_url(url):
    url = url.lower()  # 统一转换为小写
    return any(risk_path in url for risk_path in HIGH_RISK_PATHS)

CMS_PATHS = ['/magento_version', '/wp-admin/setup-config.php', '/joomla/configuration.php']


# 加载模型和向量化器
def load_model_and_vectorizer(model_file, vectorizer_file):
    with open(model_file, 'rb') as mf, open(vectorizer_file, 'rb') as vf:
        model = pickle.load(mf)
        vectorizer = pickle.load(vf)
    return model, vectorizer


# 解析日志文件，提取时间、IP、请求方法和 URL
def parse_log(log_file):
    log_entries = []
    log_pattern = re.compile(
        r'(?P<ip>\d+\.\d+\.\d+\.\d+) - - \[(?P<time>[^\]]+)\] "(?P<method>GET|POST|PUT|DELETE|HEAD|OPTIONS) (?P<url>[^ ]+) HTTP/\d\.\d"'
    )

    # 检查文件是否存在
    if not os.path.exists(log_file):
        raise FileNotFoundError(f"日志文件未找到: {log_file}")

    print(f"日志文件路径: {os.path.abspath(log_file)}")  # 调试日志路径

    try:
        with open(log_file, 'r', encoding='utf-8') as file:
            for line in file:
                match = log_pattern.search(line)
                if match:
                    ip = match.group('ip')
                    try:
                        # 解析时间，处理时区信息
                        timestamp = datetime.strptime(match.group('time'), '%d/%b/%Y:%H:%M:%S %z')
                    except ValueError:
                        print(f"时间解析失败: {match.group('time')}")  # 调试时间格式
                        continue
                    url = match.group('url')
                    method = match.group('method')  # 提取请求方法
                    log_entries.append((ip, timestamp, url, method))  # 添加请求方法
                else:
                    print(f"解析失败: {line.strip()}")  # 调试日志格式
    except UnicodeDecodeError:
        print("utf-8 解析失败，尝试 ISO-8859-1 编码")
        with open(log_file, 'r', encoding='ISO-8859-1') as file:
            for line in file:
                match = log_pattern.search(line)
                if match:
                    ip = match.group('ip')
                    try:
                        # 解析时间，处理时区信息
                        timestamp = datetime.strptime(match.group('time'), '%d/%b/%Y:%H:%M:%S %z')
                    except ValueError:
                        print(f"时间解析失败: {match.group('time')}")
                        continue
                    url = match.group('url')
                    method = match.group('method')  # 提取请求方法
                    log_entries.append((ip, timestamp, url, method))  # 添加请求方法
                else:
                    print(f"解析失败: {line.strip()}")  # 调试日志格式

    if not log_entries:
        raise ValueError("Error: No log entries parsed from log file. 请检查日志格式！")

    return log_entries


# 获取请求列表

def plot_confusion_matrix(y_true, y_pred, filename="media/confusion_matrix.png"):
    # 确保目录存在
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    # 生成混淆矩阵的代码...
    plt.savefig(filename, dpi=300, bbox_inches="tight")

def plot_confusion_matrix(y_true, y_pred, filename="media/confusion_matrix.png"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)  # Create directory if missing

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Malicious"],
        yticklabels=["Normal", "Malicious"]
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to {filename}")


# ------------------ ROC Curve ------------------
def plot_roc_curve(y_true, y_pred_proba, auc_score, filename="media/roc_curve.png"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC Curve (AUC = {auc_score:.2f}%)")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"ROC curve saved to {filename}")
# # 调用函数生成图表
# plot_confusion_matrix(y_test, y_pred)
# plot_roc_curve(y_test, y_pred_proba, auc_score)


def get_query_list(filename):
    filepath = os.path.join(os.getcwd(), filename)
    with open(filepath, 'r', encoding='utf-8') as file:
        data = file.readlines()
    return list(set([urllib.parse.unquote(line.strip()) for line in data]))

# URL 特征提取
def normalize_url(url):
    # 解码 URL 编码字符
    url = urllib.parse.unquote(url)
    # 去除空格和其他符号
    url = url.replace(' ', '').replace('+', '')
    return url.lower()  # 统一转换为小写


def extract_url_features(query):
    query = normalize_url(query)

    # 辅助函数：计算URL参数数量
    def count_params(url):
        if '?' not in url:
            return 0
        query_str = url.split('?', 1)[1]
        return len(query_str.split('&'))

    param_count = count_params(query)

    return [
        # 1. SQL注入特征
        int('union' in query),
        int('select' in query),
        int('--' in query),
        int('#' in query),
        int('>' in query),
        int('<' in query),
        int('or1=1' in query),  # 注意：归一化后空格已移除
        int(';' in query),

        # 2. 命令注入特征
        int('&&' in query),
        int('|' in query or '||' in query),
        int('whoami' in query),
        int('nc' in query),
        int('curl' in query),
        int('eval' in query),
        int('`' in query),

        # 3. 路径遍历特征
        int('..' in query),
        int('/../' in query),
        int('%2f../' in query or '%2f..%2f' in query),  # 双重编码检测

        # 4. 敏感目录泄露
        int('/wp-admin/' in query),
        int('/.git/' in query),
        int('token=' in query),
        int('password=' in query),
        int('debug=' in query),
        int('sleep(' in query),  # SQL时间盲注
        int('exec(' in query),  # 命令执行

        # 5. 配置与日志泄露
        int('/.env' in query),
        int('/phpmyadmin/' in query),
        int('/console/' in query),
        int('config.' in query),  # 匹配config.*
        int('session=' in query),

        # 6. 统计特征
        int(param_count > 5),  # 参数数量>5
        int('.log' in query),
        int('log=' in query),
        len(query),  # URL总长度
        int(query.count('/') > 6),  # 路径深度>6
        int('${' in query)  # 模板注入
    ]
# 生成 n-grams 特征
def get_ngrams(query):
    query = query.lower()  # 统一转换为小写
    ngrams = [query[i:i + 3] for i in range(len(query) - 3)]

    # 关键字检测
    dangerous_keywords = ['admin', 'login', 'config', '.env', 'wp-admin', 'manager', 'database', 'phpinfo', 'password', 'wp-config']
    for keyword in dangerous_keywords:
        if keyword in query:
            ngrams.append(f"keyword:{keyword}")

    # SQL 注入模式
    sql_patterns = ['sql', 'union', '--', '%20', 'cmd', 'passwd']
    for pattern in sql_patterns:
        if pattern in query:
            ngrams.append(f"sql_pattern:{pattern}")

    return ngrams

# 组合 TF-IDF + 手工特征
def prepare_features(queries, vectorizer=None, training=True):
    if not queries:  # 避免空列表
        print("Error: Empty query list received in prepare_features().")
        return csr_matrix((0, vectorizer.max_features)), vectorizer  # 返回空稀疏矩阵

    if training:
        vectorizer = TfidfVectorizer(tokenizer=get_ngrams, max_features=5000, max_df=0.5, min_df=5)
        X_tfidf = vectorizer.fit_transform(queries)
    else:
        X_tfidf = vectorizer.transform(queries)  # 只使用 transform，避免特征维度变化

    additional_features = [extract_url_features(query) for query in queries]
    X_additional = csr_matrix(np.array(additional_features))  # 转换为稀疏矩阵
    X_combined = hstack([X_tfidf, X_additional])  # **使用稀疏矩阵，避免内存溢出**

    return X_combined, vectorizer


# 解决类别不均衡（SMOTE 过采样）
def balance_data(X, y):
    smote = SMOTE(sampling_strategy=1.0, random_state=42)
    return smote.fit_resample(X, y)


def train_xgboost(good_file, bad_file, model_filename, vectorizer_filename, target_ratio=1.0):
    print("正在训练 XGBoost 模型.............")

    # 确保 model_filename 和 vectorizer_filename 有有效的目录
    model_dir = os.path.dirname(os.path.abspath(model_filename))
    vectorizer_dir = os.path.dirname(os.path.abspath(vectorizer_filename))

    # 打印保存路径
    print(f"模型将保存到: {model_filename}")
    print(f"向量化器将保存到: {vectorizer_filename}")

    # 确保目录存在
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(vectorizer_dir, exist_ok=True)

    good_queries = get_query_list(good_file)
    bad_queries = get_query_list(bad_file)

    target_good_queries = min(int(len(bad_queries) * target_ratio), len(good_queries))
    good_queries = random.sample(good_queries, target_good_queries)

    queries = bad_queries + good_queries
    labels = [1] * len(bad_queries) + [0] * len(good_queries)

    X, vectorizer = prepare_features(queries, training=True)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.1, random_state=42)

    X_train_resampled, y_train_resampled = balance_data(X_train, y_train)

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        objective='binary:logistic',
        eval_metric='logloss'
    )

    model.fit(X_train_resampled, y_train_resampled)

    # 计算评估指标
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred) * 100
    recall = recall_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred) * 100
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # 获取正类的预测概率
    auc_score = roc_auc_score(y_test, y_pred_proba) * 100  # 计算AUC
    print(f"测试集AUC: {auc_score:.2f}%")  # 新增输出
    print(f"训练集准确率: {model.score(X_train_resampled, y_train_resampled):.2f}")
    print(f"测试集准确率: {accuracy:.2f}%")
    print(f"recall: {recall:.2f}%")
    print(f"f1: {f1:.2f}%")
    print(classification_report(y_test, y_pred))
    # ------------------ 生成混淆矩阵与ROC曲线 ------------------
    plot_confusion_matrix(y_test, y_pred, filename="media/confusion_matrix.png")
    plot_roc_curve(y_test, y_pred_proba, auc_score, filename="media/roc_curve.png")


    # # 保存模型和向量器
    # with open(model_filename, 'wb') as model_file, open(vectorizer_filename, 'wb') as vectorizer_file:
    #     pickle.dump(model, model_file)
    #     pickle.dump(vectorizer, vectorizer_file)
    # 定义达标阈值（例如F1-score ≥ 98%）
    if f1 >= 98.0:
        # 保存模型
        with open(model_filename, 'wb') as model_file, open(vectorizer_filename, 'wb') as vectorizer_file:
            pickle.dump(model, model_file)
            pickle.dump(vectorizer, vectorizer_file)
    else:
        print("模型未达标，不保存")
    # 确认文件是否成功保存
    print(f"模型是否成功保存: {os.path.exists(model_filename)}")
    print(f"向量化器是否成功保存: {os.path.exists(vectorizer_filename)}")
    print(f"TF-IDF 特征维度: {X.shape[1]}")  #TF-IDF 特征维度

    # 返回模型、向量器和评估指标
    return model, vectorizer, (accuracy, recall, f1)




def detect_malicious_requests(log_entries, model_filename, vectorizer_filename,
                              user_whitelist=None, user_blacklist=None,
                              threshold=0.5):
    """
    :param threshold: 恶意概率阈值 (0.0~1.0)，默认 0.5（50%）
    """
    # 参数校验
    if not (0 <= threshold <= 1):
        raise ValueError("阈值必须在 0.0 到 1.0 之间")

    results = []
    malicious_ips = defaultdict(int)

    # 处理黑白名单（空值保护）
    user_whitelist = user_whitelist or []
    user_blacklist = user_blacklist or []

    combined_whitelist = set(ALLOWED_URLS) | set(user_whitelist)
    combined_blacklist = set(HIGH_RISK_PATHS) | set(user_blacklist)

    # 预处理URL
    urls = [entry[2].lower() for entry in log_entries]

    # 加载模型
    model, vectorizer = load_model_and_vectorizer(model_filename, vectorizer_filename)

    # 特征提取与预测
    X = prepare_features(urls, vectorizer=vectorizer, training=False)[0]
    probabilities = model.predict_proba(X)[:, 1]  # 0~1 概率值

    for entry, prob in zip(log_entries, probabilities):
        ip, timestamp, url, method = entry
        url_lower = url.lower()

        # 1. 白名单检查（精确匹配）
        trigger = None

        if url_lower in combined_whitelist:
            label = 0
            trigger = 'whitelist'
        else:
            # 2. 黑名单检查（正则严格匹配）
            is_black = any(search(rf"\b{re.escape(pattern)}\b", url_lower)
                           for pattern in combined_blacklist)
            if is_black:
                # 仅当概率 ≥ 阈值时标记为恶意
                if prob >= threshold:
                    label = 1
                    trigger = 'blacklist'
                else:
                    label = 0  # 概率不足视为正常
            else:
                # 3. 模型预测
                label = 1 if prob >= threshold else 0
                trigger = 'model' if label else None

        # 统计恶意IP
        if label == 1:
            malicious_ips[ip] += 1
        # print(
        #     f"URL: {url}, "
        #     f"Probability: {prob:.4f}, "
        #     f"Threshold: {threshold}, "
        #     f"Label: {label}, "
        #     f"Trigger: {trigger}"
        # )

        results.append({
            'ip': ip,
            'time': timestamp,
            'url': url,
            'method': method,
            'probability': prob,
            'malicious': bool(label),
            'trigger': trigger  # 记录判定依据
        })

    return results, malicious_ips
# def detect_malicious_requests(log_entries, model_filename, vectorizer_filename, threshold=0.5):
#     results = []
#     malicious_ips = defaultdict(int)  # 用于统计恶意 IP 的出现次数
#
#     # 从 log_entries 中提取 URL 并统一转换为小写
#     urls = [entry[2].lower() for entry in log_entries]
#
#     # 加载模型和向量化器
#     model, vectorizer = load_model_and_vectorizer(model_filename, vectorizer_filename)
#
#     if not urls:
#         print("Warning: No URLs found in log entries.")
#         return [], {}
#
#     # 准备特征并进行预测
#     X = prepare_features(urls, vectorizer=vectorizer, training=False)[0]
#     probabilities = model.predict_proba(X)[:, 1]
#
#     # 调试输出 TF-IDF 维度和模型期望的维度
#     print(f"[DEBUG] TF-IDF 维度: {X.shape[1]}，模型期望维度: {model.n_features_in_}")
#
#     for entry, prob in zip(log_entries, probabilities):
#         ip, timestamp, url, method = entry  # 需要保证 entry 包含 method
#
#         # 根据概率判断是否为恶意请求
#         label = 1 if prob >= threshold else 0
#
#         # 如果 URL 在白名单中，则跳过该请求
#         if url in ALLOWED_URLS:
#             label = 0  # 不认为这是恶意请求
#
#         if is_high_risk_url(url):  # 使用统一的小写检测
#             label = 1
#
#         if label == 1:  # 如果是恶意请求，记录 IP
#             malicious_ips[ip] += 1
#
#         # 将结果添加到返回的列表
#         results.append({
#             'ip': ip,
#             'time': timestamp,
#             'url': url,
#             'method': method,  # 确保 method 存在
#             'probability': prob,
#             'malicious': bool(label)
#         })
#
#     return results, malicious_ips

# 打印带颜色的检测结果
def print_results(results, malicious_ips):
    # 输出恶意请求
    for result in results:
        if result['url'].endswith(('.css', '.js', '.ico')):  # 过滤静态资源请求
            continue
        status = "恶意请求" if result['malicious'] else "正常请求"
        color = RED if result['malicious'] else GREEN
        print(f"{color}时间: {result['time']}, IP: {result['ip']}, URL: {result['url']} -> {status}{RESET}")

    # 输出恶意 IP 统计
    print("\n恶意 IP 统计：")
    for ip, count in sorted(malicious_ips.items(), key=lambda x: x[1], reverse=True):
        if count < 10:
            continue

        print(f"IP: {ip}, 恶意请求次数: {count}")


    # 生成恶意 IP 报告
    #generate_report(malicious_ips)生成恶意 IP 报告
    #generate_attack_map(malicious_ips)


# 加载模型并进行 URL 预测
def detect_url(url, model_filename="../data/xgboost_model.pkl", vectorizer_filename="../data/tfidf_vectorizer.pkl",
               threshold=0.5):
    model, vectorizer = load_model_and_vectorizer(model_filename, vectorizer_filename)

    features = vectorizer.transform([url])
    probability = model.predict_proba(features)[:, 1][0]

    return "恶意请求" if probability >= threshold else "正常请求"



# 运行程序
if __name__ == '__main__':
    good_file = 'goodqueries2.txt'
    bad_file = 'badqueries.txt'

    model_filename = 'xgboost_model.pkl' #5036
    vectorizer_filename = 'tfidf_vectorizer.pkl' #5036
    target_ratio = 0.5
    flag = 1


    if flag == 0:
        log_entries = parse_log('data.txt')
        if not log_entries:
            print("Error: No log entries parsed from log file.")
            exit()
        results, malicious_ips = detect_malicious_requests(log_entries, model_filename, vectorizer_filename)
        print_results(results, malicious_ips)
    else:
        model, vectorizer, metrics = train_xgboost(good_file, bad_file, model_filename, vectorizer_filename,
                                                   target_ratio)
        accuracy, recall, f1 = metrics  # 解包评估指标
        log_entries = parse_log('data.txt')
        if not log_entries:
            print("Error: No log entries parsed from log file.")
            exit()
        results, malicious_ips = detect_malicious_requests(log_entries, model_filename, vectorizer_filename)
        print_results(results, malicious_ips)
