# F016_pro  vue+flask知识图谱机器学习二手房推荐大价格预测可视化大数据

> 完整项目收费，可联系QQ: 81040295 微信: mmdsj186011 注明从git来的，谢谢！
也可以关注我的B站： 麦麦大数据 https://space.bilibili.com/1583208775
> 

关注B站，有好处！
up主B站账号： **麦麦大数据**

编号:  F016 pro
## 视频

[video(video-wZs8YVoT-1757921595998)(type-bilibili)(url-https://player.bilibili.com/player.html?aid=113491927172747)(image-https://i-blog.csdnimg.cn/img_convert/9dacd363efb4a01c6d6014adfbc8c638.jpeg)(title-python二手房价格预测推荐可视化系统+机器学习Vue+Flask+mysql+协同过滤推荐+前后端分离+百度图像识别API+阿里云)]

## 1 系统简介
系统简介：本系统是一个基于Vue+Flask构建的二手房预测与可视化推荐平台，专注于链家二手房数据的分析与智能推荐。系统整合了数据爬取、机器学习预测、协同过滤推荐以及交互式可视化功能，为用户提供全面的二手房市场洞察。主要功能包括：首页展示系统概览和热门房源轮播；数据卡片模块，提供房源基本信息概览，并支持查看房源地理位置（通过百度地图集成）及用户点赞收藏功能；可视化分析模块，利用Echarts展示房源价格分布、关注度热力图以及基于地理信息的房源分布，帮助用户直观理解市场趋势；价格预测模块，采用SVM（支持向量机）算法对二手房价格进行智能预测，为用户提供参考估价；房源推荐模块，基于UserCF（用户协同过滤）算法实现个性化房源推荐，提升用户找房效率；以及用户管理模块，支持用户注册、登录，并提供个人设置功能（如修改个人信息、头像和密码），确保用户体验的个性化和安全性。
## 2 功能设计
该系统采用B/S（浏览器/服务器）架构。用户通过浏览器访问基于Vue.js构建的前端界面，前端使用Vuex进行状态管理、Vue Router处理路由导航，并集成Echarts实现数据可视化、百度地图API展示地理信息，以及Neo4j图形界面进行知识图谱渲染。前端通过RESTful API与Flask后端交互，Flask后端负责核心业务逻辑，包括UserCF推荐算法实现、SVM价格预测模型训练与推理、知识图谱数据抽取与Neo4j集成、以及大模型问答接口调用。数据层使用MySQL存储结构化数据（如用户信息、房源详情），Neo4j存储知识图谱关系数据。此外，系统包含独立的Python爬虫模块，定期从链家抓取二手房数据并清洗入库，确保数据实时性与准确性。整体设计注重模块化、可扩展性，以支持大数据处理与智能推荐功能。
### 2.1系统架构图
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a4bb9c7930e44393ae2d8734f91d04ca.png)
### 2.2 功能模块图
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c0d21c0d548d413d8b78052504c75acc.png)
### 2.3 工程目录
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/3338b19fd9964378a287efedcf3910ba.png)
### 2.4 说明文档
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9a8709560c6347acbb6803b948fc53af.png)
## 3 功能展示
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/71360751b2a84fc1adf26bc1570230cc.png)
### 3.1 登录 & 注册
登录注册做的是一个可以切换的登录注册界面，点击去登录后者去注册可以切换，背景是一个视频，循环播放。
登录需要验证用户名和密码是否正确，如果**不正确会有错误提示**。
注册需要**验证用户名是否存在**，如果错误会有提示。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c434030cb3e449e5be3aaa60b541be7b.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/29a643a0c6924295942acafa98de17e9.png)
### 3.2 主页
主页的布局采用了左侧是菜单，右侧是操作面板的布局方法，右侧的上方还有用户的头像和退出按钮，如果是新注册用户，没有头像，这边则不显示，需要在个人设置中上传了头像之后就会显示。
### 3.3 推荐算法
本系统使用协同过滤推荐算法为用户推荐房源：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ff0601cc3d9842ae9473a11d84037dab.png)

通过点击房源卡片上的位置，可以在百度地图组件上查看二手房所在位置：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/bf3c09d84b504ad19105cdb7a80caad8.png)

### 3.4 房源搜索
通过输入关键词，可以模糊检索房源，并且以卡片方式展示：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ba1618c8c5ee41cea0c730a3791ebeb1.png)
### 3.5 可视化分析
可视化分析分为数据大屏、价格分析、词云分析、散点图分析等部分。
数据大屏以滚动数字、房源热力图、饼图、柱状图、折线图等方式展示房源价格情况：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b3fa35100956452c9411ce93d3be3de7.png)

关注分析以滚动柱状图方式展示用户对哪些类型的房源最为关注：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/755ccb2c9f214679b5e609b6ad8eeca5.png)

价格散点图以不同户型和价格用散点图方式进行可视化展示：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/179b9ce9d55e4dba829871e372c87160.png)
词云分析以jieba分词分析二手房源的关键词的出现词频：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9c9354a259b04e6ea311aa79efee4f7a.png)
### 3.6 知识图谱可视化
利用Python对房源信息提取成知识图谱，并且存储在neo4j中，前端用可视化技术进行展示，支持模糊搜索:
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/657ac068c571429fa15d62a1303cf58c.png)
检索：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/08b26cac2aa6404ab294f680b9d4090a.png)
### 3.7 房价预测
利用机器学习根据数据集训练模型，前端输入条件进行二手房价预测（SVM模型）：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/c52c56e754c142009a5672729d5e6dd9.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/0c8bd737c8894d568e8075e1900844a5.png)
### 3.8 购房助手
对接大模型以聊天方式，利用大模型的专业知识给用户购房建议：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4be9524beb114cb78bbe3d099fa724aa.png)
### 3.9 个人设置
个人设置方面包含了用户信息修改、密码修改功能。
用户信息修改中可以上传头像，完成用户的头像个性化设置，也可以修改用户其他信息。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4073dc5c94ce4693817e26ad305990af.png)
实名认证：利用OCR技术识别上传的身份证，并且进行实名认证：
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9d4dc197a614433284cf3030c1132c84.png)

修改密码需要输入用户旧密码和新密码，验证旧密码成功后，就可以完成密码修改。
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/780ef3b1245e4b51b2dc4cf64087e4c6.png)
## 4程序代码
### 4.1 代码说明
代码介绍：基于SVM的二手房价格预测功能使用Python实现，通过支持向量机回归算法对二手房历史数据（如面积、地段、房龄等特征）进行训练和预测。该功能包括数据预处理（缺失值处理、特征标准化）、模型训练与优化（网格搜索调参）、价格预测及结果评估（RMSE、R²指标），最终提供准确的房价估值，辅助用户做出房产交易决策。核心流程涵盖数据加载、特征工程、SVM建模和预测输出
### 4.2 流程图
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/2c83829954e14d649bc502eb77820ced.png)
### 4.3 代码实例
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 加载数据集
def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

# 数据预处理
def preprocess_data(data):
    # 处理缺失值
    data = data.dropna()
    # 选择特征和目标变量
    features = data[['area', 'age', 'location_level', 'room_count']]
    target = data['price']
    return features, target

# 特征标准化
def scale_features(features):
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features, scaler

# SVM模型训练与预测
def svm_price_prediction(features, target):
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    
    # 特征标准化
    X_train_scaled, scaler = scale_features(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 创建SVM回归模型
    svr = SVR(kernel='rbf')
    
    # 网格搜索优化参数
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [0.01, 0.1, 1, 'scale']
    }
    grid_search = GridSearchCV(svr, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search.fit(X_train_scaled, y_train)
    
    # 最佳模型
    best_svr = grid_search.best_estimator_
    
    # 预测
    y_pred = best_svr.predict(X_test_scaled)
    
    # 评估
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return best_svr, y_pred, rmse, r2, scaler

# 主函数
def main():
    # 加载数据
    data = load_data('second_house_data.csv')
    
    # 数据预处理
    features, target = preprocess_data(data)
    
    # 训练模型并预测
    model, predictions, rmse, r2, scaler = svm_price_prediction(features, target)
    
    print(f"模型评估结果：")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.4f}")
    
    # 可视化实际vs预测价格
    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(predictions)), predictions, alpha=0.5, label='预测价格')
    plt.scatter(range(len(predictions)), target[-len(predictions):], alpha=0.5, label='实际价格')
    plt.xlabel('样本索引')
    plt.ylabel('价格')
    plt.title('二手房价格预测结果')
    plt.legend()
    plt.show()
    
    return model, scaler

if __name__ == "__main__":
    trained_model, feature_scaler = main()

```
