import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity

# 加载预训练的ResNet50模型
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# 加载并预处理图像
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

# 提取图像的特征向量
def extract_features(img_path, model):
    img = preprocess_image(img_path)
    features = model.predict(img)
    return features.flatten()

# 图像路径
image1_path = '/Users/hang.yang/AI/pythonsamples-main/ML/CNN(卷积神经网络)/ImageVector/beizi1.jpg'
image2_path = '/Users/hang.yang/AI/pythonsamples-main/ML/CNN(卷积神经网络)/ImageVector/maomao1.jpeg'
image3_path = '/Users/hang.yang/AI/pythonsamples-main/ML/CNN(卷积神经网络)/ImageVector/maomao2.jpeg'
image4_path = '/Users/hang.yang/AI/pythonsamples-main/ML/CNN(卷积神经网络)/ImageVector/test1.jpeg'
image5_path = '/Users/hang.yang/AI/pythonsamples-main/ML/CNN(卷积神经网络)/ImageVector/zhipiao.jpg'
image6_path = '/Users/hang.yang/AI/pythonsamples-main/ML/CNN(卷积神经网络)/ImageVector/lakuten.jpg'

# 提取特征向量
features1 = extract_features(image1_path, model)
features2 = extract_features(image2_path, model)
features3 = extract_features(image3_path, model)
features4 = extract_features(image4_path, model)
features5 = extract_features(image5_path, model)
features6 = extract_features(image6_path, model)
# 计算余弦相似度
similarity = cosine_similarity([features1], [features2])[0][0]
print("杯子和猫 相似度:", similarity)
similarity = cosine_similarity([features2], [features3])[0][0]
print("两只猫 相似度:", similarity)
similarity = cosine_similarity([features4], [features5])[0][0]
print("支票和Logo1 相似度:", similarity)
similarity = cosine_similarity([features5], [features6])[0][0]
print("支票和Logo2 相似度:", similarity)
similarity = cosine_similarity([features4], [features6])[0][0]
print("Logo1和Logo2 相似度:", similarity)