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
image1_path = '/Users/hang.yang/AI/pythonsamples-main/ML/CNN(卷积神经网络)/ImageVector/houge.jpg'
image2_path = '/Users/hang.yang/AI/pythonsamples-main/ML/CNN(卷积神经网络)/ImageVector/zhipiao.jpg'

# 提取特征向量
features1 = extract_features(image1_path, model)
features2 = extract_features(image2_path, model)

# 计算余弦相似度
similarity = cosine_similarity([features1], [features2])[0][0]
print("相似度:", similarity)
