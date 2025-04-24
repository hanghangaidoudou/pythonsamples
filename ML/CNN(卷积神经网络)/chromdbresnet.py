#pip install chromadb
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
import chromadb
from chromadb.utils import embedding_functions

# 1. 初始化 Chroma 客户端
chroma_client = chromadb.PersistentClient(path="./chroma_db")  # 数据存储在本地目录

# 2. 创建或获取集合（Collection）
# 使用默认的向量相似度计算（余弦相似度）
collection = chroma_client.get_or_create_collection(name="image_vectors")

# 3. 加载预训练的ResNet50模型（用于提取特征向量）
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')

# 4. 图像预处理和特征提取函数（你的原始代码）
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def extract_features(img_path, model):
    img = preprocess_image(img_path)
    features = model.predict(img)
    return features.flatten().tolist()  # 转换为列表（Chroma需要）

# 5. 添加图像向量到 Chroma
def add_image_to_db(image_path, image_id):
    features = extract_features(image_path, model)
    collection.add(
        documents=[image_path],  # 原始图像路径（可选存储）
        embeddings=[features],   # 特征向量
        ids=[image_id]          # 唯一ID（如文件名或UUID）
    )

# 6. 搜索相似图像
def search_similar_images(query_image_path, top_k=3):
    query_features = extract_features(query_image_path, model)
    results = collection.query(
        query_embeddings=[query_features],
        n_results=top_k
    )
    return results

# ------------------- 使用示例 -------------------
if __name__ == "__main__":
    # 添加示例图像到数据库
    add_image_to_db("dfs1.jpeg", "img001")
    add_image_to_db("dfs2.jpeg", "img002")
    add_image_to_db("dfs3.jpeg", "img003")

    # 搜索与"query_cat.jpg"相似的图像
    similar_images = search_similar_images("dfs4.jpeg", top_k=2)
    print("最相似的图像：")
    for i, (img_id, distance) in enumerate(zip(similar_images["ids"][0], similar_images["distances"][0])):
        print(f"{i+1}. ID: {img_id}, 相似度: {1-distance:.2f} (路径: {similar_images['documents'][0][i]})")