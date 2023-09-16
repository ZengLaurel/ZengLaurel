from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer(r'D:\Anaconda\Lib\site-packages\roberta-large-nli-stsb-mean-tokens')

# 准备要比较的文本
orig_path = r'D:\homework\data\orig.txt'
orig_add_path = r'D:\homework\data\orig_0.8_add.txt'
orig_del_path = r'D:\homework\data\orig_0.8_del.txt'
orig_dis1_path = r'D:\homework\data\orig_0.8_dis_1.txt'
orig_dis10_path = r'D:\homework\data\orig_0.8_dis_10.txt'
orig_dis15_path = r'D:\homework\data\orig_dis15_path.txt'

# 使用模型编码文本
orig = model.encode(orig_path, convert_to_tensor=True)
orig_add = model.encode(orig_add_path, convert_to_tensor=True)
orig_del = model.encode(orig_del_path, convert_to_tensor=True)
orig_dis1 = model.encode(orig_dis1_path, convert_to_tensor=True)
orig_dis10 = model.encode(orig_dis10_path, convert_to_tensor=True)
orig_dis15 = model.encode(orig_dis15_path, convert_to_tensor=True)

# 计算文本相似度（余弦相似度）
cosine_similarity1 = util.pytorch_cos_sim(orig, orig_add)
cosine_similarity2 = util.pytorch_cos_sim(orig, orig_del)
cosine_similarity3 = util.pytorch_cos_sim(orig, orig_dis1)
cosine_similarity4 = util.pytorch_cos_sim(orig, orig_dis10)
cosine_similarity5 = util.pytorch_cos_sim(orig, orig_dis15)


# 输出相似度分数
with open(r'D:\ruangong\similarity_result.txt', 'a', encoding='utf-8') as file:
    file.write(f"文本orig和文本orig_add的相似度：{cosine_similarity1.item():.4f}\n")
    file.write(f"文本orig和文本orig_del的相似度：{cosine_similarity2.item():.4f}\n")
    file.write(f"文本orig和文本orig_dis1的相似度：{cosine_similarity3.item():.4f}\n")
    file.write(f"文本orig和文本orig_dis10的相似度：{cosine_similarity4.item():.4f}\n")
    file.write(f"文本orig和文本orig_dis15的相似度：{cosine_similarity5.item():.4f}\n")
    file.close()

