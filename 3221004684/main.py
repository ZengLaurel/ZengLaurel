from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer(r'D:\Anaconda\Lib\site-packages\roberta-large-nli-stsb-mean-tokens')


# 准备要比较的文本
def readTextPath():
    list_path = []
    list_path.append(r'D:\homework\data\orig.txt')
    list_path.append(r'D:\homework\data\orig_0.8_add.txt')
    list_path.append(r'D:\homework\data\orig_0.8_del.txt')
    list_path.append(r'D:\homework\data\orig_0.8_dis_1.txt')
    list_path.append(r'D:\homework\data\orig_0.8_dis_10.txt')
    list_path.append(r'D:\homework\data\orig_dis15_path.txt')
    return list_path


# 使用模型编码文本
def editText(list_path):
    list = []
    for path in list_path:
        list.append(model.encode(path, convert_to_tensor=True))
    return list


# 计算文本相似度（余弦相似度）
def calSimilarity(list):
    cos_similarity = []
    for i in range(len(list)):
        similarity_scores = []  # 创建一个空的子列表来存储相似度分数
        for j in range(len(list)):
            similarity_scores.append(util.pytorch_cos_sim(list[i], list[j]))
        cos_similarity.append(similarity_scores)  # 将子列表附加到cos_similarity列表中
    return cos_similarity

# 输出相似度分数
def writeResult(cos_similarity):
    with open(r'D:\ruangong\similarity_result.txt', 'a', encoding='utf-8') as file:
        file.write(f"文本相似度如下表所示：\n")
        file.write(f" \t")
        for i in range(len(cos_similarity)):
            file.write(f"  {i}  \t")
        file.write(f" \n")
        for i in range(len(cos_similarity)):
            file.write(f"{i} \t")
            for j in range(len(cos_similarity[0])):
                file.write(f"{cos_similarity[i][j][0][0]:.4f}\t")
            file.write(f" \n")
        file.close()

def main():
    list_path = readTextPath()
    list = editText(list_path)
    cos_similarity = calSimilarity(list)
    writeResult(cos_similarity)


main()
