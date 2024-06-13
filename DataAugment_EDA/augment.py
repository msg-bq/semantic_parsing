from ChineseTextEDA.eda import EDA
import json
import argparse
import data_process.check_sentence as check_sentence

parser = argparse.ArgumentParser(description="Data Augmentation by EDA")
parser.add_argument("--num_of_aug", type=int, default=4, help="Number of augmented data for every sentence.")
parser.add_argument("--input_file_path", type=str, help="Path of the sentence file to be augmented.")
parser.add_argument("--output_file_path", type=str, help="Path to store the augmented data")
parser.add_argument("--alpha", type=float, default=0.1, help="Ratio of the words to be added, deleted, swapped and replaced.")
args = parser.parse_args()

num_aug = args.num_of_aug
input_file_path = args.input_file_path
output_file_path = args.output_file_path
alpha = args.alpha


# 数据统计
num_before_sentence = 0  # 原始数据大小
num_after_sentence = 0   # 增强后数据大小
num_invalid = 0          # 丢弃语句数目

eda = EDA(num_aug=num_aug)
enhance_result = []
with open(input_file_path, 'r', encoding='utf8') as reader:
    datas = json.load(reader)
    for data in datas:
        label = data['表达式']
        sentences = list(data['自然语句'])
        res = dict()   #存储每个表达式对应语句
        res['表达式'] = label
        text = []
        for sentence in sentences:
            num_before_sentence += 1
            # 增强后的语句
            aug_sentences = eda.eda(sentence, alpha_sr=alpha, alpha_ri=alpha, alpha_rs=alpha,p_rd=alpha)
            for aug_sentence in aug_sentences:
                aug_sentence = aug_sentence.replace(" ",'')
                if check_sentence.check(aug_sentence,label):
                    print(aug_sentence)
                    num_after_sentence += 1
                    text.append(aug_sentence)
                else:
                    num_invalid += 1
        res['自然语句'] = text
        enhance_result.append(res)

print("原始数据大小：",num_before_sentence)
print("增强后数据大小：",num_after_sentence)
print("丢弃增强数据大小：",num_invalid)  

file = open(output_file_path,'w',encoding='utf-8')
json.dump(enhance_result,file,ensure_ascii=False)
file.close()