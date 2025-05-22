import json
import pandas as pd
import numpy as np

##### 读取金融指标 #####
def read_money_indicators(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        datas = f.read()
        data_json = json.loads(datas)
    return data_json

money_indicators = read_money_indicators('D:\\桌面\\generate_labelled_data\\generate_individual_func\\from_builtin\\data_sources\\finance.json')
print("金融指标数量:",len(money_indicators))



#### 读取货币种类 ####
def read_currency(filename):
    l = []
    with open(filename,'r',encoding='utf-8') as f:
        lines = f.read().split('\n')
        for line in lines:
            l.append(line)
    return l

currency = read_currency('D:\\桌面\\generate_labelled_data\\generate_individual_func\\from_builtin\\data_sources\\currency.txt')
print("货币种类数量:",len(currency))



#### 读取公司名称 ####
def read_company_name(filename):
    with open(filename,'r',encoding='utf-8') as f:
        lines = f.read()
        l = lines.split('\n')
        l = l[:-1]
    return l

company_name = read_company_name('D:\\桌面\\generate_labelled_data\\generate_individual_func\\from_builtin\\data_sources\\company_name.txt')
print("企业数量:",len(company_name))



######读取产业及其产品#######
def read_industry_product(filename):
    with open(filename,'r',encoding='utf-8') as f:
        content =f.read()
        datas = json.loads(content)
    return datas

industry_product = read_industry_product('D:\\桌面\\generate_labelled_data\\generate_individual_func\\from_builtin\\data_sources\\industry_product.json')
print("产业数量:",len(industry_product))



#### 读取地点 ####
def read_places(filename):
    datas = pd.read_csv(filename,sep=',',usecols=[1])
    datas = np.array(datas)
    return datas.tolist()

all_places = read_places('D:\\桌面\\generate_labelled_data\\generate_individual_func\\from_builtin\\data_sources\\all_places.csv')
print("地点数量:",len(all_places))



##### 读取国家 ######
def read_country(filename):
    datas = pd.read_csv(filename,sep=',',usecols=[1])
    datas = np.array(datas)
    return datas.tolist()

all_country = read_country('D:\\桌面\\generate_labelled_data\\generate_individual_func\\from_builtin\\data_sources\\countries.csv')
print("国家数量:",len(all_country))
