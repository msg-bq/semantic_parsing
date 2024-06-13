import random
from generate_individual_func.from_builtin.read_datas import money_indicators,industry_product,all_country,all_places,currency,company_name


####### 获取百分数 ########
def get_percentage():
    f = random.uniform(0,100)
    f = round(f,2)
    return str(f)+'%'


####### 获取资金 ########
def get_rmb():
    i = random.uniform(10,99999999)
    i = round(i,2)
    return str(i)+'万元'


####### 获取价格 ########
def get_price():
    i = random.uniform(10,999999)
    i = round(i,2)
    return str(i)+'元'


####### 获取EPS ########
def get_eps():
    i = random.uniform(0,10)
    i = round(i,1)
    return str(i)+'元'


####### 获取单数 ########
def get_dan():
    i = random.randint(0,99999)
    return str(i)+'单'


####### 获取供应量 ########
def get_offer():
    l = ['短缺','充足']
    return random.choice(l)


####### 获取时限 ########
def get_day():
    i = random.randint(0,2) 
    if i == 1:
        d = random.randint(1,100)
        return str(d)+'天'
    elif i == 2:
        d = random.randint(1,10)
        return str(d)+'周'
    else:
        d = random.randint(1,3)
        return str(d)+'月'
    

####### 获取年数 ########
def get_year():
    d = random.randint(1,10)
    return str(d)+'年'


####### 获取数目 ########
def get_num():
    d = random.randint(10,1000000)
    return d


####### 获取人数 ########
def get_people():
    p = random.uniform(1,99999)
    p = round(p,2)
    return str(p) + '万人'


####### 获取人口密度 ########
def get_density():
    p = random.randint(10,9999)
    return str(p) + '人/平方千米'


####### 获取企业数量 ########
def get_sum_comp():
    p = random.randint(1,9999)
    return str(p) + '家'


####### 获取评分 ########
def get_score():
    d = random.uniform(0,100)
    d = round(d,1)
    return str(d)+'分'


####### 获取gdp ########
def get_gdp():
    gdp = random.uniform(10,99999999)
    gdp = round(gdp,2)
    return str(gdp) + '亿元'


####### 获取重量 ########
def get_voltage():
    g = random.uniform(10,9999)
    g = round(g,2)
    return str(g) + '万吨'


####### 获取工资 ########
def get_salary():
    i = random.uniform(1,100)
    i = round(i,2)
    return str(i)+'千元'


####### 获取高低状况 ########
def get_gaodi():
    return random.choice(['高','较高','较低','低'])


####### 获取升降状况 ########
def get_inc_dec():
    return random.choice(['上升','下降','不变'])


####### 获取格局状况 ########
def get_struct():
    return random.choice(['激烈','平稳','放缓'])


####### 获取强弱状况 ########
def get_strength():
    return random.choice(['强','较强','较弱','弱'])


####### 获取大小状况 ########
def get_big_small():
    return random.choice(['大','较大','较小','小'])


####### 获取效果 ########
def get_effect():
    return random.choice(['高效','低效'])


####### 获取复杂度状况 ########
def get_complexity():
    return random.choice(['复杂','简单','较为复杂','较为简单'])



####### 获取舆论倾向 ########
def get_tendency():
    return random.choice(['喜爱度高','信任度高','口碑良好','有争议性','喜爱度低','信任度低','口碑不好'])


####### 获取周期状况 ########
def get_periodicity():
    return random.choice(['引入期','成长期','成熟期','衰退期'])


####### 获取程度 ########
def get_deep():
    return random.choice(['严重','放缓','持续加深','持续放缓'])


####### 获取行业状态 ########
def get_indus_state():
    return random.choice(['疲软','兴盛'])


####### 获取bool值 ########
def get_bool():
    return random.choice(['是','否'])


####### 获取速度 ########
def get_speed():
    return random.choice(['迅速','平稳','缓慢'])


####### 获取市场状况 ########
def get_mark_state():
    return random.choice(['景气','平稳','衰退'])

# ==============================================================================================

####### 获取行业 ########
def get_random_indus():
    all_indus = list(industry_product.keys())
    res = str(random.choice(all_indus))
    if ',' in res:
        res = res.split(',')[0].strip()
    if '、' in res:
        res = res.split('、')[-1].strip()
    return res


####### 获取时间(年) ########
def get_random_time1():
    year = random.randint(1980,2024)
    random_time = str(year)+'年'
    return random_time
    

####### 获取时间(年,月) ########
def get_random_time2():
    year = random.randint(1980,2024)
    month = random.randint(1,12)
    random_time = str(year)+'年'+str(month)+'月'
    return random_time
    

####### 获取时间(年,季度) ########
def get_random_time3():
    year = random.randint(1980,2024)
    season = random.choice(['第1','第2','第3','第4'])
    random_time = str(year)+'年'+str(season)+'季度'
    return random_time


def get_random_time():
    i = random.randint(0,2)
    if i == 0:
        time = get_random_time1()
    elif i == 1:
        time = get_random_time2()
    else:
        time = get_random_time3()
    return time

######## 获取时间段(年) #########
def get_random_spantime():
    i = random.randint(0,2)
    if i == 0:
        time = get_random_time1()
        year1 = int(time.split('年')[0])
        year2 = random.randint(year1+1,year1+11)
        time1 = str(year1)+'年'
        time2 = str(year2)+'年'
        return time1 + ',' + time2
    elif i == 1:
######## 获取时间段(年,月) #########
        time = get_random_time2()
        datas = time.split('年')
        year1 = int(datas[0])
        month1 = int(datas[1].split('月')[0])
        if month1 < 12:
            month2 = random.randint(month1+1,12)
            year2 = year1
        else:
            year2 = year1 + random.randint(1,10)
            month2 = month1
        time1 = str(year1)+'年'+str(month1)+'月'
        time2 = str(year2)+'年'+str(month2)+'月'
        return time1 + ',' + time2
    else:
######## 获取时间段(年,季度) #########
        time = get_random_time3()
        datas = time.split('年')
        year1 = int(datas[0])
        data = datas[1].split('季度')[0]
        season1 = int(data.split('第')[1])
        if season1 < 4:
            season2 = random.randint(season1+1,4)
            year2 = year1
        else:
            season2 = season1
            year2 = year1 + random.randint(1,10)
        time1 = str(year1)+'年'+'第'+str(season1)+'季度'
        time2 = str(year2)+'年'+'第'+str(season2)+'季度'
        return time1 + ',' + time2


####### 获取公司 ########
def get_random_comp():
    return random.choice(company_name)


####### 获取品牌 ########
def get_random_brand():
    comp = random.choice(company_name)
    brand = comp.split('股份')[0]+'品牌企业'
    return brand


####### 获取地点 ########
def get_random_place():
    i = random.randint(0,len(all_places)-1)
    place = all_places[i][0].replace('\\u3000','')
    return place


####### 获取国家 ########
def get_random_country():
    i = random.randint(0,len(all_country)-1)
    return all_country[i][0]


####### 获取货币 ########
def get_random_currency():
    return random.choice(currency)


####### 获取金融指标 ########
def get_random_finance():
    return random.choice(list(money_indicators.keys()))

####### 获取金融指标及其对应的指定单位 #############
def get_random_finance_and_result():
    flag=1
    while flag:
        key = random.choice(list(money_indicators.keys()))
        value = money_indicators[key]
        if value != ' ' and len(key)<8:
            flag = 0
    if value == '元':
        num = random.uniform(1,100)
        num = round(num,2)
        return key + ',' + str(num) + '万元'
    
    elif value == '天':
        num = random.randint(5,100)
        return key + ',' + str(num) + '天'
    
    elif value == '百分数':
        num = random.uniform(0,50)
        num = round(num,2)
        return key + ',' + str(num) + '%'
    
    elif value == '倍数':
        num = random.uniform(0,10)
        num = round(num,1)
        return key + ',' + str(num) + '倍'
    


################ 获取随机产业及其对应产品 ###################
def get_random_industry_and_product():
    keys = list(industry_product.keys())
    i = random.randint(0, len(keys)-1)
    industry = keys[i]

    j = random.randint(0,len(industry_product[industry])-1)
    product = industry_product[industry][j]

    indus = str(industry)

    if ',' in indus:
        indus = indus.split(',')[0].strip()
    if '、' in indus:
        indus = indus.split('、')[-1].strip()

    pro = str(product)
    if ',' in pro:
        pro = pro.split(',')[0].strip()
    if '、' in pro:
        pro = pro.split('、')[-1].strip()
    
    return indus + ',' + pro


################ 获取随机产品 ###################
def get_random_product():
    keys = list(industry_product.keys())
    i = random.randint(0, len(keys)-1)
    industry = keys[i]

    j = random.randint(0,len(industry_product[industry])-1)
    product = industry_product[industry][j]

    pro = str(product)
    if ',' in pro:
        pro = pro.split(',')[0].strip()
    if '、' in pro:
        pro = pro.split('、')[-1].strip()
    
    return pro


################ 获取随机能源 ###################
def get_random_sources():
    l = ['煤炭','焦炭及半焦炭','石油','原油','煤油','液化石油气','汽油','天然气','煤气','太阳能','风能','水能','地热能','生物质能','核能']
    return random.choice(l)


################ 获取随机成本类型 ###################
def get_cost_type():
    l = ['理论成本','应用成本','财务成本','管理成本','实际成本','估计成本','资本成本','重置成本',\
           '单位成本','总成本','平均成本','车间成本','工厂成本','生产成本','销售成本','边际成本',\
            '增量成本','差别成本','可控成本','不可控成本','变动成本','固定成本','加工成本','运营成本']
    return random.choice(l)
