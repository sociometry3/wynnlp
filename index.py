# import tensorflow as tf
# import spacy
# # from spacy.lang.zh import Chinese

# # nlp = Chinese()
# # cfg = {"segmenter": "pkuseg"}
# # nlp = Chinese.from_config({"nlp": {"tokenizer": cfg}})
# # nlp = Chinese.from_config({"nlp": {"tokenizer": cfg}})
# nlp = spacy.load("zh_core_web_sm")
# import zh_core_web_sm
# nlp = zh_core_web_sm.load()
# doc = nlp("用折线图表示每年的标准价格，按省份分类")
# print([(w.text, w.pos_, [child for child in w.children]) for w in doc])

import nltk;
import pandas as pd
aspect_filter = []
dic_business_id = {}
dic_business_data = {}

import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(curPath)
sys.path.append(rootPath)

# main directory

# curPath = r'D:\Github\NLPVisualizationSystem'  
data_dir =  os.path.join(curPath, 'data')
review_data_path = os.path.join(data_dir, 'UserReviewData.csv')
business_data_path = os.path.join(data_dir, 'ProductData.csv')

class ProductDataItem(object):
    def __init__(self, product_id, product_name, review_count):
        self.product_id = product_id
        self.product_name = product_name
        self.review_count = review_count

class ReviewDataItem(object):
    def __init__(self, review_id, user_name, product_id, rates, text):
        self.review_id = review_id
        self.user_name = user_name
        self.product_id = product_id
        self.rates = rates
        self.text = text

def read_data():
    review_data = pd.read_csv(review_data_path)
    product_data = pd.read_csv(business_data_path)
    for index in product_data.index:
        product_id = product_data.loc[index, "product_id"]
        product_name = product_data.loc[index,"product_name"]
        review_count = product_data.loc[index,"review_count"]
        # print(product_id, product_name, review_count)
        if review_count >= 2:
            dic_business_id[product_id] = []
            business_DataItem = ProductDataItem(product_id, product_name, review_count)
            dic_business_data[product_id] = business_DataItem

    for index in review_data.index:
        product_id = review_data.loc[index, "product_id"]    
        if product_id in dic_business_id:
            review_id = review_data.loc[index, "review_id"]
            user_name = review_data.loc[index, "user_name"]
            rates = review_data.loc[index,['rating_overall', 'rating_ease_of_use',
                                            'rating_customer_support','rating_features_functionality',
                                            'rating_value_for_money','rating_likelihood_to_recommend']].T.to_dict()
            text = review_data.loc[index, "review"]
            review_DataItem = ReviewDataItem(review_id, user_name, product_id, rates, text)
            dic_business_id[product_id].append(review_DataItem)

def extract_aspects(business_id):
    """
    从一个business的review中抽取aspects
    """

    # print("step3 extract_aspects begin==================")
    if business_id not in dic_business_id:
        print("business_id not exit")
        return None

    review_list = dic_business_id[business_id]
    aspects_dic = {}
    for review_data in review_list:
        sentence = review_data.text
        if sentence == None or str.strip(sentence) == '':
            continue
        tagged_words = []
        tokens = nltk.word_tokenize(sentence)
        tag_tuples = nltk.pos_tag(tokens)
        print(sentence, tokens, tag_tuples)
        for (word, tag) in tag_tuples:
            if tag == "NN":
                # token = {'word': string, 'pos': tag}
                # tagged_words.append(word)
                if word not in aspects_dic:
                    aspects_dic[word] = []
                aspects_dic[word].append(review_data)

    # 对字典进行排序
    aspects_sorted = sorted(aspects_dic.items(), key=lambda x: len(x[1]), reverse=True)
    aspects_dic = {}
    for index, item in enumerate(aspects_sorted):
        if item[0] in aspect_filter:
            continue

        if len(aspects_dic.items()) < 5:
            aspects_dic[item[0]] = item[1]

    # print("step3 extract_aspects end==================")
    return aspects_dic

# def aspect_based_summary(business_id):
    """
    返回一个business的summary. 针对于每一个aspect计算出它的正面负面情感以及TOP reviews.
    具体细节请看给定的文档。
    """

    aspects_dic = extract_aspects(business_id)        
    business_name = dic_business_data[business_id].product_name        

    pos_aspect_dic = {}
    neg_aspect_dic = {}
    review_segment_dic = {}

    for aspect, reviews in aspects_dic.items():
        for review in reviews:
            review_text = review.text
            if review_text == None or str.strip(review_text) == '':
                continue
            review_segment = get_segment(review_text, aspect, aspects_dic)
            # 粗略筛选一下
            if len(str.strip(review_segment)) > len(aspect) + 3:
                print(review_segment)
                key = str(review.review_id) + "_" + aspect
                review_segment_dic[key] = review_segment

                score = sentimentModel.predict_prob(review_segment)

                if score > 0.7:
                    if aspect not in pos_aspect_dic:
                        pos_aspect_dic[aspect] = []
                    pos_aspect_dic[aspect].append([key, score])
                else:
                    if aspect not in neg_aspect_dic:
                        neg_aspect_dic[aspect] = []
                    neg_aspect_dic[aspect].append([key, score])

    dic_aspect_summary = {}
    for aspect, reviews in aspects_dic.items():
        if aspect not in dic_aspect_summary:
            dic_aspect_summary[aspect] = {}

        # 算某个aspect的得分
        try:
            pos_aspect_review_nums = len(pos_aspect_dic[aspect])
            pos_aspect_total_scores = 0
            for item in pos_aspect_dic[aspect]:
                pos_aspect_total_scores += item[1]
        except:
            pos_aspect_review_nums = 0
            pos_aspect_total_scores = 0
                    
        try:
            neg_aspect_review_nums = len(neg_aspect_dic[aspect])
            neg_aspect_total_scores = 0
            for item in neg_aspect_dic[aspect]:
                neg_aspect_total_scores += item[1]
        except:
            neg_aspect_review_nums = 0
            neg_aspect_total_scores = 0
        
        aspect_review_nums = pos_aspect_review_nums + neg_aspect_review_nums if pos_aspect_review_nums + neg_aspect_review_nums>0 else 1 
        
        aspect_score = (pos_aspect_total_scores + neg_aspect_total_scores) / aspect_review_nums

        dic_aspect_summary[aspect]["rating"] = aspect_score

        # TOP 5 正面
        if aspect in pos_aspect_dic:
            aspects_pos_sorted = sorted(pos_aspect_dic[aspect], key=lambda x: x[1], reverse=True)
            aspects_pos_contents = {}
            dic_aspect_summary[aspect]["pos"] = []
            for index, item in enumerate(aspects_pos_sorted):
                if len(dic_aspect_summary[aspect]["pos"]) >= 5:
                    break
                review_content = review_segment_dic[item[0]]
                if review_content not in aspects_pos_contents:
                    dic_aspect_summary[aspect]["pos"].append(review_content)
                    aspects_pos_contents[review_content] = None
        else:
            dic_aspect_summary[aspect]["pos"] = ['None']
            

        # TOP 5 负面
        if aspect in neg_aspect_dic:
            aspects_neg_sorted = sorted(neg_aspect_dic[aspect], key=lambda x: x[1], reverse=False)
            aspects_neg_contents = {}
            dic_aspect_summary[aspect]["neg"] = []
            for index, item in enumerate(aspects_neg_sorted):
                if len(dic_aspect_summary[aspect]["neg"]) >= 5:
                    break
                review_content = review_segment_dic[item[0]]
                if review_content not in aspects_neg_contents:
                    dic_aspect_summary[aspect]["neg"].append(review_content)
                    aspects_neg_contents[review_content] = None
        else:
            dic_aspect_summary[aspect]["neg"] = ['None']

    all_aspect_scores = 0
    for item in dic_aspect_summary.items():
        all_aspect_scores += item[1]["rating"]

    business_rating = all_aspect_scores / len(dic_aspect_summary.items())
    average_user_rating = dic_business_id[business_id][0].rates

    return {'business_id':business_id,
        'business_name':business_name,
        'business_rating':business_rating,
        'average_user_rating':average_user_rating,
        'aspect_summary':dic_aspect_summary
        }

read_data()
res = extract_aspects(0)
print(res)