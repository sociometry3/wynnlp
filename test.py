# # import nltk
# import spacy
# # from spacy.lang.zh import Chinese

# # nlp = Chinese()
# # cfg = {"segmenter": "pkuseg"}
# # nlp = Chinese.from_config({"nlp": {"tokenizer": cfg}})
# # nlp = Chinese.from_config({"nlp": {"tokenizer": cfg}})
# nlp = spacy.load("zh_core_web_sm")
# import zh_core_web_sm
# nlp = zh_core_web_sm.load()


# def ie_preprocess(document):
#     doc = nlp(document)
#     print([(w.text, w.pos_, [child for child in w.children]) for w in doc])
# #    sentences = nltk.sent_tokenize(document)
# #    sentences = [nltk.word_tokenize(sent) for sent in sentences]
# #    sentences = [nltk.pos_tag(sent) for sent in sentences]
# #    return sentences
# # res = ie_preprocess('我们在各省份有多少客户')
# res = ie_preprocess('用折线图表示每年的标准价格，按省份分类')
# # print(res)

# from sinan import Sinan
# si = Sinan("去年三季度")
# si.parse()

from recognizers_date_time import DateTimeRecognizer, Culture
model = DateTimeRecognizer(Culture.Chinese).get_datetime_model()
res = model.parse("今年五月")
print(res[0].resolution.get("values")[0]["start"])
# from timefhuman import timefhuman
# print(timefhuman('next year'))
# import tensorflow as tf
# from bert import tokenization

# flags = tf.flags

# FLAGS = flags.FLAGS

# flags.DEFINE_string(
#     "bert_config_file", "multi_cased_L-12_H-768_A-12/bert_config.json",
#     "The config json file corresponding to the pre-trained BERT model. "
#     "This specifies the model architecture.")

# flags.DEFINE_string("vocab_file", "multi_cased_L-12_H-768_A-12/vocab.txt",
#                     "The vocabulary file that the BERT model was trained on.")
# tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file)

# print(tokenizer.tokenize("lastyear"))