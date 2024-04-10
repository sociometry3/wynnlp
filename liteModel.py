
import collections
import csv
import os
import sys
import tensorflow as tf
from bert import tokenization
from bert import modeling
from bert import optimization
from recognizers_date_time import DateTimeRecognizer, Culture
cnModel = DateTimeRecognizer(Culture.Chinese).get_datetime_model()
enModel = DateTimeRecognizer(Culture.English).get_datetime_model()

schemas_dict_relation_2_object_subject_type = {
    '集计': [('度量值', '描述')],
    '分类': [('度量值', '分类')],
    '模糊分类': [('度量值', '描述')],
    '维度': [('度量值', '维度')],
    '模糊维度': [('度量值', '描述')],
    '保留': [('字段', '条件')],
    '排除': [('字段', '条件')],
    '升序': [('字段', '描述')],
    '降序': [('字段', '描述')],
    '时间范围': [('字段', '条件')]
}

flags = tf.flags

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "bert_config_file", "multi_cased_L-12_H-768_A-12/bert_config.json",
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", "multi_cased_L-12_H-768_A-12/vocab.txt",
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "label_model_dir", "./output/predicate_infer_out/epochs6/ckpt168",
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "sequnce_model_dir", "./output/sequnce_labeling_model/epochs9/model.ckpt-623",
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "label_output_dir", "./output/label_temp",
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "sequnce_output_dir", "./output/sequnce_temp",
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string(
    "sequnce_init_checkpoint", "output/sequnce_labeling_model/epochs9/model.ckpt-623",
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string(
    "label_init_checkpoint", "output/predicate_classification_model/epochs6/model.ckpt-168",
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

label_list = ['集计','分类','维度','模糊分类','模糊维度','保留','排除','升序','降序','时间范围']
token_label_list = ["[Padding]", "[category]", "[##WordPiece]", "[CLS]", "[SEP]", "B-SUB", "I-SUB", "B-OBJ", "I-OBJ", "O"]

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

class SequnceInputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_token, token_label):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_token = text_token
        self.token_label = token_label

class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.

    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.

    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_ids,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.is_real_example = is_real_example

class SequnceInputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 token_label_ids,
                 predicate_label_id,
                 is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.token_label_ids = token_label_ids
        self.predicate_label_id = predicate_label_id
        self.is_real_example = is_real_example

tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file)

predicate_classification_fn = tf.contrib.predictor.from_saved_model('./output/predicate_classification_model/model/1711365798')

sequnce_labeling_fn = tf.contrib.predictor.from_saved_model('./output/sequnce_labeling_model/model/1711437241')

def _create_example(lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
        guid = "%s-%s" % (set_type, i)
        text_str = line
        predicate_label_str = '集计'
        examples.append(
            InputExample(guid=guid, text_a=text_str, text_b=None, label=predicate_label_str))
    return examples

def _create_sequnce_example(lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
        guid = "%s-%s" % (set_type, i)
        text_token = line
        token_label = None
        print(line)
        examples.append(
            SequnceInputExample(guid=guid, text_token=text_token, token_label=token_label))
    return examples

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def _predicate_label_to_id(predicate_label, predicate_label_map):
    predicate_label_map_length = len(predicate_label_map)
    predicate_label_ids = [0] * predicate_label_map_length
    for label in predicate_label:
        predicate_label_ids[predicate_label_map[label]] = 1
    return predicate_label_ids

def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""

    if isinstance(example, PaddingInputExample):
        return InputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_ids=[0] * len(label_list),
            is_real_example=False)

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = example.text_a.split(" ")
    tokens_b = None
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    else:
        # Account for [CLS] and [SEP] with "- 2"
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[0:(max_seq_length - 2)]

    # The convention in BERT is:
    # (a) For sequence pairs:
    #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
    #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
    # (b) For single sequences:
    #  tokens:   [CLS] the dog is hairy . [SEP]
    #  type_ids: 0     0   0   0  0     0 0
    #
    # Where "type_ids" are used to indicate whether this is the first
    # sequence or the second sequence. The embedding vectors for `type=0` and
    # `type=1` were learned during pre-training and are added to the wordpiece
    # embedding vector (and position vector). This is not *strictly* necessary
    # since the [SEP] token unambiguously separates the sequences, but it makes
    # it easier for the model to learn the concept of sequences.
    #
    # For classification tasks, the first vector (corresponding to [CLS]) is
    # used as the "sentence vector". Note that this only makes sense because
    # the entire model is fine-tuned.
    tokens = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    for token in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
    tokens.append("[SEP]")
    segment_ids.append(0)

    if tokens_b:
        for token in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    label_list = example.label.split(" ")
    label_ids = _predicate_label_to_id(label_list, label_map)

    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label_ids: %s" % " ".join([str(x) for x in label_ids]))

    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids,
        is_real_example=True)
    return feature

def convert_single_sequnce_example(ex_index, example, token_label_list, predicate_label_list, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    if isinstance(example, PaddingInputExample):
        return SequnceInputFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            token_label_ids=[0] * max_seq_length,
            predicate_label_id = [0],
            is_real_example=False)

    token_label_map = {}
    for (i, label) in enumerate(token_label_list):
        token_label_map[label] = i

    predicate_label_map = {}
    for (i, label) in enumerate(predicate_label_list):
        predicate_label_map[label] = i

    text_token = example.text_token.split("\t")[0].split(" ")
    if example.token_label is not None:
        token_label = example.token_label.split("\t")[0].split(" ")
    else:
        token_label = ["O"] * len(text_token)
    assert len(text_token) == len(token_label)

    text_predicate = example.text_token.split("\t")[1]
    if example.token_label is not None:
        token_predicate = example.token_label.split("\t")[1]
    else:
        token_predicate = text_predicate
    assert text_predicate == token_predicate

    tokens_b = [text_predicate] * len(text_token)
    predicate_id = predicate_label_map[text_predicate]


    _truncate_seq_pair(text_token, tokens_b, max_seq_length - 3)

    tokens = []
    token_label_ids = []
    segment_ids = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    token_label_ids.append(token_label_map["[CLS]"])

    for token, label in zip(text_token, token_label):
        tokens.append(token)
        segment_ids.append(0)
        token_label_ids.append(token_label_map[label])

    tokens.append("[SEP]")
    segment_ids.append(0)
    token_label_ids.append(token_label_map["[SEP]"])

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    #bert_tokenizer.convert_tokens_to_ids(["[SEP]"]) --->[102]
    bias = 1 #1-100 dict index not used
    for token in tokens_b:
      input_ids.append(predicate_id + bias) #add  bias for different from word dict
      segment_ids.append(1)
      token_label_ids.append(token_label_map["[category]"])

    input_ids.append(tokenizer.convert_tokens_to_ids(["[SEP]"])[0]) #102
    segment_ids.append(1)
    token_label_ids.append(token_label_map["[SEP]"])

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        token_label_ids.append(0)
        tokens.append("[Padding]")

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(token_label_ids) == max_seq_length

    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("token_label_ids: %s" % " ".join([str(x) for x in token_label_ids]))
        tf.logging.info("predicate_id: %s" % str(predicate_id))

    feature = SequnceInputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        token_label_ids=token_label_ids,
        predicate_label_id=[predicate_id],
        is_real_example=True)
    return feature

def convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer):
    """Convert a set of `InputExample`s to a TFRecord file."""

    tf_examples = []

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        # features = collections.OrderedDict()
        # features["input_ids"] = create_int_feature(feature.input_ids)
        # features["input_mask"] = create_int_feature(feature.input_mask)
        # features["segment_ids"] = create_int_feature(feature.segment_ids)
        # features["label_ids"] = create_int_feature(feature.label_ids)
        # features["is_real_example"] = create_int_feature(
        #     [int(feature.is_real_example)])

        # tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        tf_examples.append({
            "input_ids": [feature.input_ids],
            "input_mask": [feature.input_mask],
            "segment_ids":  [feature.segment_ids]
        })
    return tf_examples

def convert_examples_to_sequnce_features(
        examples, token_label_list, predicate_label_list, max_seq_length, tokenizer,):
    """Convert a set of `InputExample`s to a TFRecord file."""

    tf_examples = []

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_sequnce_example(ex_index, example, token_label_list, predicate_label_list,
                                         max_seq_length, tokenizer)
        tf_examples.append({
            "input_ids": [feature.input_ids],
            "input_mask": [feature.input_mask],
            "segment_ids":  [feature.segment_ids],
            "token_label_ids": [feature.token_label_ids],
            "predicate_label_id": feature.predicate_label_id,
        })
    return tf_examples

token_label_id2label = {}
for (i, label) in enumerate(token_label_list):
    token_label_id2label[i] = label
predicate_label_id2label = {}
for (i, label) in enumerate(label_list):
    predicate_label_id2label[i] = label

def merge_WordPiece_and_single_word(entity_sort_list):
        # [..['B-SUB', '新', '地', '球', 'ge', '##nes', '##is'] ..]---> [..('SUB', '新地球genesis')..]
        entity_sort_tuple_list = []
        for a_entity_list in entity_sort_list:
            entity_content = ""
            entity_type = None
            for idx, entity_part in enumerate(a_entity_list):
                if idx == 0:
                    entity_type = entity_part
                    if entity_type[:2] not in ["B-", "I-"]:
                        break
                else:
                    if entity_part.startswith("##"):
                        entity_content += entity_part.replace("##", "")
                    else:
                        entity_content += entity_part
            if entity_content != "":
                entity_sort_tuple_list.append((entity_type[2:], entity_content))
        return entity_sort_tuple_list

# 把模型输出实体标签按照原句中相对位置输出
def model_token_label_2_entity_sort_tuple_list(token_in_not_UNK_list, predicate_token_label_list):
    # 除去模型输出的特殊符号
    def preprocessing_model_token_lable(predicate_token_label_list, token_in_list_lenth):
        # ToDo:检查错误，纠错
        if predicate_token_label_list[0] == "[CLS]":
            predicate_token_label_list = predicate_token_label_list[1:]  # y_predict.remove('[CLS]')
        if len(predicate_token_label_list) > token_in_list_lenth:  # 只取输入序列长度即可
            predicate_token_label_list = predicate_token_label_list[:token_in_list_lenth]
        return predicate_token_label_list
    # 预处理标注数据列表
    predicate_token_label_list = preprocessing_model_token_lable(predicate_token_label_list, len(token_in_not_UNK_list))
    entity_sort_list = []
    entity_part_list = []
    #TODO:需要检查以下的逻辑判断，可能写的不够完备充分
    for idx, token_label in enumerate(predicate_token_label_list):
        # 如果标签为 "O"
        if token_label == "O":
            # entity_part_list 不为空，则直接提交
            if len(entity_part_list) > 0:
                entity_sort_list.append(entity_part_list)
                entity_part_list = []
        # 如果标签以字符 "B-" 开始
        if token_label.startswith("B-"):
            # 如果 entity_part_list 不为空，则先提交原来 entity_part_list
            if len(entity_part_list) > 0:
                entity_sort_list.append(entity_part_list)
                entity_part_list = []
            entity_part_list.append(token_label)
            entity_part_list.append(token_in_not_UNK_list[idx])
            # 如果到了标签序列最后一个标签处
            if idx == len(predicate_token_label_list) - 1:
                entity_sort_list.append(entity_part_list)
        # 如果标签以字符 "I-"  开始 或者等于 "[##WordPiece]"
        if token_label.startswith("I-") or token_label == "[##WordPiece]":
            # entity_part_list 不为空，则把该标签对应的内容并入 entity_part_list
            if len(entity_part_list) > 0:
                entity_part_list.append(token_in_not_UNK_list[idx])
                # 如果到了标签序列最后一个标签处
                if idx == len(predicate_token_label_list) - 1:
                    entity_sort_list.append(entity_part_list)
        # 如果遇到 [SEP] 分隔符，说明需要处理的标注部分已经结束
        if token_label == "[SEP]":
            break
    entity_sort_tuple_list = merge_WordPiece_and_single_word(entity_sort_list)
    return entity_sort_tuple_list

def doPredict(text):
    string = " ".join(tokenizer.tokenize(text))
    predict_examples = _create_example([string], "test")
    num_actual_predict_examples = len(predict_examples)
    if FLAGS.use_tpu:
        # TPU requires a fixed batch size for all batches, therefore the number
        # of examples must be a multiple of the batch size, or else examples
        # will get dropped. So we pad with fake examples which are ignored
        # later on.
        while len(predict_examples) % FLAGS.predict_batch_size != 0:
            predict_examples.append(PaddingInputExample())
    examples = convert_examples_to_features(predict_examples, label_list,
                                                FLAGS.max_seq_length, tokenizer,)
    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)
    print(examples)
    result = predicate_classification_fn(examples[0])
    predict_examples = []
    print(result)
    for (i, prediction) in enumerate([result]):
        probabilities = prediction["probabilities"]
        if i >= num_actual_predict_examples:
            break
        # output_line_score_value = " ".join(str(class_probability)for class_probability in probabilities) + "\n"
        for idx, class_probabilities in enumerate(probabilities):
            for idx, class_probability in enumerate(class_probabilities):
                if class_probability > 0.5:
                    tmp = string + "	" + label_list[idx]
                    predict_examples.append(tmp.replace("\n", ''))

    predict_examples = _create_sequnce_example(predict_examples, "test")
    print(predict_examples)
    num_actual_predict_examples = len(predict_examples)

    if FLAGS.use_tpu:
    # TPU requires a fixed batch size for all batches, therefore the number
    # of examples must be a multiple of the batch size, or else examples
    # will get dropped. So we pad with fake examples which are ignored
    # later on.
        while len(predict_examples) % FLAGS.predict_batch_size != 0:
            predict_examples.append(PaddingInputExample())
    sequnce_examples = convert_examples_to_sequnce_features(predict_examples, token_label_list, label_list,
                                                FLAGS.max_seq_length, tokenizer,)
    print(sequnce_examples)
    sequnce_result = []
    for (sequnce_example) in sequnce_examples:
        sequnce_result.append(sequnce_labeling_fn(sequnce_example))
    output_dict = dict()
    print(sequnce_result)
    for (i, prediction) in enumerate(sequnce_result):
        token_label_prediction = prediction["token_label_predictions"]
        predicate_probabilities = prediction["predicate_probabilities"]
        predicate_prediction = prediction["predicate_prediction"]
        if i >= num_actual_predict_examples:
            break
        token_label_output_line = " ".join(token_label_id2label[id] for id in token_label_prediction[0]) + "\n"
        # predicate_predict_line = predicate_label_id2label[predicate_prediction]
        # predicate_probabilities_line = " ".join(str(sigmoid_logit) for sigmoid_logit in predicate_probabilities) + "\n"
        # # text_sentence = text
        # token_in_not_UNK = string
        token_label = token_label_output_line
        print(token_label)
        # refer_spo_str = None
        # text1_predicate = text_sentence.split("\t")[1]
        # token_in = token_in_not_UNK.split("\t")[0].split(" ")
        # token_in_predicate = token_in_not_UNK.split("\t")[1]
        token_label_out = token_label.split(" ")
        entity_sort_tuple_list = model_token_label_2_entity_sort_tuple_list(string.split(" "), token_label_out)
        # print(entity_sort_tuple_list, string, token_label_out)
        print(entity_sort_tuple_list)
        object_type, subject_type = schemas_dict_relation_2_object_subject_type[predicate_label_id2label[predicate_prediction[0]]][0]
        subject_list = [value for name, value in entity_sort_tuple_list if name == "SUB"]
        subject_list = list(set(subject_list))
        subject_list = [value for value in subject_list if len(value) >= 1]
        object_list = [value for name, value in entity_sort_tuple_list if name == "OBJ"]
        object_list = list(set(object_list))
        object_list = [value for value in object_list if len(value) >= 1]
        print(subject_list, object_list)
        if len(subject_list) == 0 or len(object_list) == 0:
            output_dict.setdefault(text, [])
        for subject_value in subject_list:
            for object_value in object_list:
                predicate = predicate_label_id2label[predicate_prediction[0]]
                result = {  "object_type": object_type,
                            "predicate": predicate,
                            "object": object_value, 
                            "subject_type": subject_type,
                            "subject": subject_value,
                            "subject_value_parsed": "",
                        }
                if predicate == "时间范围":
                    cntestResult = cnModel.parse(subject_value)
                    entestResult = enModel.parse(subject_value)
                    try:
                        cnstart = cntestResult[0].resolution.get("values")[0]["start"]
                        cnend = cntestResult[0].resolution.get("values")[0]["end"]
                        if cnstart and cnend:
                            result["subject_value_parsed"] = cnstart + "|" + cnend
                    except:
                        print()
                    
                    try:
                        enstart = entestResult[0].resolution.get("values")[0]["start"]
                        enend = entestResult[0].resolution.get("values")[0]["end"]
                        if cnstart and cnend:
                            result["subject_value_parsed"] = enstart + "|" + enend
                    except:
                        print()
                output_dict.setdefault(text, []).append(result)
    return output_dict
# res = doPredict("28至35岁群体，哪类产品在过去六个月内呈现出的销售增长最高？")
# print(res)