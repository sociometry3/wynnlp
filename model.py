import collections
import csv
import os
import sys
import tensorflow as tf
from bert import tokenization
from bert import modeling
from bert import optimization

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
    "bert_config_file", "chinese_L-12_H-768_A-12/bert_config.json",
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("vocab_file", "chinese_L-12_H-768_A-12/vocab.txt",
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

def create_label_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.
    output_layer = model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value

    output_weights = tf.get_variable(
        "output_weights", [num_labels, hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    output_bias = tf.get_variable(
        "output_bias", [num_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("loss"):
        if is_training:
            # I.e., 0.1 dropout
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

        logits_wx = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits_wx, output_bias)
        probabilities = tf.sigmoid(logits)
        label_ids = tf.cast(labels, tf.float32)
        per_example_loss = tf.reduce_sum(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=label_ids), axis=-1)
        loss = tf.reduce_mean(per_example_loss)

        return loss, per_example_loss, logits, probabilities

def create_sequnce_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 token_label_ids, predicate_label_id, num_token_labels, num_predicate_labels,
                 use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # We "pool" the model by simply taking the hidden state corresponding
    # to the first token. float Tensor of shape [batch_size, hidden_size]
    predicate_output_layer = model.get_pooled_output()

    intent_hidden_size = predicate_output_layer.shape[-1].value

    predicate_output_weights = tf.get_variable(
        "predicate_output_weights", [num_predicate_labels, intent_hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02))

    predicate_output_bias = tf.get_variable(
        "predicate_output_bias", [num_predicate_labels], initializer=tf.zeros_initializer())

    with tf.variable_scope("predicate_loss"):
        if is_training:
            # I.e., 0.1 dropout
            predicate_output_layer = tf.nn.dropout(predicate_output_layer, keep_prob=0.9)

        predicate_logits = tf.matmul(predicate_output_layer, predicate_output_weights, transpose_b=True)
        predicate_logits = tf.nn.bias_add(predicate_logits, predicate_output_bias)
        predicate_probabilities = tf.nn.softmax(predicate_logits, axis=-1)
        predicate_prediction = tf.argmax(predicate_probabilities, axis=-1, output_type=tf.int32)
        predicate_labels = tf.one_hot(predicate_label_id, depth=num_predicate_labels, dtype=tf.float32)
        predicate_per_example_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=predicate_logits, labels=predicate_labels), -1)
        predicate_loss = tf.reduce_mean(predicate_per_example_loss)


    #     """Gets final hidden layer of encoder.
    #
    #     Returns:
    #       float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
    #       to the final hidden of the transformer encoder.
    #     """
    token_label_output_layer = model.get_sequence_output()

    token_label_hidden_size = token_label_output_layer.shape[-1].value

    token_label_output_weight = tf.get_variable(
        "token_label_output_weights", [num_token_labels, token_label_hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02)
    )
    token_label_output_bias = tf.get_variable(
        "token_label_output_bias", [num_token_labels], initializer=tf.zeros_initializer()
    )
    with tf.variable_scope("token_label_loss"):
        if is_training:
            token_label_output_layer = tf.nn.dropout(token_label_output_layer, keep_prob=0.9)
        token_label_output_layer = tf.reshape(token_label_output_layer, [-1, token_label_hidden_size])
        token_label_logits = tf.matmul(token_label_output_layer, token_label_output_weight, transpose_b=True)
        token_label_logits = tf.nn.bias_add(token_label_logits, token_label_output_bias)

        token_label_logits = tf.reshape(token_label_logits, [-1, FLAGS.max_seq_length, num_token_labels])
        token_label_log_probs = tf.nn.log_softmax(token_label_logits, axis=-1)
        token_label_one_hot_labels = tf.one_hot(token_label_ids, depth=num_token_labels, dtype=tf.float32)
        token_label_per_example_loss = -tf.reduce_sum(token_label_one_hot_labels * token_label_log_probs, axis=-1)
        token_label_loss = tf.reduce_sum(token_label_per_example_loss)
        token_label_probabilities = tf.nn.softmax(token_label_logits, axis=-1)
        token_label_predictions = tf.argmax(token_label_probabilities, axis=-1)
        # return (token_label_loss, token_label_per_example_loss, token_label_logits, token_label_predict)

    loss = 0.5 * predicate_loss + token_label_loss
    return (loss,
            predicate_loss, predicate_per_example_loss, predicate_probabilities, predicate_prediction,
            token_label_loss, token_label_per_example_loss, token_label_logits, token_label_predictions)

def label_model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    def model_fn(features, labels, mode, params):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)
        
        (total_loss, per_example_loss, logits, probabilities) = create_label_model(
            bert_config, False, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)
        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        if use_tpu:
            def tpu_scaffold():
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                return tf.train.Scaffold()

            scaffold_fn = tpu_scaffold
        else:
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        
        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)
        
        output_spec = None
        output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            predictions={"probabilities": probabilities},
            scaffold_fn=scaffold_fn)
        return output_spec
    return model_fn

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
  
tokenizer = tokenization.FullTokenizer(vocab_file=FLAGS.vocab_file)

def sequnce_model_fn_builder(bert_config,num_token_labels, num_predicate_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        token_label_ids = features["token_label_ids"]
        predicate_label_id = features["predicate_label_id"]
        is_real_example = None
        # if "is_real_example" in features:
        #     is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        # else:
        #     is_real_example = tf.ones(tf.shape(token_label_ids), dtype=tf.float32) #TO DO

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss,
         predicate_loss, predicate_per_example_loss, predicate_probabilities, predicate_prediction,
         token_label_loss, token_label_per_example_loss, token_label_logits, token_label_predictions) = create_sequnce_model(
            bert_config, is_training, input_ids, input_mask, segment_ids,
            token_label_ids, predicate_label_id, num_token_labels, num_predicate_labels,
            use_one_hot_embeddings)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            predictions={"predicate_probabilities": predicate_probabilities,
                            "predicate_prediction":   predicate_prediction,
                            "token_label_predictions": token_label_predictions},
            scaffold_fn=scaffold_fn)

        return output_spec

    return model_fn

token_label_id2label = {}
for (i, label) in enumerate(token_label_list):
    token_label_id2label[i] = label
predicate_label_id2label = {}
for (i, label) in enumerate(label_list):
    predicate_label_id2label[i] = label
    
def initModel(): 
    tf.logging.set_verbosity(tf.logging.INFO)
    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case, FLAGS.label_init_checkpoint)
    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case, FLAGS.sequnce_init_checkpoint)
    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)
    tpu_cluster_resolver = None
    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    label_run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.label_model_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    sequnce_run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.sequnce_model_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    num_train_steps = None
    num_warmup_steps = None
    
    label_model_fn = label_model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.label_init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)
    labe_estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=label_model_fn,
        config=label_run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)
    
    sequnce_model_fn = sequnce_model_fn_builder(
        bert_config=bert_config,
        num_token_labels=len(token_label_list),
        num_predicate_labels=len(label_list),
        init_checkpoint=FLAGS.sequnce_init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)
    
    sequnce_estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=sequnce_model_fn,
        config=sequnce_run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)
    
    return labe_estimator, sequnce_estimator
        
labe_estimator, sequnce_estimator = initModel()

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

def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()

def file_based_input_fn_builder(input_file, seq_length, label_length,
                                is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([label_length], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        # d = input
        d = tf.data.TFRecordDataset(input_file)

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))
        return d

    return input_fn

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

def file_based_convert_examples_to_sequnce_features(
        examples, token_label_list, predicate_label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_sequnce_example(ex_index, example, token_label_list, predicate_label_list,
                                         max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["token_label_ids"] = create_int_feature(feature.token_label_ids)
        features["predicate_label_id"] = create_int_feature(feature.predicate_label_id)
        features["is_real_example"] = create_int_feature(
            [int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()

def file_based_sequnce_input_fn_builder(input_file, seq_length,is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "token_label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "predicate_label_id": tf.FixedLenFeature([], tf.int64),
        "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
            tf.contrib.data.map_and_batch(
                lambda record: _decode_record(record, name_to_features),
                batch_size=batch_size,
                drop_remainder=drop_remainder))

        return d

    return input_fn

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

    predict_file = os.path.join(FLAGS.label_output_dir, "predict.tf_record")
    file_based_convert_examples_to_features(predict_examples, label_list,
                                                FLAGS.max_seq_length, tokenizer, predict_file)

    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(predict_examples), num_actual_predict_examples,
                    len(predict_examples) - num_actual_predict_examples)
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)
    predict_drop_remainder = True if FLAGS.use_tpu else False
    predict_input_fn = file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=FLAGS.max_seq_length,
        label_length=len(label_list),
        is_training=False,
        drop_remainder=predict_drop_remainder)
    result = labe_estimator.predict(input_fn=predict_input_fn)
    predict_examples = []
    for (i, prediction) in enumerate(result):
        probabilities = prediction["probabilities"]
        if i >= num_actual_predict_examples:
            break
        # output_line_score_value = " ".join(str(class_probability)for class_probability in probabilities) + "\n"
        for idx, class_probability in enumerate(probabilities):
            if class_probability > 0.5:
                tmp = string + "	" + label_list[idx]
                predict_examples.append(tmp.replace("\n", ''))

    predict_examples = _create_sequnce_example(predict_examples, "test")
    
    num_actual_predict_examples = len(predict_examples)
        
    if FLAGS.use_tpu:
    # TPU requires a fixed batch size for all batches, therefore the number
    # of examples must be a multiple of the batch size, or else examples
    # will get dropped. So we pad with fake examples which are ignored
    # later on.
        while len(predict_examples) % FLAGS.predict_batch_size != 0:
            predict_examples.append(PaddingInputExample())
    
    predict_file = os.path.join(FLAGS.sequnce_output_dir, "predict.tf_record")
    file_based_convert_examples_to_sequnce_features(predict_examples, token_label_list, label_list,
                                                FLAGS.max_seq_length, tokenizer,
                                                predict_file)
    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    predict_drop_remainder = True if FLAGS.use_tpu else False
    predict_input_fn = file_based_sequnce_input_fn_builder(
        input_file=predict_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=predict_drop_remainder)

    result = sequnce_estimator.predict(input_fn=predict_input_fn)
    output_dict = dict()
    for (i, prediction) in enumerate(result):
        token_label_prediction = prediction["token_label_predictions"]
        predicate_probabilities = prediction["predicate_probabilities"]
        predicate_prediction = prediction["predicate_prediction"]
        if i >= num_actual_predict_examples:
            break
        token_label_output_line = " ".join(token_label_id2label[id] for id in token_label_prediction) + "\n"
        # predicate_predict_line = predicate_label_id2label[predicate_prediction]
        # predicate_probabilities_line = " ".join(str(sigmoid_logit) for sigmoid_logit in predicate_probabilities) + "\n"
        # # text_sentence = text
        # token_in_not_UNK = string
        token_label = token_label_output_line
        # refer_spo_str = None
        # text1_predicate = text_sentence.split("\t")[1]
        # token_in = token_in_not_UNK.split("\t")[0].split(" ")
        # token_in_predicate = token_in_not_UNK.split("\t")[1]
        token_label_out = token_label.split(" ")
        entity_sort_tuple_list = model_token_label_2_entity_sort_tuple_list(string.split(" "), token_label_out)
        # print(entity_sort_tuple_list, string, token_label_out)
        print(entity_sort_tuple_list)
        object_type, subject_type = schemas_dict_relation_2_object_subject_type[predicate_label_id2label[predicate_prediction]][0]
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
                output_dict.setdefault(text, []).append({"object_type": object_type, "predicate": predicate_label_id2label[predicate_prediction],
                                                        "object": object_value, "subject_type": subject_type,
                                                        "subject": subject_value})
    return output_dict
        
# output = doPredict("28至35岁群体，哪类产品在过去六个月内呈现出的销售增长最高？") 
# print(output)