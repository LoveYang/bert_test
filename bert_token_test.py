# -*- coding: utf-8 -*-


import collections
import os
import modeling
import optimization
import tokenization
import tensorflow as tf

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", "./token_test/data/xy_data.txt",
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", "./chinese_L-12_H-768_A-12/bert_config.json",
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", "token", "The name of the task to train.")

flags.DEFINE_string("vocab_file", "./chinese_L-12_H-768_A-12/vocab.txt",
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", "./token_test/output_dir",
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_bool(
    "convert2Tf_record", False,
    "Whether to convert source data to TF_record  ")

# "./chinese_L-12_H-768_A-12/bert_model.ckpt",
flags.DEFINE_string(
    "init_checkpoint", "./token_test/output_dir/model.ckpt-53646",
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

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", True, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 50,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 3000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, label=None):
        self.guid = guid
        self.text_a = text_a
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir=None):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir=None):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir=None):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_token_file(cls, input_file):
        with tf.gfile.Open(input_file, 'r') as rf:
            data = []
            endflag = True
            for line in rf:
                if line == '\n':
                    endflag = True
                    assert len(data[-1]) == 2, "length of single examples must be 2 bu get {0} and {1}".format(
                        len(data[-1]), "*".join(data[-1]))
                    continue
                else:
                    if endflag:
                        # add Xdata and replace " "
                        data.append([line.strip().replace(' ', '')])
                        endflag = False
                    else:
                        data[-1].append(line.strip())
        return data


class TokenDataProcess(DataProcessor):
    def __init__(self, data_dir=None, train_test_dev_rate=[0.6, 0.2, 0.2], is_shuffle=True):
        if data_dir:
            self.data_dir = data_dir
            assert sum(train_test_dev_rate) == 1, "sum of train_test_dev_rate must be 1.0"
            self.train_test_dev_rate = train_test_dev_rate
            self.is_shuffle=is_shuffle
            self.is_prepare_data=False
    def _prepare_data(self):
        if not self.is_prepare_data and self.data_dir:
            self.data = self._read_token_file(self.data_dir)
            self.data_size = len(self.data)
            if self.is_shuffle:
                import random
                random.shuffle(self.data)
            self.is_prepare_data=True

    @classmethod
    def get_labels(cls):
        return ["S", "M", "E"]

    def get_train_examples_size(self):
        self._prepare_data()
        return int(self.data_size * self.train_test_dev_rate[0])

    def get_train_examples(self, data_dir=None):
        if data_dir:
            data = self._read_token_file(data_dir)
        else:
            self._prepare_data()
            assert self.data_dir is not None, "you must set data_dir on initial or get_XXX_examples"
            data = self.data[:int(self.data_size * self.train_test_dev_rate[0])]
        examples = []
        for i, xdata in enumerate(data):
            guid = "train-%d" % (i)
            text_a = tokenization.convert_to_unicode(xdata[0])
            label = tokenization.convert_to_unicode(xdata[1])
            examples.append(
                InputExample(guid, text_a, label)
            )
        return examples

    def get_test_examples(self, data_dir=None):
        if data_dir:
            data = self._read_token_file(data_dir)
        else:
            self._prepare_data()
            assert self.data_dir is not None, "you must set data_dir on initial or get_XXX_examples"
            data = self.data[int(self.data_size * self.train_test_dev_rate[0]):int(
                self.data_size * sum(self.train_test_dev_rate[:2]))]
        examples = []
        for i, xdata in enumerate(data):
            guid = "test-%d" % (i)
            text_a = tokenization.convert_to_unicode(xdata[0])
            label = tokenization.convert_to_unicode(xdata[1])
            examples.append(
                InputExample(guid, text_a, label)
            )
        return examples

    def get_dev_examples(self, data_dir=None):
        if data_dir:
            data = self._read_token_file(data_dir)
        else:
            self._prepare_data()
            assert self.data_dir is not None, "you must set data_dir on initial or get_XXX_examples"
            data = self.data[int(self.data_size * sum(self.train_test_dev_rate[:2])):]
        examples = []
        for i, xdata in enumerate(data):
            guid = "dev-%d" % (i)
            text_a = tokenization.convert_to_unicode(xdata[0])
            label = tokenization.convert_to_unicode(xdata[1])
            examples.append(
                InputExample(guid, text_a, label)
            )
        return examples


def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    assert len(example.text_a) == len(example.label), 'id: {2} textlen is {0} and labellen is {1} \n'.format(
        len(example.text_a), len(example.label), ex_index) + example.text_a + '\n' + example.label
    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i
    tokens_a = tokenizer.tokenize(example.text_a)
    label = list(example.label)
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0:(max_seq_length - 2)]
        label = label[0:(max_seq_length - 2)]

    tokens = []
    labels = []
    segment_ids = []
    tokens.append("[CLS]")
    labels.append(label_list[0])  # append "S" to CLS and SEP
    segment_ids.append(0)
    for token, xlabel in zip(tokens_a, label):
        tokens.append(token)
        labels.append(xlabel)
        segment_ids.append(0)
    tokens.append("[SEP]")
    labels.append(label_list[0])  # append "S" to CLS and SEP
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    label_ids = list(map(lambda x: label_map[x], labels))
    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)
    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        label_ids.append(0)
        segment_ids.append(0)
    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label: %s \n(id = %s)" % (example.label, " ".join([str(x) for x in label_ids])))
    feature = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_ids=label_ids)
    return feature


def file_based_convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    def create_int_feature(values):
        f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
        return f

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        feature = convert_single_example(ex_index, example, label_list,
                                         max_seq_length, tokenizer)
        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_ids)

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
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

def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
    """Creates a sequence model."""
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    final_hidden=model.get_sequence_output()
    final_hidden_shape=modeling.get_shape_list(final_hidden,expected_rank=3)
    batch_size = final_hidden_shape[0]
    seq_length = final_hidden_shape[1]
    hidden_size = final_hidden_shape[2]
    output_size=num_labels

    with tf.variable_scope("bert_finetuning"):
        output_weights = tf.get_variable(
            "token_output_weights", [output_size, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            "token_output_bias", [output_size], initializer=tf.zeros_initializer())

        final_hidden_matrix = tf.reshape(final_hidden,
                                       [batch_size * seq_length, hidden_size])
        logits = tf.matmul(final_hidden_matrix, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        logits = tf.reshape(logits, [batch_size, seq_length, output_size])

        one_hot_labels = tf.one_hot(labels, depth=num_labels,axis=-1, dtype=tf.float32)
        entropy_loss=tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_labels,logits=logits,dim=-1,name="loss")
        per_example_loss=tf.reduce_sum(entropy_loss,axis=-1)
        loss=tf.reduce_mean(per_example_loss)
        probs=tf.nn.softmax(logits,axis=-1)

        return (loss,per_example_loss,probs,logits)



def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps,use_one_hot_embeddings):
    """Returns `model_fn` closure for Estimator."""

    def model_fn(features, labels, mode, params):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss,per_example_loss, probabilities, logits) = create_model(
            bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
            num_labels, use_one_hot_embeddings)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        if init_checkpoint:
            (assignment_map, initialized_variable_names
             ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op= optimization.create_optimizer(
                total_loss,learning_rate,num_train_steps,num_warmup_steps,use_tpu=False
            )
            predictions=tf.argmax(logits,axis=-1,output_type=tf.int32)
            accuracy=tf.metrics.accuracy(label_ids,predictions)
            tf.summary.scalar("accuracy", accuracy[1])
            output_spec=tf.estimator.EstimatorSpec(mode=mode,loss=total_loss,train_op=train_op)

        elif mode ==tf.estimator.ModeKeys.EVAL:
            predictions=tf.argmax(logits,axis=-1,output_type=tf.int32)
            accuracy=tf.metrics.accuracy(label_ids,predictions)
            eval_loss=tf.metrics.mean(per_example_loss)
            eval_metrics={"accuracy":accuracy,"eval_loss":eval_loss}
            tf.summary.scalar("accuracy",accuracy[1])
            tf.summary.scalar("eval_loss", eval_loss[1])
            output_spec=tf.estimator.EstimatorSpec(mode=mode,loss=total_loss,eval_metric_ops=eval_metrics)

        else:
            predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
            output_spec=tf.estimator.EstimatorSpec(mode=mode,predictions={"predictions":predictions,"probabilities":probabilities})

        return output_spec
    return  model_fn






def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")
    bert_config_file=os.path.abspath(FLAGS.bert_config_file)
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    if not tf.gfile.Exists(os.path.abspath(FLAGS.output_dir)):
        tf.gfile.MakeDirs(os.path.abspath(FLAGS.output_dir))

    label_list = TokenDataProcess.get_labels()
    max_seq_length = FLAGS.max_seq_length
    data_path = os.path.abspath(FLAGS.data_dir)
    dataprocess = TokenDataProcess(data_path, train_test_dev_rate=[0.65, 0.2, 0.15])
    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    # build model
    if FLAGS.do_train:
        num_train_steps = int(
            dataprocess.get_train_examples_size() / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
    # multi-GPU
    # distribute=tf.contrib.distribute.MirroredStrategy(num_gpus=4)
    distribute=None
    run_config=tf.estimator.RunConfig(
        model_dir=os.path.abspath(FLAGS.output_dir),
        save_summary_steps=10,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        train_distribute=distribute,
        eval_distribute=distribute
    )

    model_fn=model_fn_builder(bert_config=bert_config,
                              num_labels=len(label_list),
                              init_checkpoint=os.path.abspath(FLAGS.init_checkpoint),
                              learning_rate=FLAGS.learning_rate,
                              num_train_steps=num_train_steps,
                              num_warmup_steps=num_warmup_steps,
                              use_one_hot_embeddings=False # when use tpu ,it's True
                              )

    estimator= tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config
    )

    # data path
    train_file = os.path.join(os.path.abspath(FLAGS.output_dir), "train.tf_record")
    eval_file = os.path.join(os.path.abspath(FLAGS.output_dir), "eval.tf_record")
    predict_file = os.path.join(os.path.abspath(FLAGS.output_dir), "predict.tf_record")

    if FLAGS.convert2Tf_record:

        train_examples = dataprocess.get_train_examples()
        file_based_convert_examples_to_features(train_examples, label_list, max_seq_length, tokenizer, train_file)
        tf.logging.info("  Num train_examples = %d", len(train_examples))

        eval_examples = dataprocess.get_dev_examples()
        file_based_convert_examples_to_features(eval_examples, label_list, max_seq_length, tokenizer, eval_file)
        tf.logging.info("  Num dev_examples = %d", len(eval_examples))

        predict_examples = dataprocess.get_test_examples()
        file_based_convert_examples_to_features(predict_examples, label_list, max_seq_length, tokenizer, predict_file)
        tf.logging.info("  Num test_examples = %d", len(predict_examples))


    if FLAGS.do_train:


        tf.logging.info("***** Running training *****")

        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)

        train_dataset_fn=file_based_input_fn_builder(train_file,max_seq_length,is_training=True,drop_remainder=True)
        estimator.train(input_fn=lambda :train_dataset_fn({"batch_size":FLAGS.train_batch_size}),max_steps=num_train_steps)


    if FLAGS.do_eval:
        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        eval_dataset_fn = file_based_input_fn_builder(eval_file, max_seq_length, is_training=False,
                                                       drop_remainder=False)
        result = estimator.evaluate(input_fn=lambda :eval_dataset_fn({"batch_size":FLAGS.train_batch_size}), steps=None)
        output_eval_file = os.path.join(os.path.abspath(FLAGS.output_dir), "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if FLAGS.do_predict:
        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)
        predict_dataset_fn = file_based_input_fn_builder(predict_file, max_seq_length, is_training=False,
                                                      drop_remainder=False)
        result = estimator.predict(input_fn=lambda :predict_dataset_fn({"batch_size":FLAGS.train_batch_size}))
        output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            tf.logging.info("***** Predict results *****")
            for prediction in result:
                output_line = "\t".join(
                    str(class_probability) for class_probability in prediction) + "\n"
                writer.write(output_line)


if __name__ == "__main__":

    tf.app.run()
