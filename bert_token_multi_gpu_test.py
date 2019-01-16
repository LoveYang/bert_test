# -*- coding: utf-8 -*-


import collections
import os
import modeling
import optimization
import tokenization
import tensorflow as tf

from bert_token_test import TokenDataProcess, file_based_convert_examples_to_features, file_based_input_fn_builder, \
    model_fn_builder

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
    "output_dir", "./token_test/gpu_output_dir",
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_bool(
    "convert2Tf_record", False,
    "Whether to convert source data to TF_record  ")

# "./chinese_L-12_H-768_A-12/bert_model.ckpt",
flags.DEFINE_string(
    "init_checkpoint", "./chinese_L-12_H-768_A-12/bert_model.ckpt",
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
    "warmup_proportion", 0.1,
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


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")
    bert_config_file = os.path.abspath(FLAGS.bert_config_file)
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
    distribute = None
    distribute = tf.contrib.distribute.MirroredStrategy(num_gpus=4)

    run_config = tf.estimator.RunConfig(
        model_dir=os.path.abspath(FLAGS.output_dir),
        save_summary_steps=10,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        train_distribute=distribute,
        eval_distribute=distribute
    )

    model_fn = model_fn_builder(bert_config=bert_config,
                                num_labels=len(label_list),
                                init_checkpoint=os.path.abspath(FLAGS.init_checkpoint),
                                learning_rate=FLAGS.learning_rate,
                                num_train_steps=num_train_steps,
                                num_warmup_steps=num_warmup_steps,
                                use_one_hot_embeddings=False  # when use tpu ,it's True
                                )

    estimator = tf.estimator.Estimator(
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

        train_dataset_fn = file_based_input_fn_builder(train_file, max_seq_length, is_training=True,
                                                       drop_remainder=True)
        estimator.train(input_fn=lambda: train_dataset_fn({"batch_size": FLAGS.train_batch_size}),
                        max_steps=num_train_steps)

    if FLAGS.do_eval:
        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        eval_dataset_fn = file_based_input_fn_builder(eval_file, max_seq_length, is_training=False,
                                                      drop_remainder=False)
        result = estimator.evaluate(input_fn=lambda: eval_dataset_fn({"batch_size": FLAGS.train_batch_size}),
                                    steps=None)
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
        result = estimator.predict(input_fn=lambda: predict_dataset_fn({"batch_size": FLAGS.train_batch_size}))
        output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            tf.logging.info("***** Predict results *****")
            for prediction in result:
                output_line = "\t".join(
                    str(class_probability) for class_probability in prediction) + "\n"
                writer.write(output_line)


if __name__ == "__main__":
    tf.app.run()
