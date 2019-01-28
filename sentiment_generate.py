# -*- coding: utf-8 -*-
import tensorflow as tf
import os

import modeling
import optimization
import tokenization

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", "./token_test/data/data_sentiment_analysis",
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", "./chinese_L-12_H-768_A-12/bert_config.json",
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", "sentiment_generate", "The name of the task to train.")

flags.DEFINE_string("vocab_file", "./chinese_L-12_H-768_A-12/vocab.txt",
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", "./token_test/output_sentiment_generate_dir",
    "The output directory where the model checkpoints will be written.")

## Other parameters

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
flags.DEFINE_integer(
    "max_target_seq_length", 10,
    "The maximum total output sequence length after decoder generate"
)

flags.DEFINE_bool("do_train", True, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 5,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 3000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")


def file_based_input_fn_builder(input_file, input_seq_length, output_seq_length, is_training,
                                drop_remainder):
    name_to_features = {
        "sentiment_labels": tf.FixedLenFeature([20], tf.int64),
        "input_token_ids": tf.FixedLenFeature([input_seq_length], tf.int64),
        "input_mask": tf.FixedLenFeature([input_seq_length], tf.int64),
        "target_token_ids": tf.FixedLenFeature([output_seq_length], tf.int64),
        "target_mask": tf.FixedLenFeature([output_seq_length], tf.int64),
        "segment_ids": tf.FixedLenFeature([input_seq_length], tf.int64),
    }

    def decode_record(record):
        features = tf.parse_single_example(record, name_to_features)
        features["sentiment_labels"] = tf.add(features["sentiment_labels"],
                                              tf.constant([x * 4 for x in range(20)], dtype=tf.int64)) + 2
        return features

    def input_fn(batch_size):
        dataset = tf.data.TFRecordDataset(input_file)
        if is_training:
            dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=1000))
        dataset = dataset.apply(tf.data.experimental.map_and_batch(decode_record,
                                                                   batch_size=batch_size,
                                                                   num_parallel_calls=os.cpu_count() // 2))
        dataset = dataset.prefetch(1000)
        return dataset

    return input_fn


def create_model(bert_config, is_training, input_token_ids, sentiment_labels, input_mask, segment_ids,
                 target_token_ids, target_mask, target_start_ids, target_end_ids
                 , input_seq_length, mode, use_one_hot_embeddings):
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_token_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    label_dim = 20
    label_size = label_dim * 4
    label_hidden_size = bert_config.hidden_size
    batch_size = input_token_ids.shape[0]
    with tf.variable_scope("encoder_label_embedding"):
        # [label_size,hiddent_size]
        label_embedding_table = tf.get_variable("label_embedding_table", shape=[label_size, label_hidden_size]
                                                , initializer=tf.truncated_normal_initializer(stddev=0.02))
        sentiment_labels_one_hot = tf.one_hot(sentiment_labels, depth=label_size)  # [batch_size,label_dim,label_size]
        sentiment_labels_hidden = tf.matmul(tf.reshape(sentiment_labels_one_hot, shape=[-1, label_size]),
                                            label_embedding_table)  # [batch_size*label_dim,hiddent_size]
        sentiment_labels_hidden = tf.reduce_sum(tf.reshape(sentiment_labels_hidden, shape=[-1, label_dim, label_size]),
                                                axis=1)  # [batch_size,hiddent_size]

    with tf.variable_scope("encoder_label_bert_attention"):
        bert_final_state = model.get_sequence_output()  # [batch_size,length,hidden_size]
        sentiment_labels_hidden_length = tf.tile(tf.expand_dims(sentiment_labels_hidden, axis=1),
                                                 [1, input_seq_length, 1])
        encoder_state = bert_final_state + sentiment_labels_hidden_length

    with tf.variable_scope("decoder"):
        rnn_cell = [tf.nn.rnn_cell.LSTMCell(bert_config.hidden_size) for i in range(2)]
        mul_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_cell)
        # Helper
        if mode == tf.estimator.ModeKeys.TRAIN:
            target_embedding = tf.nn.embedding_lookup(model.embedding_table, target_token_ids)
            target_sequence_length = tf.count_nonzero(target_mask)
            helper = tf.contrib.seq2seq.TrainingHelper(input=target_embedding, sequence_length=target_sequence_length)
        elif mode == tf.estimator.ModeKeys.EVAL or mode == tf.estimator.ModeKeys.PREDICT:
            helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=model.embedding_table,
                                                              start_tokens=tf.tile([target_start_ids], [batch_size]),
                                                              end_token=target_end_ids)
        decoder = tf.contrib.seq2seq.BasicDecoder(
            cell=mul_rnn_cell,
            helper=helper,
            initial_state=encoder_state,
            output_layer=tf.layers.Dense(bert_config.vocab_size,
                                         activation=modeling.get_activation(bert_config.hidden_act)))
        outputs, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder=decoder,
            output_time_major=False,
            impute_finished=True,
            maximum_iterations=10)

    with tf.variable_scope("loss"):
        # [batch_size,length,hidden]
        logts = outputs.rnn_output
        sample_id = outputs.sample_id
        loss = tf.contrib.sequence_loss(logts, target_token_ids, target_mask, average_across_timesteps=True,
                                        average_across_batch=True)

    return loss, logts, sample_id


def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_one_hot_embeddings,
                     input_seq_length, target_seq_length, target_start_ids, target_end_ids):
    """Returns `model_fn` closure for Estimator."""

    def model_fn(features, labels, mode, params):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))
        sentiment_labels = features["sentiment_labels"]
        input_mask = features["input_mask"]
        input_token_ids = features["input_token_ids"]
        target_token_ids = features["target_token_ids"]
        target_mask = features["target_mask"]
        segment_ids = features["segment_ids"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        (loss, logts, sample_id) = create_model(bert_config, is_training, input_token_ids, sentiment_labels, input_mask,
                                                segment_ids,
                                                target_token_ids, target_mask, target_start_ids, target_end_ids
                                                , input_seq_length, mode, use_one_hot_embeddings)

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
            train_op = optimization.create_optimizer(
                loss, learning_rate, num_train_steps, num_warmup_steps
            )
            accuracy = tf.metrics.accuracy(target_token_ids, sample_id, weights=target_mask)
            tf.summary.scalar("accuracy", accuracy[1])
            output_spec = tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        elif mode == tf.estimator.ModeKeys.EVAL:
            accuracy = tf.metrics.accuracy(target_token_ids, sample_id)
            eval_metrics = {"accuracy": accuracy}
            tf.summary.scalar("accuracy", accuracy[1])
            output_spec = tf.estimator.EstimatorSpec(mode=mode, loss=loss
                                                     , eval_metric_ops=eval_metrics)

        else:
            output_spec = tf.estimator.EstimatorSpec(mode=mode, predictions={"predictions": sample_id})

        return output_spec

    return model_fn


def main(args):
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

    max_seq_length = FLAGS.max_seq_length

    # multi-GPU
    distribute = None
    # distribute=tf.contrib.distribute.MirroredStrategy(num_gpus=4)
    run_config = tf.estimator.RunConfig(
        model_dir=os.path.abspath(FLAGS.output_dir),
        save_summary_steps=10,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        train_distribute=distribute,
        eval_distribute=distribute
    )

    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        num_train_example = 3000000  # 300w
        num_train_steps = int(
            num_train_example / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)


    vocab_dict = tokenization.load_vocab(FLAGS.vocab_file)
    target_start_ids = vocab_dict["[CLS]"]
    target_end_ids = vocab_dict["[SEP]"]

    model_fn=model_fn_builder(bert_config,
                     init_checkpoint=os.path.abspath(FLAGS.init_checkpoint),
                     learning_rate=FLAGS.learning_rate,
                     num_train_steps=num_train_steps,
                     num_warmup_steps=num_warmup_steps,
                     use_one_hot_embeddings=False,  # when use tpu ,it's True
                     input_seq_length=FLAGS.max_seq_length,
                     target_seq_length=FLAGS.max_target_seq_length,
                     target_start_ids=target_start_ids,
                     target_end_ids=target_end_ids
                     )
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config
    )

    if FLAGS.do_train:
        tf.logging.info("***** Running training *****")

        train_file = os.path.join(os.path.abspath(FLAGS.data_dir), "sentiment_analysis_trainingset.tf_record")


        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)

        train_dataset_fn = file_based_input_fn_builder(train_file, max_seq_length,FLAGS.max_target_seq_length, is_training=True,
                                                       drop_remainder=True)
        estimator.train(input_fn=lambda: train_dataset_fn(FLAGS.train_batch_size),
                        max_steps=num_train_steps)



if __name__ == "__main__":
    tf.app.run()
