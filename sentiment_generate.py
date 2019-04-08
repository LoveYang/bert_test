# -*- coding: utf-8 -*-
import tensorflow as tf
import os

import modeling
import optimization
import tokenization
from tensorflow.python.util import nest

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
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
    "output_dir", "./token_test/output_sentiment_generate_withoutInit_dir",
    "The output directory where the model checkpoints will be written.")

## Other parameters

# "./chinese_L-12_H-768_A-12/bert_model.ckpt",
flags.DEFINE_string(
    "init_checkpoint", None,
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

flags.DEFINE_integer("predict_batch_size", 4, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 8,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 10000,
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
                                                                   num_parallel_calls=os.cpu_count() // 2,
                                                                   drop_remainder=drop_remainder))
        dataset = dataset.prefetch(1000)
        return dataset

    return input_fn


def create_model(bert_config, is_training, input_token_ids, sentiment_labels, input_mask, segment_ids,
                 target_token_ids, target_mask, target_start_ids, target_end_ids
                 , target_seq_length, mode, batch_size, use_one_hot_embeddings):
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_token_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)
    target_seq_length=target_seq_length-1
    if mode!=tf.estimator.ModeKeys.PREDICT:
        with tf.variable_scope("prepare_data"):
            output_token_ids=tf.slice(target_token_ids,[0,1],[-1,target_seq_length],name="output_token_ids")
            outputs_mask=tf.slice(target_mask,[0,1],[-1,target_seq_length],name="outputs_mask")
            target_token_ids=tf.slice(target_token_ids,[0,0],[-1,target_seq_length])


    label_dim = 20
    label_size = label_dim * 4
    label_hidden_size = bert_config.hidden_size
    tf.logging.info("lstm_decoder building..")
    with tf.variable_scope("encoder_label_embedding"):
        # [label_size,hiddent_size]
        label_embedding_table = tf.get_variable("label_embedding_table", shape=[label_size, label_hidden_size]
                                                , initializer=tf.truncated_normal_initializer(stddev=0.02))
        sentiment_labels_one_hot = tf.one_hot(sentiment_labels, depth=label_size)  # [batch_size,label_dim,label_size]
        sentiment_labels_hidden = tf.matmul(
            tf.reshape(sentiment_labels_one_hot, shape=[-1, label_size]),
            label_embedding_table)  # [batch_size*label_dim,hiddent_size]

        sentiment_labels_hidden = tf.reduce_sum(
            tf.reshape(sentiment_labels_hidden, shape=[-1, label_dim, label_hidden_size]),
            axis=1)  # [batch_size,hiddent_size]

    with tf.variable_scope("encoder_label_bert_attention"):
        bert_final_state = model.get_sequence_output()  # [batch_size,length,hidden_size]
        # sentiment_labels_hidden_length = tf.tile(tf.expand_dims(sentiment_labels_hidden, axis=1),
        #                                          [1, input_seq_length, 1])
        encoder_state = tf.reduce_sum(bert_final_state,axis=1) + sentiment_labels_hidden

    with tf.variable_scope("decoder"):
        beam_width=10

        rnn_cell = [tf.nn.rnn_cell.LSTMCell(bert_config.hidden_size) for i in range(2)]
        mul_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_cell)
        transformer2lstm_encode_state=tuple(tf.nn.rnn_cell.LSTMStateTuple(encoder_state,state_c_h[1]) for state_c_h in mul_rnn_cell.zero_state(batch_size,tf.float32))


        # Helper
        if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL :
            target_embedding = tf.nn.embedding_lookup(model.embedding_table, target_token_ids)
            target_sequence_length = tf.cast(tf.count_nonzero(target_mask,axis=1),dtype=tf.int32)
            helper = tf.contrib.seq2seq.TrainingHelper(inputs=target_embedding,
                                                       sequence_length=tf.tile(tf.constant([target_seq_length],shape=[1],dtype=tf.int32),[batch_size]) )

            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=mul_rnn_cell,
                helper=helper,
                initial_state=transformer2lstm_encode_state,
                output_layer=tf.layers.Dense(bert_config.vocab_size,
                                             activation=modeling.get_activation(bert_config.hidden_act)))
            outputs, state,_ = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder,
                output_time_major=False,
                impute_finished=True,
                maximum_iterations=target_seq_length)


        elif  mode == tf.estimator.ModeKeys.PREDICT:
            # helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=model.embedding_table,
            #                                                   start_tokens=tf.tile([target_start_ids], [batch_size]),
            #                                                   end_token=target_end_ids)
            # helper = tf.contrib.seq2seq.SampleEmbeddingHelper(embedding=model.embedding_table,
            #                                                   start_tokens=tf.tile([target_start_ids], [batch_size]),
            #                                                   end_token=target_end_ids,softmax_temperature=0.5)
            def state_tiled_batch(t):
                return tf.contrib.seq2seq.tile_batch(t,multiplier=beam_width)
            beam_search_state=tuple(  state_tiled_batch(single_lstm_initial_state)
            for single_lstm_initial_state in transformer2lstm_encode_state)
            decoder=tf.contrib.seq2seq.BeamSearchDecoder(
                cell=mul_rnn_cell,
                embedding=model.embedding_table,
                start_tokens=tf.tile([target_start_ids], [batch_size]),
                end_token=target_end_ids,
                initial_state=beam_search_state,
                beam_width=beam_width,
                output_layer=tf.layers.Dense(bert_config.vocab_size,
                                             activation=modeling.get_activation(bert_config.hidden_act))
            )



            outputs, state,_ = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder,
                output_time_major=False,
                impute_finished=False,
                maximum_iterations=target_seq_length)

    with tf.variable_scope("loss"):
        # [batch_size,length,hidden]
        if mode != tf.estimator.ModeKeys.PREDICT:
            logts = outputs.rnn_output
            sample_id = outputs.sample_id
            loss = tf.reduce_sum(
                tf.contrib.seq2seq.sequence_loss(logts, output_token_ids, tf.cast(outputs_mask, dtype=tf.float32),
                                                 average_across_timesteps=False,
                                                 average_across_batch=True))
            scores = None



        else:
            sample_id = outputs.predicted_ids
            logts = None
            loss = None
            scores=outputs.beam_search_decoder_output.scores
            output_token_ids=None
            outputs_mask=None


    return loss, logts, sample_id,scores,output_token_ids,outputs_mask

def create_model_lstm_attention(bert_config, is_training, input_token_ids, sentiment_labels, input_mask, segment_ids,
                 target_token_ids, target_mask, target_start_ids, target_end_ids
                 , input_seq_length,target_seq_length, mode, batch_size, use_one_hot_embeddings):
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_token_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    target_seq_length=target_seq_length-1
    if mode!=tf.estimator.ModeKeys.PREDICT:
        with tf.variable_scope("prepare_data"):
            output_token_ids=tf.slice(target_token_ids,[0,1],[-1,target_seq_length])
            outputs_mask=tf.slice(target_mask,[0,1],[-1,target_seq_length])
            target_token_ids=tf.slice(target_token_ids,[0,0],[-1,target_seq_length])

    label_dim = 20
    label_size = label_dim * 4
    label_hidden_size = bert_config.hidden_size
    tf.logging.info("lstm_attention_decoder building..")

    with tf.variable_scope("encoder_label_embedding"):
        # [label_size,hiddent_size]
        label_embedding_table = tf.get_variable("label_embedding_table", shape=[label_size, label_hidden_size]
                                                , initializer=tf.truncated_normal_initializer(stddev=0.02))
        sentiment_labels_one_hot = tf.one_hot(sentiment_labels, depth=label_size)  # [batch_size,label_dim,label_size]
        sentiment_labels_hidden = tf.matmul(
            tf.reshape(sentiment_labels_one_hot, shape=[batch_size * label_dim, label_size]),
            label_embedding_table)  # [batch_size*label_dim,hiddent_size]

        sentiment_labels_hidden = tf.reduce_sum(
            tf.reshape(sentiment_labels_hidden, shape=[batch_size, label_dim, label_hidden_size]),
            axis=1)  # [batch_size,hiddent_size]



    with tf.variable_scope("encoder_label_bert_attention"):
        input_sequence_length=tf.cast(tf.count_nonzero(input_mask,axis=1),dtype=tf.int32)
        bert_final_state = model.get_sequence_output()  # [batch_size,length,hidden_size]

        encoder_state_h = tf.div( tf.reduce_sum(bert_final_state*tf.reshape(tf.cast(input_mask,tf.float32),[-1,input_seq_length,1]),axis=1)
                                  ,tf.cast(tf.reshape(input_sequence_length,[-1,1]),tf.float32)+1e-8)
        encoder_state_seq=bert_final_state

    with tf.variable_scope("decoder"):

        beam_width=10
        rnn_layer_size=2


        rnn_cells = [tf.nn.rnn_cell.BasicLSTMCell(bert_config.hidden_size) for i in range(rnn_layer_size)]
        mul_rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_cells)
        # mul_rnn_cell=tf.nn.rnn_cell.BasicLSTMCell(bert_config.hidden_size)
        encoder_state=tuple(tf.nn.rnn_cell.LSTMStateTuple(_rnn_state.c,encoder_state_h)for _rnn_state in mul_rnn_cell.zero_state(batch_size,dtype=tf.float32))


        def _prepare_beam_search_decode_inputs(beam_width,memory,source_sequence_length,encoder_state):
            memory=tf.contrib.seq2seq.tile_batch(memory, multiplier=beam_width)
            source_sequence_length = tf.contrib.seq2seq.tile_batch(source_sequence_length, multiplier=beam_width)

            # if isinstance(encoder_state,tuple) or isinstance(encoder_state):
            #     #     deal multi_lstm state
            #     tmp=[]
            #     for lstm_state in encoder_state:
            #         assert isinstance(lstm_state,tf.nn.rnn_cell.LSTMStateTuple)
            #         tmp.append(tf.nn.rnn_cell.LSTMStateTuple(tf.contrib.seq2seq.tile_batch(lstm_state[0], multiplier=beam_width),
            #                                                  tf.contrib.seq2seq.tile_batch(lstm_state[1],multiplier=beam_width)))
            #         encoder_state = tuple(tmp)
            # else:
            encoder_state = tf.contrib.seq2seq.tile_batch(encoder_state, multiplier=beam_width)
            return memory,source_sequence_length,encoder_state

        if mode==tf.estimator.ModeKeys.PREDICT:
            encoder_state_seq, input_sequence_length, encoder_state = _prepare_beam_search_decode_inputs(beam_width,
                                                                                                         encoder_state_seq,
                                                                                                         input_sequence_length,
                                                                                                         encoder_state)


        print(input_seq_length,encoder_state_seq.get_shape().as_list())
        # attention_mechanisms=tf.contrib.seq2seq.LuongAttention(bert_config.hidden_size,encoder_state_seq,memory_sequence_length=input_sequence_length,scale=True)
        attention_mechanisms=tf.contrib.seq2seq.BahdanauAttention(bert_config.hidden_size,encoder_state_seq,memory_sequence_length=input_sequence_length)

        atten_rnn_cells=tf.contrib.seq2seq.AttentionWrapper(mul_rnn_cell,attention_mechanisms,
                                                            attention_layer_size=bert_config.hidden_size,
                                                            output_attention=True)

        cells=atten_rnn_cells
        if mode==tf.estimator.ModeKeys.PREDICT:
            decoder_initial_state = cells.zero_state(batch_size=batch_size * beam_width, dtype=tf.float32)
        else:
            decoder_initial_state = cells.zero_state(batch_size=batch_size , dtype=tf.float32)
        decoder_initial_state=decoder_initial_state.clone(cell_state=encoder_state)
        # decoder_initial_state = cells.zero_state(batch_size=batch_size, dtype=tf.float32)

        # decoder_initial_state=encoder_state

        encoder_embedding=model.get_embedding_table()
        # Helper
        def get_help_embedding(ids):
            target_embedding = tf.nn.embedding_lookup(encoder_embedding, ids)
            target_input=target_embedding+tf.reshape(sentiment_labels_hidden,[-1,1,label_hidden_size])
            return target_input
        if mode != tf.estimator.ModeKeys.PREDICT:




            helper = tf.contrib.seq2seq.TrainingHelper(inputs=get_help_embedding(target_token_ids),
                                                           sequence_length=tf.tile(
                                                               tf.constant([target_seq_length], shape=[1], dtype=tf.int32),
                                                               [batch_size]))

            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=cells,
                helper=helper,
                initial_state=decoder_initial_state,
                output_layer=tf.layers.Dense(bert_config.vocab_size,
                                             activation=modeling.get_activation(bert_config.hidden_act)))



        else:


            decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=cells,
                embedding=get_help_embedding,
                start_tokens=tf.tile([target_start_ids], [batch_size]),
                end_token=target_end_ids,
                initial_state=decoder_initial_state,
                beam_width=beam_width,
                output_layer=tf.layers.Dense(bert_config.vocab_size,
                                             activation=modeling.get_activation(bert_config.hidden_act))
            )

        outputs, state, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder=decoder,
            output_time_major=False,
            impute_finished=False,
            maximum_iterations=target_seq_length)

    with tf.variable_scope("loss"):
        # [batch_size,length,hidden]
        if mode == tf.estimator.ModeKeys.TRAIN:
            logts = outputs.rnn_output
            sample_id = outputs.sample_id
            loss = tf.reduce_sum(
                tf.contrib.seq2seq.sequence_loss(logts, output_token_ids, tf.cast(outputs_mask, dtype=tf.float32),
                                                 average_across_timesteps=False,
                                                 average_across_batch=True))
            scores = None
        elif mode == tf.estimator.ModeKeys.EVAL:
            scores=None
            logts = outputs.rnn_output
            sample_id = outputs.sample_id
            loss = tf.reduce_sum(
                tf.contrib.seq2seq.sequence_loss(logts, output_token_ids, tf.cast(outputs_mask, dtype=tf.float32),
                                                 average_across_timesteps=False,
                                                 average_across_batch=True))

        else:
            sample_id = outputs.predicted_ids
            logts = None
            loss = None
            scores = outputs.beam_search_decoder_output.scores
            output_token_ids=None
            outputs_mask=None

    return loss, logts, sample_id, scores,output_token_ids,outputs_mask

def create_model_seq2seq_lstm_attention(bert_config, is_training, input_token_ids, sentiment_labels, input_mask, segment_ids,
                 target_token_ids, target_mask, target_start_ids, target_end_ids
                 , input_seq_length,target_seq_length, mode, batch_size, use_one_hot_embeddings):
    # div the [SEP] char
    target_seq_length=target_seq_length-1
    if mode!=tf.estimator.ModeKeys.PREDICT:
        with tf.variable_scope("prepare_data"):
            output_token_ids=tf.slice(target_token_ids,[0,1],[-1,target_seq_length])
            outputs_mask=tf.slice(target_mask,[0,1],[-1,target_seq_length])
            target_token_ids=tf.slice(target_token_ids,[0,0],[-1,target_seq_length])


    vocab_size = bert_config.vocab_size
    with tf.variable_scope("embedding"):
        encoder_embedding=tf.get_variable("input_embeding",shape=[vocab_size,bert_config.hidden_size],dtype=tf.float32)
        encoder_emb_inp=tf.nn.embedding_lookup(encoder_embedding,input_token_ids)

    with tf.variable_scope("encoder"):
        encoder_rnn_size=2
        input_sequence_length=tf.cast(tf.count_nonzero(input_mask,axis=1),dtype=tf.int32)
        encoder_lstm_cells = [tf.nn.rnn_cell.BasicLSTMCell(bert_config.hidden_size) for i in range(encoder_rnn_size)]
        encoder_cells=tf.nn.rnn_cell.MultiRNNCell(encoder_lstm_cells)
        encoder_initial_state=encoder_cells.zero_state(batch_size,tf.float32)
        encoder_outputs,encoder_state=tf.nn.dynamic_rnn(encoder_cells,encoder_emb_inp,initial_state=encoder_initial_state,sequence_length=input_sequence_length)

    with tf.variable_scope("decoder"):
        dec_rnn_size = 2
        output_proj=tf.layers.Dense(vocab_size,activation=modeling.get_activation(bert_config.hidden_act))
        dec_lstm_cells = [tf.nn.rnn_cell.BasicLSTMCell(bert_config.hidden_size) for i in range(dec_rnn_size)]
        dec_cells = tf.nn.rnn_cell.MultiRNNCell(dec_lstm_cells)

        if mode==tf.estimator.ModeKeys.EVAL or mode==tf.estimator.ModeKeys.TRAIN:
            target_embedding = tf.nn.embedding_lookup(encoder_embedding, target_token_ids)

            helper = tf.contrib.seq2seq.TrainingHelper(inputs=target_embedding,
                                                       sequence_length=tf.tile(
                                                           tf.constant([target_seq_length], shape=[1], dtype=tf.int32),
                                                           [batch_size]))
            with tf.variable_scope("attention"):
                attention_mechanisms = tf.contrib.seq2seq.LuongAttention(bert_config.hidden_size, encoder_outputs,
                                                                         memory_sequence_length=input_sequence_length,
                                                                         scale=True)
                cells = tf.contrib.seq2seq.AttentionWrapper(dec_cells, attention_mechanisms,
                                                                      attention_layer_size=bert_config.hidden_size,
                                                                      output_attention=True)
            dec_init_state=cells.zero_state(batch_size=batch_size,dtype=tf.float32).clone(cell_state=encoder_state)
            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=cells,
                helper=helper,
                initial_state=dec_init_state,
                output_layer=output_proj)
            outputs, state, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder,
                output_time_major=False,
                impute_finished=False,
                maximum_iterations=target_seq_length)

            logts = outputs.rnn_output
            sample_id = outputs.sample_id
            loss = tf.reduce_sum(
                tf.contrib.seq2seq.sequence_loss(logts, output_token_ids, tf.cast(outputs_mask, dtype=tf.float32),
                                                 average_across_timesteps=False,
                                                 average_across_batch=True))
            scores = None



        else:
            beam_width=10
            enc_out_t=tf.contrib.seq2seq.tile_batch(encoder_outputs,beam_width)
            enc_state_t = tf.contrib.seq2seq.tile_batch(encoder_state, beam_width)
            enc_seq_len_t = tf.contrib.seq2seq.tile_batch(input_sequence_length, beam_width)
            with tf.variable_scope("attention"):
                attention_mechanisms = tf.contrib.seq2seq.LuongAttention(bert_config.hidden_size, enc_out_t,
                                                                         memory_sequence_length=enc_seq_len_t,
                                                                         scale=True)
                cells = tf.contrib.seq2seq.AttentionWrapper(dec_cells, attention_mechanisms,
                                                                      attention_layer_size=bert_config.hidden_size,
                                                                      output_attention=True)
            dec_init_state=cells.zero_state(batch_size=batch_size*beam_width,dtype=tf.float32).clone(cell_state=enc_state_t)
            decoder=tf.contrib.seq2seq.BeamSearchDecoder(
                cell=cells,
                embedding=encoder_embedding,
                start_tokens=tf.tile(tf.constant([target_start_ids],tf.int32),[batch_size]),
                end_token=target_end_ids,
                beam_width=beam_width,
                initial_state=dec_init_state,
                output_layer=output_proj
            )
            outputs, state, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder,
                output_time_major=False,
                impute_finished=False,
                maximum_iterations=target_seq_length)
            if mode==tf.estimator.ModeKeys.EVAL:
                logts=None
                scores=None
                loss=tf.constant(0,dtype=tf.float32)
                sample_id = tf.squeeze(tf.slice(outputs.predicted_ids,[0,0,0],[-1,-1,1]),axis=-1)
                sample_id=tf.pad(sample_id,)
                print(sample_id)
            else:
                logts=None
                scores=None
                loss=tf.constant(0,dtype=tf.float32)
                sample_id = outputs.predicted_ids
                output_token_ids=None
                outputs_mask=None



    return loss, logts, sample_id, scores,output_token_ids,outputs_mask

def create_model_seq2seq_lstm_attention_with_condition(bert_config, is_training, input_token_ids, sentiment_labels, input_mask, segment_ids,
                 target_token_ids, target_mask, target_start_ids, target_end_ids
                 , input_seq_length,target_seq_length, mode, batch_size, use_one_hot_embeddings):
    # div the [SEP] char
    target_seq_length=target_seq_length-1
    if mode!=tf.estimator.ModeKeys.PREDICT:
        with tf.variable_scope("prepare_data"):
            output_token_ids=tf.slice(target_token_ids,[0,1],[-1,target_seq_length])
            outputs_mask=tf.slice(target_mask,[0,1],[-1,target_seq_length])
            target_token_ids=tf.slice(target_token_ids,[0,0],[-1,target_seq_length])


    vocab_size = bert_config.vocab_size
    with tf.variable_scope("embedding"):
        encoder_embedding=tf.get_variable("input_embeding",shape=[vocab_size,bert_config.hidden_size],dtype=tf.float32)
        encoder_emb_inp=tf.nn.embedding_lookup(encoder_embedding,input_token_ids)

    with tf.variable_scope("encoder"):
        encoder_rnn_size=2
        input_sequence_length=tf.cast(tf.count_nonzero(input_mask,axis=1),dtype=tf.int32)
        encoder_lstm_cells = [tf.nn.rnn_cell.BasicLSTMCell(bert_config.hidden_size) for i in range(encoder_rnn_size)]
        encoder_cells=tf.nn.rnn_cell.MultiRNNCell(encoder_lstm_cells)
        encoder_initial_state=encoder_cells.zero_state(batch_size,tf.float32)
        encoder_outputs,encoder_state=tf.nn.dynamic_rnn(encoder_cells,encoder_emb_inp,initial_state=encoder_initial_state,sequence_length=input_sequence_length)
        print("encoder",encoder_outputs)

    label_dim = 20
    label_size = label_dim * 4
    label_hidden_size = bert_config.hidden_size
    with tf.variable_scope("decoder_label_embedding"):
        # [label_size,hiddent_size]
        label_embedding_table = tf.get_variable("label_embedding_table", shape=[label_size, label_hidden_size]
                                                , initializer=tf.truncated_normal_initializer(stddev=0.02))
        sentiment_labels_one_hot = tf.one_hot(sentiment_labels, depth=label_size)  # [batch_size,label_dim,label_size]
        sentiment_labels_hidden = tf.matmul(
            tf.reshape(sentiment_labels_one_hot, shape=[-1, label_size]),
            label_embedding_table)  # [batch_size*label_dim,hiddent_size]

        sentiment_labels_hidden = tf.reduce_sum(
            tf.reshape(sentiment_labels_hidden, shape=[-1, label_dim, label_hidden_size]),
            axis=1)  # [batch_size,hiddent_size]



    with tf.variable_scope("decoder"):
        dec_rnn_size = 2
        output_proj=tf.layers.Dense(vocab_size,activation=modeling.get_activation(bert_config.hidden_act))
        dec_lstm_cells = [tf.nn.rnn_cell.BasicLSTMCell(bert_config.hidden_size) for i in range(dec_rnn_size)]
        dec_cells = tf.nn.rnn_cell.MultiRNNCell(dec_lstm_cells)
        sentiment_labels_hidden = tf.reshape(sentiment_labels_hidden, [-1, 1, label_hidden_size])

        if mode==tf.estimator.ModeKeys.EVAL or mode==tf.estimator.ModeKeys.TRAIN:
            target_embedding = tf.nn.embedding_lookup(encoder_embedding, target_token_ids)+sentiment_labels_hidden
            if mode==tf.estimator.ModeKeys.TRAIN:
                helper = tf.contrib.seq2seq.TrainingHelper(inputs=target_embedding,
                                                           sequence_length=tf.tile(
                                                               tf.constant([target_seq_length], shape=[1], dtype=tf.int32),
                                                               [batch_size]))


            with tf.variable_scope("attention"):
                attention_mechanisms = tf.contrib.seq2seq.LuongAttention(bert_config.hidden_size, encoder_outputs,
                                                                         memory_sequence_length=input_sequence_length,
                                                                         scale=True)
                cells = tf.contrib.seq2seq.AttentionWrapper(dec_cells, attention_mechanisms,
                                                                      attention_layer_size=bert_config.hidden_size,
                                                                      output_attention=True)
            dec_init_state=cells.zero_state(batch_size=batch_size,dtype=tf.float32).clone(cell_state=encoder_state)
            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=cells,
                helper=helper,
                initial_state=dec_init_state,
                output_layer=output_proj)
            outputs, state, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder,
                output_time_major=False,
                impute_finished=False,
                maximum_iterations=target_seq_length)

            logts = outputs.rnn_output
            sample_id = outputs.sample_id
            loss = tf.reduce_sum(
                tf.contrib.seq2seq.sequence_loss(logts, output_token_ids, tf.cast(outputs_mask, dtype=tf.float32),
                                                 average_across_timesteps=False,
                                                 average_across_batch=True))
            scores = None



        else:
            beam_width=10
            enc_out_t=tf.contrib.seq2seq.tile_batch(encoder_outputs,beam_width)
            enc_state_t = tf.contrib.seq2seq.tile_batch(encoder_state, beam_width)
            enc_seq_len_t = tf.contrib.seq2seq.tile_batch(input_sequence_length, beam_width)
            with tf.variable_scope("attention"):
                attention_mechanisms = tf.contrib.seq2seq.LuongAttention(bert_config.hidden_size, enc_out_t,
                                                                         memory_sequence_length=enc_seq_len_t,
                                                                         scale=True)
                cells = tf.contrib.seq2seq.AttentionWrapper(dec_cells, attention_mechanisms,
                                                                      attention_layer_size=bert_config.hidden_size,
                                                                      output_attention=True)
            dec_init_state=cells.zero_state(batch_size=batch_size*beam_width,dtype=tf.float32).clone(cell_state=enc_state_t)
            decoder=tf.contrib.seq2seq.BeamSearchDecoder(
                cell=cells,
                embedding=lambda ids:tf.nn.embedding_lookup(encoder_embedding, ids)+sentiment_labels_hidden,
                start_tokens=tf.tile(tf.constant([target_start_ids],tf.int32),[batch_size]),
                end_token=target_end_ids,
                beam_width=beam_width,
                initial_state=dec_init_state,
                output_layer=output_proj
            )
            outputs, state, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder,
                output_time_major=False,
                impute_finished=False,
                maximum_iterations=target_seq_length)
            if mode==tf.estimator.ModeKeys.EVAL:
                logts=None
                scores=None
                loss=tf.constant(0,dtype=tf.float32)
                sample_id = tf.squeeze(tf.slice(outputs.predicted_ids,[0,0,0],[-1,-1,1]),axis=-1)
                sample_id=tf.pad(sample_id,)
                print(sample_id)
            else:
                logts=None
                scores=None
                loss=tf.constant(0,dtype=tf.float32)
                sample_id = outputs.predicted_ids
                output_token_ids=None
                outputs_mask=None



    return loss, logts, sample_id, scores,output_token_ids,outputs_mask



def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_one_hot_embeddings,
                     input_seq_length, target_seq_length, target_start_ids, target_end_ids, batch_size,mode_type="lstm"):
    """Returns `model_fn` closure for Estimator."""
    mode_type=mode_type.lower()
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
        tf.logging.info(input_token_ids.shape)
        is_training = (mode == tf.estimator.ModeKeys.TRAIN)
        if mode_type=="lstm_attention":
            #lstm-attention
            (loss, logts, sample_id,scores,output_token_ids,outputs_mask) = create_model_lstm_attention(bert_config, is_training, input_token_ids, sentiment_labels, input_mask,
                                                    segment_ids,
                                                    target_token_ids, target_mask, target_start_ids, target_end_ids
                                                    , input_seq_length,target_seq_length, mode, batch_size, use_one_hot_embeddings)
        elif mode_type=="seq2seq_lstm_attention":
            (loss, logts, sample_id,scores,output_token_ids,outputs_mask) = create_model_seq2seq_lstm_attention(bert_config, is_training, input_token_ids, sentiment_labels, input_mask,
                                                    segment_ids,
                                                    target_token_ids, target_mask, target_start_ids, target_end_ids
                                                    , input_seq_length,target_seq_length, mode, batch_size, use_one_hot_embeddings)


        elif mode_type=="lstm":
            #lstm-nonattention
            (loss, logts, sample_id,scores,output_token_ids,outputs_mask) = create_model(bert_config, is_training, input_token_ids, sentiment_labels, input_mask,
                                                    segment_ids,
                                                    target_token_ids, target_mask, target_start_ids, target_end_ids
                                                    , target_seq_length, mode, batch_size, use_one_hot_embeddings)
        elif mode_type=="seq2seq_lstm_attention_with_condition":
            (loss, logts, sample_id,scores,output_token_ids,outputs_mask) = create_model_seq2seq_lstm_attention_with_condition(bert_config, is_training, input_token_ids, sentiment_labels, input_mask,
                                                    segment_ids,
                                                    target_token_ids, target_mask, target_start_ids, target_end_ids
                                                    , input_seq_length,target_seq_length, mode, batch_size, use_one_hot_embeddings)

        else:
            raise TypeError("None type with {} in ['lstm','lstm_attention']".format(mode_type))

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
            accuracy = tf.metrics.accuracy(output_token_ids, sample_id, weights=outputs_mask)
            tf.summary.scalar("accuracy_train", accuracy[1])
            output_spec = tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        elif mode == tf.estimator.ModeKeys.EVAL:

            accuracy = tf.metrics.accuracy(output_token_ids,sample_id , weights=outputs_mask)
            eval_metrics = {"accuracy": accuracy}
            tf.summary.scalar("accuracy_eval", accuracy[1])
            output_spec = tf.estimator.EstimatorSpec(mode=mode, loss=loss
                                                     , eval_metric_ops=eval_metrics)

        else:
            predictions={"sample_id": sample_id,"inputs":input_token_ids}
            if scores is not None:
                predictions["scores"]=scores

            output_spec = tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        return output_spec

    return model_fn




def test_input_fn():
    import numpy as np
    from tokenization import FullTokenizer
    file1="/home/hesheng/python_project/bert/token_test/data/data_sentiment_analysis/sentiment_analysis_trainingset.tf_record"
    file2="/home/hesheng/python_project/bert/token_test/data/data_sentiment_analysis/sentiment_analysis_validationset.tf_record"
    file3="/home/hesheng/python_project/bert/token_test/data/data_sentiment_analysis/predict.tf_record"
    fulltoken=FullTokenizer("/home/hesheng/python_project/bert/chinese_L-12_H-768_A-12/vocab.txt")

    fn=file_based_input_fn_builder(file3,128,10,is_training=False,drop_remainder=True)
    dataset=fn(1)
    iterator=dataset.make_initializable_iterator()
    element=iterator.get_next()
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        i=0
        while True:
            i+=1
            feature=sess.run(element)
            # print("step:{0},target_mask :\n{1}".format(i,feature["input_token_ids"].shape))
            print(fulltoken.convert_ids_to_tokens(np.squeeze(feature["input_token_ids"])))
            print(fulltoken.convert_ids_to_tokens(np.squeeze(feature["target_token_ids"])))

            break

PARAMS = {
    'embed_dims': 15,
    'rnn_size': 50,
    'num_layers': 1,
    'beam_width': 5,
    'clip_norm': 5.0,
    'batch_size': 128,
    'n_epochs': 60,
    'src_char2idx':0,
    'tgt_char2idx':1

}

def clip_grads(loss):
    variables = tf.trainable_variables()
    grads = tf.gradients(loss, variables)
    clipped_grads, _ = tf.clip_by_global_norm(grads, PARAMS['clip_norm'])
    return zip(clipped_grads, variables)


def rnn_cell():
    def cell_fn():
        cell = tf.nn.rnn_cell.GRUCell(PARAMS['rnn_size'],
                                      kernel_initializer=tf.orthogonal_initializer())
        return cell

    return tf.nn.rnn_cell.MultiRNNCell([cell_fn() for _ in range(PARAMS['num_layers'])])


def dec_cell(enc_out, enc_seq_len):
    attention = tf.contrib.seq2seq.BahdanauAttention(
        num_units=PARAMS['rnn_size'],
        memory=enc_out,
        memory_sequence_length=enc_seq_len)

    return tf.contrib.seq2seq.AttentionWrapper(
        cell=rnn_cell(),
        attention_mechanism=attention,
        attention_layer_size=PARAMS['rnn_size'])


def dec_input(labels):
    x = tf.fill([tf.shape(labels)[0], 1], PARAMS['tgt_char2idx']['<GO>'])
    x = tf.to_int32(x)
    return tf.concat([x, labels[:, :-1]], 1)


def forward(features, labels, mode):
    inputs=features["input_token_ids"]
    inputs_mask=features["input_mask"]
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    enc_seq_len = tf.count_nonzero(inputs_mask, 1, dtype=tf.int32)
    batch_sz = tf.shape(inputs)[0]
    vocab_size=100
    with tf.variable_scope('Encoder'):
        embedding = tf.get_variable('lookup_table',
                                    [vocab_size, PARAMS['embed_dims']])
        x = tf.nn.embedding_lookup(embedding, inputs)
        enc_out, enc_state = tf.nn.dynamic_rnn(rnn_cell(), x, enc_seq_len, dtype=tf.float32)

    with tf.variable_scope('Decoder'):
        output_proj = tf.layers.Dense(vocab_size)

        if is_training:
            cell = dec_cell(enc_out, enc_seq_len)
            dec_seq_len = tf.count_nonzero(labels, 1, dtype=tf.int32)

            init_state = cell.zero_state(batch_sz, tf.float32).clone(
                cell_state=enc_state)

            helper = tf.contrib.seq2seq.TrainingHelper(
                inputs=tf.nn.embedding_lookup(embedding, dec_input(labels)),
                sequence_length=dec_seq_len)
            decoder = tf.contrib.seq2seq.BasicDecoder(
                cell=cell,
                helper=helper,
                initial_state=init_state,
                output_layer=output_proj)
            decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder,
                maximum_iterations=tf.reduce_max(dec_seq_len))

            return decoder_output.rnn_output
        else:
            enc_out_t = tf.contrib.seq2seq.tile_batch(enc_out, PARAMS['beam_width'])
            enc_state_t = tf.contrib.seq2seq.tile_batch(enc_state, PARAMS['beam_width'])
            enc_seq_len_t = tf.contrib.seq2seq.tile_batch(enc_seq_len, PARAMS['beam_width'])

            cell = dec_cell(enc_out_t, enc_seq_len_t)

            init_state = cell.zero_state(batch_sz * PARAMS['beam_width'], tf.float32).clone(
                cell_state=enc_state_t)

            decoder = tf.contrib.seq2seq.BeamSearchDecoder(
                cell=cell,
                embedding=embedding,
                start_tokens=tf.tile(tf.constant([0], tf.int32),
                                     [batch_sz]),
                end_token=1,
                initial_state=init_state,
                beam_width=PARAMS['beam_width'],
                output_layer=output_proj)
            decoder_output, _, _ = tf.contrib.seq2seq.dynamic_decode(
                decoder=decoder)

            return decoder_output.predicted_ids[:, :, 0]
def model_fn_test(features, labels, mode):
    logits_or_ids = forward(features, labels, mode)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=logits_or_ids)

    if mode == tf.estimator.ModeKeys.TRAIN:
        loss_op = tf.contrib.seq2seq.sequence_loss(logits=logits_or_ids,
                                                   targets=labels,
                                                   weights=tf.to_float(tf.sign(labels)))
        train_op = tf.train.AdamOptimizer().apply_gradients(
            clip_grads(loss_op),
            global_step=tf.train.get_global_step())

        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss_op, train_op=train_op)
    if mode== tf.estimator.ModeKeys.EVAL:
        loss_op = tf.contrib.seq2seq.sequence_loss(logits=logits_or_ids,
                                                   targets=labels,
                                                   weights=tf.to_float(tf.ones_like))
        return tf.estimator.EstimatorSpec(
            mode=mode, loss=loss_op)



def main(args):
    from tokenization import FullTokenizer
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
    mode_types=['lstm','seq2seq_lstm_attention',"seq2seq_lstm_attention_with_condition",'lstm_attention','transformer']
    mode_type=mode_types[3]
    output_dir=os.path.abspath(FLAGS.output_dir)

    output_dir='_'.join([output_dir,mode_type])
    if not tf.gfile.Exists(output_dir):
        tf.gfile.MakeDirs(output_dir)
    tf.logging.info("output_dir:{}".format(output_dir))

    max_seq_length = FLAGS.max_seq_length
    # multi-GPU
    distribute = None
    num_gpus=len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
    tf.logging.info("num_gpus is {}".format(num_gpus))
    distribute=tf.contrib.distribute.MirroredStrategy(num_gpus=num_gpus)
    run_config = tf.estimator.RunConfig(
        model_dir=os.path.abspath(output_dir),
        save_summary_steps=200,
        keep_checkpoint_max=2,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        train_distribute=distribute,
        eval_distribute=distribute
    )

    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        num_train_example = 4702806  # 300w
        num_train_steps = int(
            num_train_example / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    fulltoken=FullTokenizer(FLAGS.vocab_file)
    vocab_dict = fulltoken.vocab

    target_start_ids = vocab_dict["[CLS]"]
    target_end_ids = vocab_dict["[SEP]"]

    if FLAGS.do_train:
        batch_size = FLAGS.train_batch_size
    elif FLAGS.do_eval:
        batch_size = FLAGS.eval_batch_size
    else:
        batch_size = FLAGS.predict_batch_size

    model_fn = model_fn_builder(bert_config,
                                init_checkpoint=os.path.abspath(FLAGS.init_checkpoint) if FLAGS.init_checkpoint is not None else None,
                                learning_rate=FLAGS.learning_rate,
                                num_train_steps=num_train_steps,
                                num_warmup_steps=num_warmup_steps,
                                use_one_hot_embeddings=False,  # when use tpu ,it's True
                                input_seq_length=FLAGS.max_seq_length,
                                target_seq_length=FLAGS.max_target_seq_length,
                                target_start_ids=target_start_ids,
                                target_end_ids=target_end_ids,
                                batch_size=batch_size,
                                mode_type=mode_type
                                )
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config
    )

    if FLAGS.do_train:
        tf.logging.info("***** Running training *****")

        train_file = os.path.join(os.path.abspath(FLAGS.data_dir), "sentiment_analysis_trainingset.tf_record")
        # train_file=os.path.join(os.path.abspath(FLAGS.data_dir), "predict.tf_record")
        eval_file = os.path.join(os.path.abspath(FLAGS.data_dir), "sentiment_analysis_validationset.tf_record")
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("  Num steps = %d", num_train_steps)

        train_dataset_fn = file_based_input_fn_builder(train_file, max_seq_length, FLAGS.max_target_seq_length,
                                                       is_training=True,
                                                       drop_remainder=True)
        eval_dataset_fn = file_based_input_fn_builder(eval_file, max_seq_length,FLAGS.max_target_seq_length, is_training=False,
                                                      drop_remainder=True)

        tf.estimator.train_and_evaluate(
            estimator,
            train_spec=tf.estimator.TrainSpec(input_fn=lambda :train_dataset_fn(batch_size),max_steps=num_train_steps),
            eval_spec=tf.estimator.EvalSpec(input_fn=lambda: eval_dataset_fn(batch_size),steps=100)
        )
    if FLAGS.do_eval:
        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

        eval_file= os.path.join(os.path.abspath(FLAGS.data_dir), "sentiment_analysis_validationset.tf_record")
        # eval_file=os.path.join(os.path.abspath(FLAGS.data_dir), "sentiment_analysis_trainingset.tf_record")
        eval_dataset_fn = file_based_input_fn_builder(eval_file, max_seq_length,FLAGS.max_target_seq_length, is_training=False,
                                                      drop_remainder=False)
        result = estimator.evaluate(input_fn=lambda: eval_dataset_fn(FLAGS.eval_batch_size),
                                    steps=1000)
        output_eval_file = os.path.join(os.path.abspath(output_dir), "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if FLAGS.do_predict:
        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)
        predict_file=os.path.join(os.path.abspath(FLAGS.data_dir), "predict.tf_record")
        # predict_file=os.path.join(os.path.abspath(FLAGS.data_dir), "sentiment_analysis_validationset.tf_record")
        predict_dataset_fn = file_based_input_fn_builder(predict_file, max_seq_length,FLAGS.max_target_seq_length, is_training=False,
                                                         drop_remainder=True)
        result = estimator.predict(input_fn=lambda: predict_dataset_fn(batch_size))
        output_predict_file = os.path.join(output_dir, "test_results.tsv")
        tf.logging.info(result)
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            tf.logging.info("***** Predict results *****")
            for prediction in result:
                sample_id=prediction['sample_id'][:,:3].T.tolist()
                if "scores" in prediction:
                    scores=prediction['scores']
                    print(scores)
                input=prediction['inputs'].tolist()
                for token in sample_id:
                    print(fulltoken.convert_ids_to_tokens(token))
                print(fulltoken.convert_ids_to_tokens(input) )
                print("\n")

                # writer.write(output_line)


if __name__ == "__main__":
    tf.app.run()
    # test_input_fn()