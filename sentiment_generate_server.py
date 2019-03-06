#!/usr/bin/python3
"""
Created on 19-2-13
@project:bert 
@ide:PyCharm
@author: hesheng
"""
# from flask.app import
import os
import tensorflow as tf
import modeling
from sentiment_generate import model_fn_builder
from tokenization import FullTokenizer

class ModelServer:
    def __init__(self,param):

        self.model_path=os.path.abspath(param["model_path"])
        self.bert_config_file = os.path.abspath(param["bert_config_file"])
        bert_config = modeling.BertConfig.from_json_file(self.bert_config_file)
        self.fulltoken = FullTokenizer(os.path.abspath(param["vocab_file"]))
        self.vocab_dict = self.fulltoken.vocab

        target_start_ids = self.vocab_dict["[CLS]"]
        target_end_ids = self.vocab_dict["[SEP]"]

        num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(','))
        tf.logging.info("num_gpus is {}".format(num_gpus))
        if param["use_mul_gpu"] :
            distribute = tf.contrib.distribute.MirroredStrategy(num_gpus=num_gpus)
        else:
            distribute=None
        run_config = tf.estimator.RunConfig(
            model_dir=os.path.abspath(self.model_path),
            save_summary_steps=200,
            keep_checkpoint_max=2,
            save_checkpoints_steps=3000,
            train_distribute=distribute,
            eval_distribute=distribute
        )
        self.input_max_seq_length=param["max_seq_length"]
        model_fn = model_fn_builder(bert_config,
                                    init_checkpoint=None,
                                    learning_rate=0.0001,
                                    num_train_steps=10000,
                                    num_warmup_steps=100,
                                    use_one_hot_embeddings=False,  # when use tpu ,it's True
                                    input_seq_length=param["max_seq_length"],
                                    target_seq_length=param["max_target_seq_length"],
                                    target_start_ids=target_start_ids,
                                    target_end_ids=target_end_ids,
                                    batch_size=param["batch_size"],
                                    mode_type=param["mode_type"]
                                    )
        self.estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            config=run_config
        )

    #input:[(str_mask_tokens,str_labels),list_str_mask_words]
    #label 0:Not mentioned,
    #   1:Negative,
    #   2:Neutral,
    #   3:Positive
    def predict(self,inputs,limitNum=3):
        predicts=[]
        if not isinstance(inputs,list):
            inputs=[inputs]
        def token_input():
            for input in inputs:
                tokens=input[0]
                labels=[int(label)for label in input[1]][:20]
                mask_words=input[2]
                assert max(labels) < 4 and min(labels) >= 0
                tokens=self.fulltoken.tokenize(tokens)[:self.input_max_seq_length-2]
                def replace_Mask(tokens,mask_words):
                    mask_index=[]
                    first_maskwords=[x[0] for x in mask_words]

                    for index,token in enumerate(tokens):
                        if token in first_maskwords:
                            for mask_words_x in mask_words:
                                if token==mask_words_x[0]:
                                    _token="".join([_t.replace("#",'')for _t in tokens[index:index+len(mask_words_x)]])
                                    if _token==mask_words_x:
                                        for i in range(len(mask_words_x)):
                                            mask_index.append(index+i)
                                        mask_words=[x_ for x_ in  mask_words if x_!=mask_words_x]
                                        first_maskwords = [x[0] for x in mask_words]
                        if len(mask_words)<1:
                            break
                    for mask_index_ in mask_index:
                        tokens[mask_index_]='[MASK]'
                    return tokens

                tokens=replace_Mask(tokens,mask_words)
                ids=self.fulltoken.convert_tokens_to_ids(['[CLS]']+tokens+['[SEP]'])
                input_mask=[1]*len(ids)
                segment_ids=[0]*self.input_max_seq_length
                while len(ids)<self.input_max_seq_length:
                    ids.append(0)
                    input_mask.append(0)
                while len(labels)<20:
                    labels.append(0)

                yield ([ids],[input_mask],[labels],[segment_ids])

        def input_fn():

            dataset=tf.data.Dataset.from_generator(token_input,(tf.int64,tf.int64,tf.int64,tf.int64)
                                                   ,output_shapes=(tf.TensorShape([None,self.input_max_seq_length]),
                                                                   tf.TensorShape([None, self.input_max_seq_length]),
                                                                   tf.TensorShape([None,20]),
                                                                   tf.TensorShape([None, self.input_max_seq_length])))
            dataset=dataset.map(lambda ids,input_mask,labels,segment_ids:{
            "sentiment_labels": labels,
            "input_token_ids": ids,
            "input_mask": input_mask,
            "target_token_ids": tf.zeros_like([1,1]),
            "target_mask": tf.zeros_like([1,1]),
            "segment_ids": segment_ids})

            # (ids, input_mask, labels, segment_ids)=dataset
            # features={
            #     "sentiment_labels": labels,
            #     "input_token_ids": ids,
            #     "input_mask": input_mask,
            #     "target_token_ids": tf.zeros_like([1, 1]),
            #     "target_mask": tf.zeros_like([1, 1]),
            #     "segment_ids": segment_ids}
            #
            # return features

            return dataset

        result=self.estimator.predict(input_fn=input_fn)
        for prediction in result:
            sample_id = prediction['sample_id'][:, :limitNum].T.tolist()
            ans=[]
            for sample_id_ in sample_id:
                token=self.fulltoken.convert_ids_to_tokens(sample_id_)
                ans.append("".join(token[:-1]))
            predicts.append(ans)
            input = prediction['inputs'].tolist()
            print(self.fulltoken.convert_ids_to_tokens(input))


        return predicts

class Label:
    def __init__(self,locations=None,services=None,prices=None,environments=None,dishes=None,others=None):
        self.location(locations)
        self.service(services)
        self.price(prices)
        self.environment(environments)
        self.dish(dishes)
        self.other(others)

    def _meta2str(self,meta):
        return "".join(meta)

    def labels(self):
        return self._meta2str(self.locations)+self._meta2str(self.services)+self._meta2str(self.prices)+ \
               self._meta2str(self.environments)+self._meta2str(self.dishes)+self._meta2str(self.others)

    def _limit_score(self,meta):
        if meta is None or len(meta)==0:
            meta=[0]
        assert max(meta)<4 and min(meta)>=0
        return meta
    def _meta_default(self,meta,limitNum):
        meta=self._limit_score(meta)
        meta="".join(str(meta_)for meta_ in meta)
        while len(meta)<limitNum:
            meta+="0"
        return meta

    def dish(self,dishes):
        self.dishes=self._meta_default(dishes,4)


    def other(self,others):
        self.others=self._meta_default(others,2)

    def price(self,price):
        self.prices=self._meta_default(price,3)

    def environment(self,environment):
        self.environments=self._meta_default(environment,4)


    def service(self, service):
        self.services = self._meta_default(service, 4)

    def location(self,location):
        self.locations=self._meta_default(location,3)


def server_seq2seq_lstm_attention():
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    model_param={
        "use_mul_gpu":False,
        "model_path":"./token_test/output_sentiment_generate_dir_seq2seq_lstm_attention",
        "bert_config_file":"./chinese_L-12_H-768_A-12/bert_config.json",
        "vocab_file":"./chinese_L-12_H-768_A-12/vocab.txt",
        "max_seq_length":128,
        "max_target_seq_length":10,
        "batch_size":1,
        "mode_type":"seq2seq_lstm_attention"

    }
    labels=Label(services=[1,1,1,1])
    negitive=[Label(locations=[1,1,1]),Label(services=[1,1,1,1]),Label(prices=[1,1,1]),Label(environments=[1,1,1,1]),Label(dishes=[1,1,1,1]),Label(others=[1,1])]
    postive=[Label(locations=[3,3,3]),Label(services=[3,3,3,3]),Label(prices=[3,3,3]),Label(environments=[3,3,3,3]),Label(dishes=[3,3,3,3]),Label(others=[3,3])]
    neutral=[Label(locations=[2,2,2]),Label(services=[2,2,2,2]),Label(prices=[2,2,2]),Label(environments=[2,2,2,2]),Label(dishes=[2,2,2,2]),Label(others=[2,2])]
    model=ModelServer(model_param)
    '[MASK]'
    input_content="""幸运随点评团体验霸王餐，心情好~蜀九香刚进驻泉州不久，招牌大名气响，以至于刚到店门口的我被门廊密密麻麻排队取号等候桌位的食客们的热情吓到了！！
        整体感觉:1.装修布置很新，新开业嘛！不管是餐位桌椅，隔断墙，仿古门窗，卫生间，都干干净净古色古香，还有民乐师演奏。2.服务超级赞，尽管人很多！
        刚说被楼下排队的人数吓到了，结果食客们都悠然嗑着瓜子唠着嗑，喝着果汁吹着风…足见店家礼遇；再说我们在楼上吃的时候，一位阿姨忙前忙后谦恭有礼，加高汤填茶水调火力送纸巾，提醒个人物品…真的很贴心！
        听口音很有可能来自川蜀地区，为来自远方的高质量服务人员点赞！3.细节取胜！大屏幕作为背景墙，上演琵琶曲，古筝曲，嘈嘈切切错杂弹，真是与店内装修风格相得益彰；
        还有在卫生间看到了各种温馨提示；再比如结账门口预备的去火去辣的清凉薄荷糖圈…总之，你，值得拥有！"""

    # inputs=[(input_content,labels.labels(),["密密麻麻"])]
    label_list=negitive+postive+neutral+[Label()]
    inputs=[(input_content,la_.labels(),["进驻泉州"]) for la_ in label_list]
    res=model.predict(inputs)
    for res_ in res:
        print(res_)

def server_lstm():
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    model_param={
        "use_mul_gpu":False,
        "model_path":"./token_test/output_sentiment_generate_dir_lstm",
        "bert_config_file":"./chinese_L-12_H-768_A-12/bert_config.json",
        "vocab_file":"./chinese_L-12_H-768_A-12/vocab.txt",
        "max_seq_length":128,
        "max_target_seq_length":10,
        "batch_size":1,
        "mode_type":"lstm"

    }
    labels=Label(services=[1,1,1,1])
    negitive=[Label(locations=[1,1,1]),Label(services=[1,1,1,1]),Label(prices=[1,1,1]),Label(environments=[1,1,1,1]),Label(dishes=[1,1,1,1]),Label(others=[1,1])]
    postive=[Label(locations=[3,3,3]),Label(services=[3,3,3,3]),Label(prices=[3,3,3]),Label(environments=[3,3,3,3]),Label(dishes=[3,3,3,3]),Label(others=[3,3])]
    neutral=[Label(locations=[2,2,2]),Label(services=[2,2,2,2]),Label(prices=[2,2,2]),Label(environments=[2,2,2,2]),Label(dishes=[2,2,2,2]),Label(others=[2,2])]
    model=ModelServer(model_param)
    '[MASK]'
    input_content="""幸运随点评团体验霸王餐，心情好~蜀九香刚进驻泉州不久，招牌大名气响，以至于刚到店门口的我被门廊密密麻麻排队取号等候桌位的食客们的热情吓到了！！
        整体感觉:1.装修布置很新，新开业嘛！不管是餐位桌椅，隔断墙，仿古门窗，卫生间，都干干净净古色古香，还有民乐师演奏。2.服务超级赞，尽管人很多！
        刚说被楼下排队的人数吓到了，结果食客们都悠然嗑着瓜子唠着嗑，喝着果汁吹着风…足见店家礼遇；再说我们在楼上吃的时候，一位阿姨忙前忙后谦恭有礼，加高汤填茶水调火力送纸巾，提醒个人物品…真的很贴心！
        听口音很有可能来自川蜀地区，为来自远方的高质量服务人员点赞！3.细节取胜！大屏幕作为背景墙，上演琵琶曲，古筝曲，嘈嘈切切错杂弹，真是与店内装修风格相得益彰；
        还有在卫生间看到了各种温馨提示；再比如结账门口预备的去火去辣的清凉薄荷糖圈…总之，你，值得拥有！"""

    # inputs=[(input_content,labels.labels(),["密密麻麻"])]
    label_list=negitive+postive+neutral+[Label()]
    inputs=[(input_content,la_.labels(),["热情"]) for la_ in label_list]
    res=model.predict(inputs)
    for res_ in res:
        print(res_)

def server_seq2seq_lstm_attention_with_condition():
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    model_param={
        "use_mul_gpu":False,
        "model_path":"./token_test/output_sentiment_generate_dir_seq2seq_lstm_attention_with_condition",
        "bert_config_file":"./chinese_L-12_H-768_A-12/bert_config.json",
        "vocab_file":"./chinese_L-12_H-768_A-12/vocab.txt",
        "max_seq_length":128,
        "max_target_seq_length":10,
        "batch_size":1,
        "mode_type":"seq2seq_lstm_attention_with_condition"

    }
    labels=Label(services=[1,1,1,1])
    negitive=[Label(locations=[1,1,1]),Label(services=[1,1,1,1]),Label(prices=[1,1,1]),Label(environments=[1,1,1,1]),Label(dishes=[1,1,1,1]),Label(others=[1,1])]
    postive=[Label(locations=[3,3,3]),Label(services=[3,3,3,3]),Label(prices=[3,3,3]),Label(environments=[3,3,3,3]),Label(dishes=[3,3,3,3]),Label(others=[3,3])]
    neutral=[Label(locations=[2,2,2]),Label(services=[2,2,2,2]),Label(prices=[2,2,2]),Label(environments=[2,2,2,2]),Label(dishes=[2,2,2,2]),Label(others=[2,2])]
    model=ModelServer(model_param)
    '[MASK]'
    input_content="""幸运随点评团体验霸王餐，心情好~蜀九香刚进驻泉州不久，招牌大名气响，以至于刚到店门口的我被门廊密密麻麻排队取号等候桌位的食客们的热情吓到了！！
        整体感觉:1.装修布置很新，新开业嘛！不管是餐位桌椅，隔断墙，仿古门窗，卫生间，都干干净净古色古香，还有民乐师演奏。2.服务超级赞，尽管人很多！
        刚说被楼下排队的人数吓到了，结果食客们都悠然嗑着瓜子唠着嗑，喝着果汁吹着风…足见店家礼遇；再说我们在楼上吃的时候，一位阿姨忙前忙后谦恭有礼，加高汤填茶水调火力送纸巾，提醒个人物品…真的很贴心！
        听口音很有可能来自川蜀地区，为来自远方的高质量服务人员点赞！3.细节取胜！大屏幕作为背景墙，上演琵琶曲，古筝曲，嘈嘈切切错杂弹，真是与店内装修风格相得益彰；
        还有在卫生间看到了各种温馨提示；再比如结账门口预备的去火去辣的清凉薄荷糖圈…总之，你，值得拥有！"""

    # inputs=[(input_content,labels.labels(),["密密麻麻"])]
    label_list=negitive+postive+neutral+[Label()]
    inputs=[(input_content,la_.labels(),["密密麻麻"]) for la_ in label_list]
    res=model.predict(inputs)
    for res_ in res:
        print(res_)

def server_lstm_attention():
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    model_param={
        "use_mul_gpu":False,
        "model_path":"./token_test/output_sentiment_generate_dir_lstm_attention",
        "bert_config_file":"./chinese_L-12_H-768_A-12/bert_config.json",
        "vocab_file":"./chinese_L-12_H-768_A-12/vocab.txt",
        "max_seq_length":128,
        "max_target_seq_length":10,
        "batch_size":1,
        "mode_type":"lstm_attention"

    }
    labels=Label(services=[1,1,1,1])
    negitive=[Label(locations=[1,1,1]),Label(services=[1,1,1,1]),Label(prices=[1,1,1]),Label(environments=[1,1,1,1]),Label(dishes=[1,1,1,1]),Label(others=[1,1])]
    postive=[Label(locations=[3,3,3]),Label(services=[3,3,3,3]),Label(prices=[3,3,3]),Label(environments=[3,3,3,3]),Label(dishes=[3,3,3,3]),Label(others=[3,3])]
    neutral=[Label(locations=[2,2,2]),Label(services=[2,2,2,2]),Label(prices=[2,2,2]),Label(environments=[2,2,2,2]),Label(dishes=[2,2,2,2]),Label(others=[2,2])]
    model=ModelServer(model_param)
    '[MASK]'
    input_content="""整体感觉:1.装修布置很新，新开业嘛！不管是餐位桌椅，隔断墙，仿古门窗，卫生间，都干干净净古色古香，还有民乐师演奏。2.服务超级赞，尽管人很多！
        刚说被楼下排队的人数吓到了，结果食客们都悠然嗑着瓜子唠着嗑，喝着果汁吹着风…足见店家礼遇；再说我们在楼上吃的时候，一位阿姨忙前忙后谦恭有礼，加高汤填茶水调火力送纸巾，提醒个人物品…真的很贴心！
        听口音很有可能来自川蜀地区，为来自远方的高质量服务人员点赞！3.细节取胜！大屏幕作为背景墙，上演琵琶曲，古筝曲，嘈嘈切切错杂弹，真是与店内装修风格相得益彰；
        还有在卫生间看到了各种温馨提示；再比如结账门口预备的去火去辣的清凉薄荷糖圈…总之，你，值得拥有！"""

    # inputs=[(input_content,labels.labels(),["密密麻麻"])]
    label_list=negitive+postive+neutral+[Label()]
    inputs=[(input_content,la_.labels(),["超级赞"]) for la_ in label_list]
    res=model.predict(inputs,5)
    for res_ in res:
        print(res_)

def main():
    server_lstm_attention()

if __name__ == '__main__':
    main()