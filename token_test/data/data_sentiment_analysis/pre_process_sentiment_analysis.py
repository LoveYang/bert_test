# -*- coding: utf-8 -*-
import tensorflow as tf
from tokenization import FullTokenizer,BasicTokenizer
import os
import numpy as np
from thulac import thulac
import re
import csv
import collections

files_list=["sentiment_analysis_trainingset.csv","sentiment_analysis_validationset.csv"]
file=[files_list[-1]]


class CN_Token():
    def __init__(self):
        self.cn_token=thulac(seg_only=True)
        self.re_punctuationspace=re.compile("\s")
    def tokenize(self,text):
        tokens=[]
        for _t in re.split(self.re_punctuationspace,text.strip()):
            for token in self.cn_token.cut(_t):
                tokens.append(token[0])
        return tokens


cn_token=CN_Token()
fulltoken=FullTokenizer(os.path.abspath("../../../chinese_L-12_H-768_A-12/vocab.txt"))

# return features
def token_parse(string,maxtoken_length,maxTarget_length=20,returntype="list"):
    if isinstance(maxtoken_length,np.ndarray):
        maxtoken_length=maxtoken_length[0]
    token_a=fulltoken.tokenize(string)
    token_b=cn_token.tokenize(string)

    def random_generate_ml_out(token_a,token_b,maxtoken_length,pvalue=0.1):
        # pvalue=0.1 10% token mask
        import random
        token_a_start=0
        output_input_index=[]
        for token_b_s in token_b:
            mask_input_index=[]
            if pvalue>=random.uniform(0,1):
                # mask ch_token
                for token_a_s in fulltoken.tokenize(token_b_s):
                    if token_a_s=="[UNK]":
                        mask_input_index = []
                        break
                    else:
                        try:
                            token_a_index=token_a.index(token_a_s,token_a_start)
                        except ValueError as e:
                            mask_input_index=[]
                            break
                        token_a_start=token_a_index+1
                        mask_input_index.append(token_a_index)

                        if token_a_start>maxtoken_length-2:
                            return output_input_index
                if len(mask_input_index)>0:
                    output_input_index.append(mask_input_index)

        return output_input_index

    output_input_index=random_generate_ml_out(token_a,token_b,maxtoken_length,pvalue=0.5)
    features=[]
    def list2INT32Array(lst):
        return np.array(lst,dtype=np.int32)
    for single_output_index in output_input_index:
        single_output=[]
        token_a_copy=token_a.copy()
        for x in single_output_index:
            single_output.append(token_a_copy[x])
            token_a_copy[x]="[MASK]"

        #  CLS and SEP
        if len(token_a_copy) > maxtoken_length - 2:
            token_a_copy=token_a_copy[:maxtoken_length-2]
        token_a_copy = ["[CLS]"] + token_a_copy + ["[SEP]"]

        if len(single_output) > maxTarget_length - 2:
            single_output=single_output[:maxtoken_length-2]
        single_output = ["[CLS]"] + single_output + ["[SEP]"]

        input_token_ids=fulltoken.convert_tokens_to_ids(token_a_copy)
        target_token_ids=fulltoken.convert_tokens_to_ids(single_output)
        input_mask=[1]*len(input_token_ids)
        target_mask=[1]*len(target_token_ids)
        while len(input_token_ids)<maxtoken_length:
            input_token_ids.append(0)
            input_mask.append(0)
        while len(target_token_ids)<maxTarget_length:
            target_token_ids.append(0)
            target_mask.append(0)
        segment_ids=[0]*maxtoken_length
        if returntype=="list":
            features.append( (input_token_ids ,input_mask \
                              ,target_token_ids, target_mask, segment_ids ) )

        else:
            features.append( (list2INT32Array(input_token_ids) ,list2INT32Array(input_mask) \
                              ,list2INT32Array(target_token_ids), list2INT32Array(target_mask), list2INT32Array(segment_ids) ) )

    return features


def save_tfrecord_Features(files,datatype="list"):
    if  (not isinstance(files,list)) and (not isinstance(files,tuple)):
        files=[files]

    def create_int_feature(values):
        f=tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
        return f
    for file in files:
        output_file=".".join(file.split('.')[:-1]+["tf_record"])
        writer = tf.python_io.TFRecordWriter(output_file)
        with open(file,"r",encoding='utf-8-sig') as rf:
            lines=csv.reader(rf)
            for line_num,content in enumerate(lines):
                # skip header
                if line_num>1:
                    assert len(content)==22
                    id=int(content[0])
                    features_tulpe=token_parse(content[1],maxtoken_length=128,maxTarget_length=10,returntype=datatype)
                    sentiment_labels=[int(label) for label in content[2:]] if datatype=="list" \
                        else np.array([int(label) for label in content[2:]],dtype=np.int32)

                    for feature_tulpe in features_tulpe:
                        features = collections.OrderedDict()
                        features["sentiment_labels"]=create_int_feature(sentiment_labels)
                        features["input_token_ids"]=create_int_feature(feature_tulpe[0])
                        features["input_mask"] = create_int_feature(feature_tulpe[1])
                        features["target_token_ids"] = create_int_feature(feature_tulpe[2])
                        features["target_mask"] = create_int_feature(feature_tulpe[3])
                        features["segment_ids"] = create_int_feature(feature_tulpe[4])
                        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
                        writer.write(tf_example.SerializeToString())
                        yield (file,line_num)
        writer.close()





def test_tfrecord_save():
    from tqdm import tqdm
    files_numlines=collections.defaultdict(set)
    for f,i in tqdm(save_tfrecord_Features(files_list)):
        files_numlines[f].add(i)
    for f,i in files_numlines.items():
        print("the file({0}) LinesNumber is {1} ".format(f,max(i)))




def test_token():
    text="""趁着国庆节，一家人在白天在山里玩耍之后，晚上决定吃李记搅团。
东门外这家店门口停车太难了，根本没空位置，所以停在了旁边的地下停车场。"""
    print(cn_token.tokenize(text))
    print(fulltoken.tokenize(text))
    token_parse(text,128,10)


if __name__=="__main__":
    test_tfrecord_save()