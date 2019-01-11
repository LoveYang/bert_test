# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 18:28:35 2019

@author: hesheng
"""

import os 
import re

_root_path=os.path.abspath("./2014")


re_token_ex=re.compile(r'/[a-z0-9]{1,5}$')
re_token_alter_ex=re.compile(r'\[[\s\S]+?\]/[a-z]+')
re_mode=re.compile(r'\[[\s\S]+\]')

# S是词开始，M是词的中部，E是词的结束， 对于单字词取S
LABEL=['S','M','E']

#replace '[十二/m 五/m 规划/n]/nz' to mode1:十二/m 五/m 规划/n  mode2:十二五规划/nz
def alterReplace(text,mode='mode1'):
    assert mode in ['mode1','mode2'] ,'mode only two mode:mode1 or mode2'
    res=text
    for x in re.findall(re_token_alter_ex,text):
        modetext=re.match(re_mode,x).group(0)[1:-1]
        if 'mode2'==mode:
            modetext=''.join( map(lambda x:re.split(re_token_ex,x)[0], modetext.split(' ')))
        res=res.replace(x,modetext)
    return res

#mode :str or list
def tokenbyLABEL(text,yLabel,mode="str"):
    assert len(text)==len(yLabel),"length of input and yLabel must be equal "
    token=[]
    for i,char_i in enumerate(yLabel):
        if char_i==LABEL[0]:
            token.append(text[i])
        else:
            token[-1]+=text[i]
    return token if 'list'==mode else ' '.join(token)

def testToken(token,yLabel):
    re_token=tokenbyLABEL(token.replace(' ',''),yLabel,mode='str')
    assert len(re_token)==len(token)

def main_write_data():
    unseek_path=os.listdir(_root_path)

    while len(unseek_path)>0:
        deal_path=unseek_path.pop()
        deal_pathlist=os.listdir(os.path.join(_root_path,deal_path))
        for file in deal_pathlist:
            with open(os.path.join(_root_path,deal_path,file),'r',encoding='utf8') as rf:
                text=rf.read().strip().replace('\n','').replace('\t','')
                if len(text)<1:
                    continue
    #            把可选分词选择mode1,
                text=alterReplace(text,mode='mode1')
                
                token_se_list=text.split(' ')
                token=[]
                yLabel=''
                x=''
                for token_se in token_se_list:
                    _token=re.split(re_token_ex,token_se)
                    if len(_token)>2:
                        raise ValueError("*".join(_token)+' must len==2 '+token_se,os.path.join(_root_path,deal_path,file))
                    else:
                        try:
                            token_len=len(_token[0])
                            if token_len>0:
                                token+=[_token[0]]
                                yLabel+= LABEL[0]+LABEL[1]*(token_len-2)+LABEL[-1]if token_len >1 else LABEL[0]
                        except IndexError:
                            raise IndexError("string index out of range"+token_se)
                restoken=' '.join(token)
                vail_token=''.join(token)
                assert len(vail_token)==len(yLabel), \
                "token len:{0},label len :{1}\n{2}\n{3}".format(len(vail_token),len(yLabel),vail_token,yLabel)
                pre_write=restoken.strip()+'\n'+yLabel.strip()+'\n\n'
                yield pre_write

if __name__=="__main__":
    from tqdm import tqdm
    with open(os.path.abspath('./xy_data.txt'),'w',encoding='utf8') as wf:
        for sw in tqdm(main_write_data()):
            wf.write(sw)
            
    

