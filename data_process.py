import json, glob
import numpy as np
from tqdm import tqdm
from multiprocessing import  Process
from chatglm3_tokenizer.tokenization_chatglm import ChatGLMTokenizer
#import pandas as pd

BATCH_SIZE = 1000000

def process_wiki(tokenizer):
    print('正在处理wiki')
    with open('./pretrain_data/Wikipedia/wikipedia-cn-20230720-filtered.json','r',encoding='utf-8') as f:
        data=json.load(f)
    doc_ids=[]
    for line in tqdm(data):
        text=line['completion']
        text_id=tokenizer.encode(text,add_special_tokens=False)
        text_id.append(tokenizer.special_tokens['<eos>'])
        if len(text_id)>5:
            doc_ids+=text_id
    token = len(doc_ids)
    arr = np.array(doc_ids,dtype=np.uint16)
    with open('./pretrain_data/wiki.bin','wb') as f:
        f.write(arr.tobytes())
    print('wiki处理完成，总token:',token)

def process_medical(tokenizer,data_path,name):
    print('正在处理medical_',name)
    f=open(data_path,'r',encoding='utf-8')
    doc_ids=[]
    while True:
        line=f.readline()
        if not line:
            break
        line=json.loads(line)
        text=line['text']
        text_id=tokenizer.encode(text,add_special_tokens=False)
        text_id.append(tokenizer.special_tokens['<eos>'])
        if len(text_id)>5:
            doc_ids+=text_id
    token = len(doc_ids)
    arr = np.array(doc_ids,dtype=np.uint16)
    with open('./pretrain_data/medical_{}.bin'.format(name),'wb') as f:
        f.write(arr.tobytes())
    print('medical_',name,'处理完成，总token:',token)

def sft_to_pretrain(tokenizer):
    print('正在处理medical,sft_to_pretrain')
    '''
    df=pd.read_csv('./finetune_data/medical_qa_144w.csv')
    for _,q,a in tqdm(df.itertuples()):
        q_id = tokenizer.encode(q,add_special_tokens=False)
        a_id = tokenizer.encode(a,add_special_tokens=False)

        print('question=',q,'answer=',a)
        print('------------\n')
        text_id=q_id+a_id+[tokenizer.special_tokens['<eos>']]
        if len(text_id)>5:
            doc_ids+=text_id
    '''
    doc_ids=[]
    with open('./finetune_data/shibing624/train_en_1.json','r',encoding='utf-8') as f:
        for row in f:
            line=json.loads(row)
            q=line['input']
            a=line['output']
            q_id=tokenizer.encode(q,add_special_tokens=False)
            a_id=tokenizer.encode(a,add_special_tokens=False)
            text_id=q_id+a_id+[tokenizer.special_tokens['<eos>']]
            if len(text_id)>5:
                doc_ids+=text_id
    with open('./finetune_data/shibing624/test_en_1.json','r',encoding='utf-8') as f:
        for row in f:
            line=json.loads(row)
            q=line['input']
            a=line['output']
            q_id=tokenizer.encode(q,add_special_tokens=False)
            a_id=tokenizer.encode(a,add_special_tokens=False)
            text_id=q_id+a_id+[tokenizer.special_tokens['<eos>']]
            if len(text_id)>5:
                doc_ids+=text_id
    with open('./finetune_data/shibing624/valid_en_1.json','r',encoding='utf-8') as f:
        for row in f:
            line=json.loads(row)
            q=line['input']
            a=line['output']
            q_id=tokenizer.encode(q,add_special_tokens=False)
            a_id=tokenizer.encode(a,add_special_tokens=False)
            text_id=q_id+a_id+[tokenizer.special_tokens['<eos>']]
            if len(text_id)>5:
                doc_ids+=text_id
    with open('./finetune_data/shibing624/train_zh_0.json','r',encoding='utf-8') as f:
        for row in f:
            line=json.loads(row)
            q=line['instruction']+line['input']
            a=line['output']
            q_id=tokenizer.encode(q,add_special_tokens=False)
            a_id=tokenizer.encode(a,add_special_tokens=False)
            text_id=q_id+a_id+[tokenizer.special_tokens['<eos>']]
            if len(text_id)>5:
                doc_ids+=text_id
    with open('./finetune_data/shibing624/test_zh_0.json','r',encoding='utf-8') as f:
        for row in f:
            line=json.loads(row)
            q=line['instruction']+line['input']
            a=line['output']
            q_id=tokenizer.encode(q,add_special_tokens=False)
            a_id=tokenizer.encode(a,add_special_tokens=False)
            text_id=q_id+a_id+[tokenizer.special_tokens['<eos>']]
            if len(text_id)>5:
                doc_ids+=text_id
    with open('./finetune_data/shibing624/valid_zh_0.json','r',encoding='utf-8') as f:
        for row in f:
            line=json.loads(row)
            q=line['instruction']+line['input']
            a=line['output']
            q_id=tokenizer.encode(q,add_special_tokens=False)
            a_id=tokenizer.encode(a,add_special_tokens=False)
            text_id=q_id+a_id+[tokenizer.special_tokens['<eos>']]
            if len(text_id)>5:
                doc_ids+=text_id
    token = len(doc_ids)
    arr = np.array(doc_ids,dtype=np.uint16)
    with open('./pretrain_data/medical_qa.bin','wb') as f:
        f.write(arr.tobytes())
    print('medical,sft_to_pretrain处理完成，总token:',token)

def process_baidu(tokenizer):
    cnt=0
    token=0
    batch_cnt=0
    doc_ids=[]
    print('正在处理baidubaike')
    f1=open('./pretrain_data/baidubaike/563w_baidubaike.json','r',encoding='utf-8')
    while True:
        line = f1.readline()
        if not line:
            break
        line=json.loads(line)
        text=''
        try:
            text+=line['title']+'：'+line['summary']
        except:
            pass
        for per in line['sections']:
            text+=per['title']+'：'+per['content']+'。'
        text_id=tokenizer.encode(text,add_special_tokens=False)
        text_id.append(tokenizer.special_tokens['<eos>'])
        if len(text_id)>5:
            doc_ids+=text_id
        cnt+=1
        if cnt%BATCH_SIZE==0:
            batch_cnt+=1
            token+=len(doc_ids)
            arr = np.array(doc_ids,dtype=np.uint16)
            print('cnt:',cnt,'arr_shape:',arr.shape)
            with open('./pretrain_data/baidubaike_563w_{}.bin'.format(batch_cnt),'wb') as f2:
                f2.write(arr.tobytes())
            doc_ids=[]
            del arr

    if len(doc_ids) > 1:
        batch_cnt+=1
        token+=len(doc_ids)
        arr = np.array(doc_ids,dtype=np.uint16)
        print('cnt:',cnt,'arr_shape:',arr.shape)
        with open('./pretrain_data/baidubaike_563w_{}.bin'.format(batch_cnt),'wb') as f2:
            f2.write(arr.tobytes())
        doc_ids=[]
        del arr
    print('baidubaike处理完成，总token:',token)
    
def process_c4(tokenizer):
    print('正在处理c4')
    c4_zh_paths = glob.glob('./pretrain_data/c4_zh/*')
    c4_zh_paths=sorted(c4_zh_paths)
    print('Number of files:',len(c4_zh_paths))
    cnt=0
    token=0
    batch_cnt=0
    doc_ids=[]
    for per in tqdm(c4_zh_paths):
        with open(per,'r',encoding='utf-8') as f:
            for line in f:
                text = json.loads(line)
                text = text['text']
                text_id=tokenizer.encode(text,add_special_tokens=False)
                text_id.append(tokenizer.special_tokens['<eos>'])
                if len(text_id)>5:
                    doc_ids+=text_id
                cnt+=1
                if cnt%BATCH_SIZE==0:
                    batch_cnt+=1
                    token+=len(doc_ids)
                    arr = np.array(doc_ids,dtype=np.uint16)
                    print('\ncnt:',cnt,'arr_shape:',arr.shape)
                    with open('./pretrain_data/c4_zh_{}.bin'.format(batch_cnt),'wb') as f2:
                        f2.write(arr.tobytes())
                    doc_ids=[]
                    del arr

    if len(doc_ids) > 1:
        batch_cnt+=1
        token+=len(doc_ids)
        arr = np.array(doc_ids,dtype=np.uint16)
        print('\ncnt:',cnt,'arr_shape:',arr.shape)
        with open('./pretrain_data/c4_zh_{}.bin'.format(batch_cnt),'wb') as f2:
            f2.write(arr.tobytes())
        doc_ids=[]
        del arr
    print('c4处理完成，总token:',token)

def process_wudao(tokenizer):
    print('正在处理wudao')
    wudao_zh_paths = glob.glob('./pretrain_data/WuDaoCorpus/*')
    wudao_zh_paths=sorted(wudao_zh_paths)
    print('Number of files:',len(wudao_zh_paths))
    cnt=0
    token=0
    batch_cnt=0
    doc_ids=[]
    for per in tqdm(wudao_zh_paths):
        with open(per,'r',encoding='utf-8') as f:
            data=json.load(f)
            for text in data:
                text = text['title'] + text['content']
                text_id=tokenizer.encode(text,add_special_tokens=False)
                text_id.append(tokenizer.special_tokens['<eos>'])
                if len(text_id)>5:
                    doc_ids+=text_id
                cnt+=1
                if cnt%BATCH_SIZE==0:
                    batch_cnt+=1
                    token+=len(doc_ids)
                    arr = np.array(doc_ids,dtype=np.uint16)
                    doc_ids=[]
                    print('\ncnt:',cnt,'arr_shape:',arr.shape)
                    with open('./pretrain_data/wudaocorpus_zh_{}.bin'.format(batch_cnt),'wb') as f2:
                        f2.write(arr.tobytes())
                    del arr

    if len(doc_ids) > 1:
        batch_cnt+=1
        token+=len(doc_ids)
        arr = np.array(doc_ids,dtype=np.uint16)
        print('\ncnt:',cnt,'arr_shape:',arr.shape)
        with open('./pretrain_data/wudaocorpus_zh_{}.bin'.format(batch_cnt),'wb') as f2:
            f2.write(arr.tobytes())
    print('wudao处理完成，总token:',token)

if __name__=="__main__":
    tokenizer = ChatGLMTokenizer(vocab_file='./chatglm3_tokenizer/tokenizer.model')
    # 数据预处理-如果下载分词处理后的数据，可以不用执行以下函数
    p1=Process(target=process_wiki,args=(tokenizer,))
    p1.start()

    p2=Process(target=process_baidu,args=(tokenizer,))
    p2.start()

    p3=Process(target=process_c4,args=(tokenizer,))
    p3.start()

    p4=Process(target=process_wudao,args=(tokenizer,))
    p4.start()

    p5=Process(target=sft_to_pretrain,args=(tokenizer,))
    p5.start()
    
    p6=Process(target=process_medical,args=(tokenizer,'./pretrain_data/shibing624/medical_book_zh.json','book',))
    p6.start()
    
    p7=Process(target=process_medical,args=(tokenizer,'./pretrain_data/shibing624/train_encyclopedia.json','encyclopedia',))
    p7.start()
    
    #等待4个进程执行完
    p1.join()
    p2.join()
    p3.join()
    p4.join()
    p5.join()
    p6.join()
    p7.join()

    '''process_wiki()
    process_baidu()
    process_c4()
    process_wudao()
    sft_to_pretrain()
    process_medical('./pretrain_data/shibing624/medical_book_zh.json','book')
    process_medical('./pretrain_data/shibing624/train_encyclopedia.json','encyclopedia')
    '''
    print('data processing finished! now combined file......')

    # 分词处理后的文件列表
    data_path_list = glob.glob('./pretrain_data/*.bin')
    data_path_list=sorted(data_path_list)
    print('Number of files:',len(data_path_list))
    # print(data_path_list)
    data_lst=[]
    for data_path in tqdm(data_path_list):
        if 'pretrain_data.bin' in data_path:  # 最终的输出文件也在同一个目录，这个不要加进来
            continue
        with open(data_path,'rb') as f:
            data=np.fromfile(f,dtype=np.uint16)
            data_lst.append(data)
    arr = np.concatenate(data_lst)
    print('combined file finished! total shape:',arr.shape)
    with open('./pretrain_data/pretrain_data.bin','wb') as f:
        f.write(arr.tobytes())
