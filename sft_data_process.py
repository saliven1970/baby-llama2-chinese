import json,os
from tqdm import tqdm
import pandas as pd


def sft_process():
    q_lst = []
    a_lst = []
    # 这是json文件的处理方法，整个文件是一个json对象
    data_path_list = ['./finetune_data/alpaca_gpt4_data_zh.json']
    for data_path in tqdm(data_path_list):
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)      # 一次把文件读入，并加载到json
        for per in data:             # 然后一行行数据处理
            q = per['instruction']
            i = per['input']
            a = per['output']
            q = q + i
            if len(q) < 10 or len(a) < 5:
                continue
            if len(q) > 512 or len(a) > 512:
                continue
            q_lst.append(q)
            a_lst.append(a)

    # 这是jsonl文件的处理方法，一行一个json对象
    data_path_list = ['./finetune_data/Belle_open_source_1M.json']
    for data_path in tqdm(data_path_list):        
        f = open(data_path, 'r', encoding='utf-8')
        while True:
            line = f.readline()    # 一行一行读入文件
            if not line:
                break
            per = json.loads(line) # 加载到json
            q = per['instruction']
            i = per['input']
            a = per['output']
            q = q + i
            if len(q) < 10 or len(a) < 5:
                continue
            if len(q) > 512 or len(a) > 512:
                continue
            q_lst.append(q)
            a_lst.append(a)
            
    df = pd.DataFrame(columns=['prompt', 'answer'])
    df['prompt'] = q_lst
    df['answer'] = a_lst
    df.to_csv('./finetune_data/sft_data.csv', index=False)

if __name__=="__main__":
    save_dir = './finetune_data'
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    sft_process()
