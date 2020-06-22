import pickle
import os
import json

def save_config(congfig_file='config'):#max_seq_len,embeddingdim,Tproperties,vocabSize,headNum,layerNum,calssNum
    configdict = {'max_seq_len':20,'embeddingdim':100,'Tproperties':300,'vocabSize':4563,'headNum':10,'layerNum':1,'calssNum':15,'dropout_rate':0.3,'batchSize':100}
    with open(congfig_file, 'w', encoding='utf-8') as f:
        json.dump(configdict,f,ensure_ascii=True,indent=4)#ensure_ascii=False,

def load_config(congfig_file='config'):
    with open(congfig_file,'r',encoding = 'utf-8') as f:
        return json.load(f)

save_config()
# if __name__ =='__main__':
#     config1 = {'batchSize':10,'embeddingdim':100,'layerSize':3,'dropoutRate':0.2}
#     save_config(config1,'config')
#     aa = load_config('config')
#     print(aa)