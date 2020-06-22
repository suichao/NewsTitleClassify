import tensorflow as tf
import utils
from Bertmodel import Encoder
from dataUtils import loadDict,sentense2id
print(tf.__version__)

config=utils.load_config()
max_seq_len =config.get('max_seq_len')
embeddingdim =config.get('embeddingdim')
Tproperties = config['Tproperties']
vocabSize = config['vocabSize']
headNum = config['headNum']
layerNum = config['layerNum']
calssNum = config['calssNum']
dropout_rate =config['dropout_rate']
batchSize =config['batchSize']
labeldict={0: 'news_story', 1: 'news_culture', 2: 'news_entertainment', 3: 'news_sports', 4: 'news_finance', 6: 'news_house',
      7: 'news_car', 8: 'news_edu', 9: 'news_tech', 10: 'news_military', 12: 'news_travel', 13: 'news_world', 14: 'stock',
      5: 'news_agriculture', 11: 'news_game'}


encoder = Encoder(batchSize,max_seq_len,embeddingdim,Tproperties,vocabSize,headNum,layerNum,calssNum,dropout_rate)
encoder.compile(optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.01),loss=tf.keras.losses.sparse_categorical_crossentropy,metrics=['acc'])
encoder.load_weights('save_model/weight.ckpt')

sport=['中国体操功勋教练陆善真因病逝世','男足国少队集训29天完成29堂视频战术课','武磊首发 西班牙人1-3告负遭重启后首败','佩佩世界波 阿森纳惨遭绝杀1-2负布莱顿']
culture=['什么是上海味道？这样才叫“懂经”','阎维文雷佳等唱响经典选段','五彩线兮端午节','数字时代 图书仍是好选择','壁画变动画 民俗数字化']
finance=['今年快递业收入或超8600亿 年中行业迎小高峰','创业板退市新规出台 一批企业游走退市边缘','财政部:1-5月一般公共预算收入同比下降13.6%']
edu=['专家评论：期待毕业季留下温暖注脚','疫情防控常态化下的大学担当','北京：高校毕业生行李打包不能“一刀切”','9月开学前，北京高校毕业生再次返校可能性小']
car=['福特F-150纯电动版将延迟至2022年推出','全新广汽本田飞度官图发布 7月预售','长安福特探险者正式下线 将于6月上市','宝骏360换装新发动机 售价保持不变']

def predict(encoder,new_title):
      _,word2id=loadDict()
      inputs = sentense2id(new_title,word2id)
      logits = encoder.predict(inputs)
      calssPred=[[labeldict.get(logit.tolist().index(max(logit)))] for logit in logits]
      print(calssPred)

predict(encoder,sport)

