import tensorflow as tf
import numpy as np

def positional_encodeing(batchSize,max_seq_len,embeddingdim):
    PosE = np.array([[pos/np.power(10000.,2*(i//2)/embeddingdim) for i in range(embeddingdim)] for pos in range(max_seq_len)])
    PosE[:,0::2] = np.sin(PosE[:,0::2])
    PosE[:,1::2] = np.cos(PosE[:,1::2])
    # PosE = tf.tile(tf.expand_dims(tf.cast(PosE,tf.float32),axis=0),[batchSize,1,1])
    PosE = tf.expand_dims(tf.cast(PosE, tf.float32), axis=0)
    return PosE

def multilhead_attention(k,q,v,headNum,rawKeys,embeddingdim):#返回attention处理后的结果
    # embeddingdim=q.shape[-1]
    assert embeddingdim%headNum == 0, '输入的headNum必须可以被embeddingdim整除'
    # k = keras.layers.Dense(embeddingdim)(k) #batchsize,seqlen,embeddingdim
    # q = keras.layers.Dense(embeddingdim)(q)
    # v = keras.layers.Dense(embeddingdim)(v)
    K = tf.concat(tf.split(k, headNum, axis=-1), axis=0)#batchsize*headNum,seqlen,embeddingdim/headNum
    Q = tf.concat(tf.split(q, headNum, axis=-1), axis=0)
    V = tf.concat(tf.split(v, headNum, axis=-1), axis=0)
    Q_Kw = tf.matmul(Q,tf.transpose(K,[0,2,1]))/tf.math.sqrt(tf.cast(Q.shape[-1],dtype=tf.float32))#batchsize*headNum,seqlen,seqlen
    #通过softmax计算权重系数,计算前使用mask处理padding数据为-1e6
    mask = tf.math.equal(tf.tile(tf.expand_dims(tf.tile(rawKeys,[headNum,1]),1),[1,Q_Kw.shape[1],1]),0)#batchsize*headNum,seqlen,seqlen
    padding = tf.ones_like(tf.cast(mask,dtype=tf.float32))*-1e6
    Q_Kw = tf.where(mask,padding,Q_Kw)
    attention_weight = tf.nn.softmax(Q_Kw)
    attention_dealing = tf.matmul(attention_weight,V)#batchsize*headNum,seqlen,embeddingdim/headNum
    attention_output = tf.concat(tf.split(attention_dealing,headNum,axis=0),axis=-1)#batchsize,seqlen,embeddingdim
    return attention_output,attention_weight
# rawKeys = tf.cast(tf.constant([[1,2,3,0,0],[2,1,0,0,0]]),dtype=tf.float32)#2,5
# input = embeddingLayer(rawKeys,5,10)#2,5,10
# aa,bb=multilhead_attention(input,input,input,2,rawKeys,10)
# print(aa)
class LayerNormalization(tf.keras.layers.Layer):
    def __init__(self,epsilon=1e-6):
        self.epsilon = epsilon
        super(LayerNormalization,self).__init__()
    def build(self,input_shape):
        self.gamma = self.add_weight(name='gamma',shape=input_shape[-1:],initializer=tf.random_normal_initializer,trainable=True)
        self.beta = self.add_weight(name='bate',shape=input_shape[-1:],initializer=tf.zeros_initializer,trainable=True)
        super(LayerNormalization,self).build(input_shape)
    def call(self,inputs):
        Mean = tf.keras.backend.mean(inputs,axis=-1,keepdims=True)
        std = tf.keras.backend.std(inputs,axis=-1,keepdims=True)
        return self.gamma*(inputs-Mean)/(std+self.epsilon)+self.beta
    # def compute_output_shape(self, input_shape):
    #     return input_shape
# rawKeys = tf.cast(tf.constant([[1,2],[2,0]]),dtype=tf.float32)#1,5,6
# input = embeddingLayer(rawKeys,5,2)
# LN = LayerNormalization()
# ll=LN(input)
# print(ll)
def point_wise_feed_forward_network(embeddingdim,Tproperties):
    return tf.keras.models.Sequential([tf.keras.layers.Dense(Tproperties),tf.keras.layers.Dense(embeddingdim)])

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self,embeddingdim,Tproperties,headNum,dropout_rate=0.5):
        super(EncoderLayer, self).__init__()
        self.dropoutlayer1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropoutlayer2 = tf.keras.layers.Dropout(dropout_rate)
        self.normalization1 = LayerNormalization()
        self.normalization2 = LayerNormalization()
        self.forward_network =point_wise_feed_forward_network(embeddingdim,Tproperties)
        self.densek = tf.keras.layers.Dense(embeddingdim)
        self.denseq = tf.keras.layers.Dense(embeddingdim)
        self.densev = tf.keras.layers.Dense(embeddingdim)
        self.embeddingdim = embeddingdim
        self.headNum = headNum
    def call(self,input,rawKeys):
        k = self.densek(input)  # batchsize,seqlen,embeddingdim
        q = self.denseq(input)
        v = self.densev(input)
        output1,att_weight = multilhead_attention(k,q,v,self.headNum,rawKeys,self.embeddingdim)
        outputDroped1 =self.dropoutlayer1(output1)
        outputNorm1 = self.normalization1(input+outputDroped1)#残差连接

        output2 = self.forward_network(outputNorm1)
        outputDroped2 = self.dropoutlayer2(output2)
        outputNorm2 = self.normalization1(outputNorm1+outputDroped2)
        return outputNorm2,att_weight

class Encoder(tf.keras.Model):
    def __init__(self,batchSize,max_seq_len,embeddingdim,Tproperties,vocabSize,headNum,layerNum,calssNum,dropout_rate):
        super(Encoder,self).__init__()
        self.layerNum = layerNum
        self.pos_ombedding = positional_encodeing(batchSize,max_seq_len,embeddingdim)
        self.Embedded = tf.keras.layers.Embedding(vocabSize,embeddingdim)
        self.encoderlayerList = [EncoderLayer(embeddingdim,Tproperties,headNum,dropout_rate) for _ in range(layerNum)]
        # self.testecoderlayer = EncoderLayer(embeddingdim,Tproperties,headNum,dropout_rate)
        self.flatten = tf.keras.layers.Flatten(input_shape=(max_seq_len,embeddingdim))
        self.lastLayer = tf.keras.layers.Dense(calssNum,activation='softmax')
        self._set_inputs(tf.TensorSpec([None,max_seq_len],name='input'))

    def call(self,input):
        rawKeys = input
        # 合并词嵌入和位置编码
        posembedding = self.pos_ombedding
        embedding = self.Embedded(input)
        x=embedding#+posembedding
        weight={}
        for i in range(self.layerNum):
            x,att_weight = self.encoderlayerList[i](x,rawKeys)
            weight['att_weight{}'.format(i)]=att_weight
        logits = self.flatten(x)
        logits = self.lastLayer(logits)
        return logits

    def loss(self,logits,target):
        Y=tf.one_hot(target,15)
        loss =tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=Y)
        return tf.reduce_sum(loss)



