from Bertmodel import Encoder
import utils
import tensorflow as tf
from dataUtils import genData
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
encoder = Encoder(batchSize,max_seq_len,embeddingdim,Tproperties,vocabSize,headNum,layerNum,calssNum,dropout_rate)
genD =genData()
xtrain, ytrain,xtest, ytest = genD.returnData(batchSize)

trainLoss = tf.keras.metrics.Mean(name='trainLoss')
trainAcc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')#sparse_categorical_crossentropy
valibLoss = tf.keras.metrics.Mean(name='valid_loss')
valibAcc = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')
optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.01)
# optimizer = tf.keras.optimizers.SGD(learning_rate=0.0000001)
def train(epoch=10):
    for ep in range(epoch):
        trainLoss.reset_states()
        trainAcc.reset_states()
        valibLoss.reset_states()
        valibAcc.reset_states()
        for i,(X_train,Y_train) in enumerate(zip(xtrain, ytrain)):
            with tf.GradientTape() as tape:
                train_logits = encoder(X_train)
                train_loss = tf.keras.losses.sparse_categorical_crossentropy(Y_train,train_logits)
                gradients = tape.gradient(train_loss, encoder.trainable_variables)
                optimizer.apply_gradients(zip(gradients,encoder.trainable_variables))
                trainLoss(train_loss)
                trainAcc.update_state(Y_train,train_logits)

                #验证集
                j=i%len(xtest)
                x_test,y_test=xtest[j],ytest[j]
                test_logits = encoder(x_test)
                test_loss = tf.keras.losses.sparse_categorical_crossentropy(y_test, test_logits)
                valibLoss(test_loss)
                valibAcc.update_state(y_test, test_logits)
        print('epoch:{},loss:{:.4f},acc:{:.4f},vild_loss:{:.4f},vild_acc:{:.4f}'.format(ep + 1, trainLoss.result(),
                                                                                        trainAcc.result(),
                                                                                        valibLoss.result(),
                                                                                        valibAcc.result()))
train(epoch=5)
encoder.save_weights("save_model/weight.ckpt")