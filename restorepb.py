import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
sess = tf.Session()
#将保存的模型文件解析为GraphDef
model_f = gfile.FastGFile("model.pb",'rb')
graph_def = tf.GraphDef()
graph_def.ParseFromString(model_f.read())
_= tf.import_graph_def(graph_def,name="")
input_x = sess.graph.get_tensor_by_name("a:0")
print(sess.run(input_x))
#[array([ 11.], dtype=float32)]
