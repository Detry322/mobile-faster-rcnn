import tensorflow as tf

def load_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="prefix")
    return graph

def load_all(base_name):
    initial = base_name + '-initial.pb'
    last_layer = base_name + '-last_layer.pb'
    inference = base_name + '-inference.pb'
    return {
        'initial': load_graph(initial),
        'last_layer': load_graph(last_layer),
        'inference': load_graph(inference)
    }
