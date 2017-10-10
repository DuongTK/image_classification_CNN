# Convolutional Layer 1.
filter_size1 = 5          # filters 5 x 5 pixels.
num_filters1 = 16         # có 16 filters.
# Convolutional Layer 2.
filter_size2 = 5          # filters 5 x 5 pixels.
num_filters2 = 36         # có 36 filters.
# Fully-connected layer.
fc_size = 128             # số lượng neurons của fully-connected layer.
# kích cỡ của ảnh MNIST 28 pixel
img_size = 28
# kích cỡ mảng 1 chiều khi ảnh bị flat
img_size_flat = img_size * img_size
# Tuple biểu diễn shape của ảnh
img_shape = (img_size, img_size)
# Số lượng  colour channels
num_channels = 1
# 10 số nên có 10 classes
num_classes = 10

#LOAD DATA
from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot=True)
#HÀM TẠO WEIGHT VÀ BIAS
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

#HÀM TẠO CONVOLUTIONAL LAYER
def new_conv_layer(input, num_input_channels, filter_size, num_filters, use_pooling=True): 
    shape = [filter_size, filter_size, num_input_channels, num_filters]
    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)

    layer = tf.nn.conv2d(input=input,filter=weights,strides=[1, 1, 1, 1],padding='SAME')
    layer += biases

    if use_pooling:
        layer = tf.nn.max_pool(value=layer,ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')
    layer = tf.nn.relu(layer)
    return layer, weights

#HÀM FLATTENING LAYER	
def flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer_flat = tf.reshape(layer, [-1, num_features])

    return layer_flat, num_features
	
#HÀM TẠO Fully-Connected Layer
def new_fc_layer(input, num_inputs, num_outputs, use_relu=True): 
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)
    layer = tf.matmul(input, weights) + biases
	
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer
	
#Placeholder variables
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

#Convolutional Layer 1
layer_conv1, weights_conv1 = new_conv_layer(input=x_image, num_input_channels=num_channels, filter_size=filter_size1, num_filters=num_filters1,use_pooling=True)
#Convolutional Layer 2
layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1, num_input_channels=num_filters1, filter_size=filter_size2, num_filters=num_filters2,use_pooling=True)
#Flatten Layer
layer_flat, num_features = flatten_layer(layer_conv2)
#Fully-Connected Layer 1	
layer_fc1 = new_fc_layer(input=layer_flat, num_inputs=num_features, num_outputs=fc_size, use_relu=True)
#Fully-Connected Layer 2
layer_fc2 = new_fc_layer(input=layer_fc1, num_inputs=fc_size, num_outputs=num_classes, use_relu=False)

#Predicted Class
y_pred = tf.nn.softmax(layer_fc2)	
y_pred_cls = tf.argmax(y_pred, dimension=1)

#TỐI ƯU LOSS-FUNCTION
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,labels=y_true)
cost = tf.reduce_mean(cross_entropy)

#Optimization Method
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
#Performance Measures
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))	

#CHẠY TensorFlow 
session = tf.Session()
session.run(tf.global_variables_initializer())		
											
#HÀM LẶP TRANING
train_batch_size = 64
total_iterations = 0

def optimize(num_iterations):
    global total_iterations
    start_time = time.time()

    for i in range(total_iterations,
                   total_iterations + num_iterations):

        x_batch, y_true_batch = data.train.next_batch(train_batch_size)
        feed_dict_train = {x: x_batch,y_true: y_true_batch}

        session.run(optimizer, feed_dict=feed_dict_train)

        if i % 100 == 0:
            acc = session.run(accuracy, feed_dict=feed_dict_train)
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"
            print(msg.format(i + 1, acc))

    total_iterations += num_iterations
    end_time = time.time()
    time_dif = end_time - start_time
	
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

	
optimize(num_iterations=10000)
				   