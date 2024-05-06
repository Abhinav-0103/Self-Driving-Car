import os
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()
from tensorflow.core.protobuf import saver_pb2
import driving_data
import model

LOGDIR = './save'

sess = tf.compat.v1.InteractiveSession()

L2NormConst = 0.001

train_vars = tf.compat.v1.trainable_variables()

loss = tf.compat.v1.reduce_mean(tf.compat.v1.square(tf.compat.v1.subtract(model.y_, model.y))) + tf.compat.v1.add_n([tf.compat.v1.nn.l2_loss(v) for v in train_vars]) * L2NormConst
train_step = tf.compat.v1.train.AdamOptimizer(1e-4).minimize(loss)
sess.run(tf.compat.v1.global_variables_initializer())

# create a summary to monitor cost tensor
tf.compat.v1.summary.scalar("loss", loss)
# merge all summaries into a single op
merged_summary_op = tf.compat.v1.summary.merge_all()

saver = tf.compat.v1.train.Saver(write_version = saver_pb2.SaverDef.V2)

# op to write logs to Tensorboard
logs_path = './logs'
summary_writer = tf.compat.v1.summary.FileWriter(logs_path, graph=tf.compat.v1.get_default_graph())

epochs = 10 #30
batch_size = 500 #100

# train over the dataset about 30 times
for epoch in range(epochs):
  for i in range(int(driving_data.num_images/batch_size)):
    xs, ys = driving_data.LoadTrainBatch(batch_size)
    train_step.run(feed_dict={model.x: xs, model.y_: ys, model.keep_prob: 0.8})
    if i % 10 == 0:
      xs, ys = driving_data.LoadValBatch(batch_size)
      loss_value = loss.eval(feed_dict={model.x:xs, model.y_: ys, model.keep_prob: 1.0})
      print("Epoch: %d, Step: %d, Loss: %g" % (epoch, epoch * batch_size + i, loss_value))

    # write logs at every iteration
    summary = merged_summary_op.eval(feed_dict={model.x:xs, model.y_: ys, model.keep_prob: 1.0})
    summary_writer.add_summary(summary, epoch * driving_data.num_images/batch_size + i)

    if i % batch_size == 0:
      if not os.path.exists(LOGDIR):
        os.makedirs(LOGDIR)
      checkpoint_path = os.path.join(LOGDIR, "model.ckpt")
      filename = saver.save(sess, checkpoint_path)
  print("Model saved in file: %s" % filename)

print("Run the command line:\n" \
      "--> tensorboard --logdir=./logs " \
      "\nThen open http://0.0.0.0:6006/ into your web browser")
