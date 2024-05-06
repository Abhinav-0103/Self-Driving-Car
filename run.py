import tensorflow as tf
# tf.disable_v2_behavior()
import model
import cv2
from subprocess import call
import os

# Check if on Windows OS
windows = os.name == 'nt'

# Disable TensorFlow v2 behavior
tf.compat.v1.disable_v2_behavior()

# Create a TensorFlow session
sess = tf.compat.v1.Session()

# Restore the model
saver = tf.compat.v1.train.Saver()
saver.restore(sess, "save/model.ckpt")

# Load the steering wheel image
img = cv2.imread('steering_wheel_image.jpg', 0)
rows, cols = img.shape

smoothed_angle = 0

# Video capture initialization
cap = cv2.VideoCapture(0)
while cv2.waitKey(10) != ord('q'):
    ret, frame = cap.read()
    
    # Resize and normalize the frame
    image = cv2.resize(frame, (200, 66)) / 255.0
    
    # Predict the steering angle
    # degrees = sess.run(model.y.eval(feed_dict={model.x: [image], model.keep_prob: 1.0})[0][0] * 180 / 3.14159265)
    degrees = sess.run(model.y, feed_dict={model.x: [image], model.keep_prob: 1.0})[0][0] * 180 / 3.14159265
    
    if not windows:
        # Clear the console
        os.system("clear")
    
    # Print the predicted steering angle
    print("Predicted steering angle: {:.2f} degrees".format(degrees))
    
    # Display the frame
    cv2.imshow('frame', frame)
    
    # Smooth angle transitions
    smoothed_angle += 0.2 * pow(abs((degrees - smoothed_angle)), 2.0 / 3.0) * (degrees - smoothed_angle) / abs(degrees - smoothed_angle)
    
    # Rotate the steering wheel image
    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -smoothed_angle, 1)
    dst = cv2.warpAffine(img, M, (cols, rows))
    cv2.imshow("steering wheel", dst)

cap.release()
cv2.destroyAllWindows()