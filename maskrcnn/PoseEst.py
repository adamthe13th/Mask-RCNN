import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np

# Load the pre-trained PoseNet model from TensorFlow Hub
model = hub.load('https://tfhub.dev/google/movenet/singlepose/lightning/4')

# Function to process the image for PoseNet
def process_image(image_path):
    image = cv2.imread(image_path)
    input_image = tf.image.resize_with_pad(np.expand_dims(image, axis=0), 192, 192)
    input_image = tf.cast(input_image, dtype=tf.int32)
    return image, input_image

# Perform pose estimation
def estimate_pose(image_path):
    image, input_image = process_image(image_path)
    outputs = model.signatures["serving_default"](input_image)
    keypoints = outputs['output_0'].numpy()[0, 0, :, :]

    # Draw keypoints on the image
    for keypoint in keypoints:
        y, x, confidence = keypoint
        if confidence > 0.5:
            cv2.circle(image, (int(x * image.shape[1]), int(y * image.shape[0])), 5, (0, 255, 0), -1)

    cv2.imshow("Pose Estimation", image)
    cv2.waitKey(0)

# Call pose estimation on an image
#estimate_pose("path/to/image.jpg")

