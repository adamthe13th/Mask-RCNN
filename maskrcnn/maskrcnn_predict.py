import mrcnn
import mrcnn.config
import mrcnn.model
import mrcnn.visualize
import cv2
import os
#import PoseEst

# load the class label names from disk, one label per line
# CLASS_NAMES = open("coco_labels.txt").read().strip().split("\n")

CLASS_NAMES = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

class SimpleConfig(mrcnn.config.Config):
    # Give the configuration a recognizable name
    NAME = "coco_inference"
    
    # set the number of GPUs to use along with the number of images per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

	# Number of classes = number of classes + 1 (+1 for the background). The background class is named BG
    NUM_CLASSES = len(CLASS_NAMES)

# Initialize the Mask R-CNN model for inference and then load the weights.
# This step builds the Keras model architecture.
model = mrcnn.model.MaskRCNN(mode="inference", 
                             config=SimpleConfig(),
                             model_dir=os.getcwd())

# Load the weights into the model.
model.load_weights(filepath="mask_rcnn_coco.h5", 
                   by_name=True)


#-----------------added to work more large scale --------------------------------------------
folder_path = "image_test"
answer = str(input("\033[31m Do you want to run the model on all images in the folder test_image? \n No will result in running all the images in image_test \n (y/n): \033[0m")).lower().strip()
if answer[0] != 'y':
    folder_path = "test_image"
# Define a list of valid image extensions (you can add more if needed)
valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']

# Loop through the files in the specified folder
for filename in os.listdir(folder_path):
    # Check if the file is an image by its extension
    if any(filename.endswith(ext) for ext in valid_extensions):
        # Build the full file path
        image_path = os.path.join(folder_path, filename)
# load the input image, convert it from BGR to RGB channel

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Perform a forward pass of the network to obtain the results
    r = model.detect([image])

    # Get the results for the first image.
    r = r[0]

    # Visualize the detected objects.
    mrcnn.visualize.display_instances(image=image, 
                                  boxes=r['rois'], 
                                  masks=r['masks'], 
                                  class_ids=r['class_ids'], 
                                  class_names=CLASS_NAMES, 
                                  scores=r['scores'], out_name=("output/" + filename))

    #run detectron2 on same image
