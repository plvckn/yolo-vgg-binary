## Intro

This project combines the tasks of object detection and image classification. This works in a multi-step process:

1. Object detection model tries to predict bounding boxes for a custom object in an image (object is a *lemon* in this case).
2. Using predicted bounding box data get the crops of the objects from an image.
3. Pass the crops to a classifier to perform binary classification on these crops and predict if a crop belongs to a *lemon* class.

Usually, a well-trained object detection model should be enough, but object detection data is more scarce, and the classifier helps by better distinguishing between similar objects.

Here is a short video demo:

https://user-images.githubusercontent.com/50446201/138952919-e2e404f9-b763-4ab1-b522-fe3d6640451b.mp4

As you can see, the detection model detects objects similar to a lemon, but the classifier on top rightly classifies which objects belong or do not belong to a lemon class. For demonstration purposes, the detection confidence threshold in the video was set to be low (0.2).

## Libraries and other Resources

Objection detection model is a [yolov2](https://arxiv.org/abs/1612.08242) model.

Classifier is a modified [VGG-16](https://arxiv.org/abs/1409.1556) model.

##### Libraries:

- [Darkflow](https://github.com/thtrieu/darkflow) - Implementation of Darknet (original yolo framework) in Tensorflow.
- [mAP](https://github.com/Cartucho/mAP) - Evaluation of object detection performance by calculating mean Average Precision
- [gen_anchors.py](https://github.com/AlexeyAB/darknet/blob/master/scripts/gen_anchors.py) - Script to generate custom anchor boxes by using k-means clustering on your dataset
- [labelImg](https://github.com/tzutalin/labelImg)  - Bounding box labeling tool

##### Datasets:

- [Fruits-262](https://www.kaggle.com/aelchimminut/fruits262) - Images of fruits of 262 classes. Classifier was trained on a part of this dataset by using lemon images as a positive class and a combination of other fruit images as a negative class for a balanced binary classification dataset.
- [Open Images V6](https://storage.googleapis.com/openimages/web/index.html) - Images with bounding box labels. Used to train an object detection model.

## Requirements / How to use

- Clone [Darkflow](https://github.com/thtrieu/darkflow) repository. Build required cython extensions for darkflow by running following command inside darkflow repository: ```python setup.py build_ext --inplace``` 
- Move ```detection/yolov2-voc-1c.cfg``` in this repo to ```darkflow/cfg/``` . Download [weights](https://drive.google.com/drive/folders/1iJ0rHytFkQQ8IJpXNnQtg7TrjUYx4xzx?usp=sharing)  and move both ```yolov2-voc-1c.pb``` and ```yolov2-voc-1c.meta``` to ```darkflow/built_graph/```
- Download classification [model](https://drive.google.com/drive/folders/1OBaGQBSSpj31xi9V3QRDAzYP49aq7P-A?usp=sharing). Exact directory does not matter, you can specify classifier path in ```yolo_vgg_demo.ipynb``` notebook.  
- To process a video (like the one above) follow code and instructions inside ```yolo_vgg_demo.ipynb``` notebook. This can also work for processing images or even real-time demo with a webcam.

## Classifier architecture

The classifier is a VGG-16 pretrained on Imagenet. I used transfer learning to get the weights from convolutional layers and then added global average pooling and a single fully-connected layer with one sigmoid neuron for binary classification. The model was trained until convergence with convolutional layers frozen. Model was trained on fruits-262 dataset.

Finetuning last few convolutional layers did not result in any significant impact.

## YOLO architecture

For object detection, a slightly modified [yolov2-voc](https://github.com/pjreddie/darknet/blob/master/cfg/yolov2-voc.cfg) architecture was used. The model divides an image into 13x13 grid cells and uses 5 different anchor boxes to predict bounding boxes for each grid cell. To detect smaller objects, this architecture concatenates outputs from earlier convolutional layers. 

I tried training models with both default anchors and modified anchors according to my dataset, but the results were slightly better with default anchors. The Average Precision of both types of models were very close (within 1%) .

## Classification and detection results

The classifier evaluates on test set as follows:

```loss: 0.2299 - binary_accuracy: 0.9046 - precision: 0.9798 - recall: 0.9104```

AP results (@IOU=.5, detection threshold=.01):

yolov2 default anchors - ```51.96%```

yolov2 modified anchors - ```51.08%```
