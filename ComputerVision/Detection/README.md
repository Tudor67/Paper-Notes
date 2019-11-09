
# Detection
1. [Papers related to object detection](#papers-related-to-object-detection)  
    *  __2018__
	    - [YOLOv3: An Incremental Improvement (Joseph Redmon and Ali Farhadi, 2018)](#yolov3-an-incremental-improvement-joseph-redmon-and-ali-farhadi-2018)
	*  __2017__
	    - [Focal Loss for Dense Object Detection (Tsung-Yi Lin et al., 2017)](#focal-loss-for-dense-object-detection-tsung-yi-lin-priya-goyal-ross-girshick-kaiming-he-and-piotr-dollar-2017)
	*  __2016__
	    - [YOLO9000: Better, Faster, Stronger (Joseph Redmon and Ali Farhadi, 2016)](#yolo9000-better-faster-stronger-joseph-redmon-and-ali-farhadi-2016)
	    - [SSD: Single Shot MultiBox Detector (Wei Liu et al., 2016)](#ssd-single-shot-multibox-detector-wei-liu-dragomir-anguelov-dumitru-erhan-christian-szegedy-scott-reed-cheng-yang-fu-and-alexander-c-berg-2016)
	    - [You Only Look Once: Unified, Real-Time Object Detection (Joseph Redmon et al., 2016)](#you-only-look-once-unified-real-time-object-detection-joseph-redmon-santosh-divvala-ross-girshick-and-ali-farhadi-2016)
	*  __2015__
	    - [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks (Shaoqing Ren et al., 2015)](#faster-r-cnn-towards-real-time-object-detection-with-region-proposal-networks-shaoqing-ren-kaiming-he-ross-girshick-and-jian-sun-2015)
        - [Fast R-CNN (Ross Girshick, 2015)](#fast-r-cnn-ross-girshick-2015)
	*  __2014__
	    - [Rich feature hierarchies for accurate object detection and semantic segmentation (Ross Girshick et al., 2014)](#rich-feature-hierarchies-for-accurate-object-detection-and-semantic-segmentation-ross-girshick-jeff-donahue-trevor-darrell-and-jitendra-malik-2014)


# Papers related to object detection
## [YOLOv3: An Incremental Improvement (Joseph Redmon and Ali Farhadi, 2018)](https://arxiv.org/abs/1804.02767)
*
*


## [Focal Loss for Dense Object Detection (Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He and Piotr Dollar, 2017)](https://arxiv.org/abs/1708.02002)
*
*


## [YOLO9000: Better, Faster, Stronger (Joseph Redmon and Ali Farhadi, 2016)](https://arxiv.org/abs/1612.08242)
*
*


## [SSD: Single Shot MultiBox Detector (Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed, Cheng-Yang Fu and Alexander C. Berg, 2016)](https://arxiv.org/abs/1512.02325)
*
*


## [You Only Look Once: Unified, Real-Time Object Detection (Joseph Redmon, Santosh Divvala, Ross Girshick and Ali Farhadi, 2016)](https://arxiv.org/abs/1506.02640)
*
*


## [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks (Shaoqing Ren, Kaiming He, Ross Girshick and Jian Sun, 2015)](https://arxiv.org/abs/1506.01497)
*
*


## [Fast R-CNN (Ross Girshick, 2015)](https://arxiv.org/abs/1504.08083)
*
*


## [Rich feature hierarchies for accurate object detection and semantic segmentation (Ross Girshick, Jeff Donahue, Trevor Darrell and Jitendra Malik, 2014)](https://arxiv.org/abs/1311.2524)
* The authors propose a simple method for object detection called R-CNN (Regions with CNN features):
  1. Extract ~2000 region proposals using selective search method;
  2. Compute features for each proposal using a large CNN (AlexNet);
  3. Classify each region using class-specific linear SVMs;
  4. Apply class-specific bounding box regressors on each detection to improve the localization performance.
* Training:
  1. Supervised pre-training on ILSVRC2012 classification dataset;
  2. Domain-specific fine-tuning (only on warped region proposals):
      * Replace 1000-way classification layer with a (N + 1)-way classification layer (N is the number of object classes, plus 1 for background);
	  * Region proposals with >= 0.5 IoU overlap with a GT box are considered positives for that box's class and the rest as negatives;
	  * Construct a mini-batch: 32 positive windows (uniformly sampled over all classes) + 96 background windows;
	  * Fine-tune with a learning rate of 0.001 (1/10th of the initial pre-training rate).
  3. Object category classifiers (train N different SVM classifiers):
      * Extract features from the fine-tuned CNN and train one linear SVM per class;
	  * The positive and negative examples are defined differently at this stage (compared to fine-tuning):
	      - Only the GT bounding boxes are positive examples for their respective classes (== 1.0 IoU);
		  - Proposals with < 0.3 IoU overlap with all instances of a class are negative examples for that class;
		  - Proposals with IoU in [0.3, 1.0) are ignored.
	  * For PASCAL VOC datasets, adopt standard hard negative mining for a faster convergence (=> mAP stops increasing after only a single pass over all images).
  4. Bounding box regression:
      * Train a linear regression model (for each class separately) to predict a new detection window given the predicted bounding box and its pool5 features;
      * This improves the localization performance of the model.
  - Different definitions of positives and negatives in fine-tuning and SVM training allow:
      * at fine-tuning stage: to create more positive examples (x30) => a larger dataset => avoids overfitting, but the network is not fine-tuned for precise localization;
	  * at SVM training stage: to emphasize precise localization.
* Test-time detection:
  1. Use selective search on the test image to extract ~2000 region proposals;
  2. Warp each region proposal to a fixed size (227x227);
  3. Extract a 4096 feature vector for each warped region proposal (with CNN: AlexNet);
  4. For each class, score each extracted feature vector using SVM trained for that class;
  5. Given all scored regions in an image, apply greedy non-maximum suppression (NMS) (for each class independently):
      * reject a region if there is another region with a higher score and their IoU is larger than a learned threshold;
	  * for each class independently == apply this method only for regions from the same class;
	  * NMS removes duplicates and false positives.
* Problems of the proposed R-CNN:
  - Training is a slow multi-stage process;
  - Inference is very slow: 47s/image when VGG16 is used for feature extraction;
  - Region proposals are category-independent (the generation method is fixed and cannot be adapted for specific domains).
* State-of-the-art results (in 2014) on the following datasets:
  - PASCAL VOC 2012 (53.3% mAP);
  - ILSVRC2013 (31.4% mAP).  
![rcnn_2014](./images/rcnn_2014.png)  
[Image source](http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture11.pdf)

