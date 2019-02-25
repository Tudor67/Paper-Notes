# Counting
1. [Counting approaches](#counting-approaches)
2. [Evaluation metrics](#evaluation-metrics)
3. [Applications](#applications)
4. [Challenges for vision-based object counting](#challenges-for-vision-based-object-counting)
5. [Papers related to counting](#papers-related-to-counting)
    - [Class-Agnostic Counting (Erika Lu, Weidi Xie and Andrew Zisserman, 2018)](#class-agnostic-counting-erika-lu-weidi-xie-and-andrew-zisserman-2018)
    - [CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes (Yuhong Li et al., 2018)](#csrnet-dilated-convolutional-neural-networks-for-understanding-the-highly-congested-scenes-yuhong-li-xiaofan-zhang-and-deming-chen-2018)
    - [Object Counting with Small Datasets of Large Images (Shubhra Aich and Ian Stavness, 2018)](#object-counting-with-small-datasets-of-large-images-shubhra-aich-and-ian-stavness-2018)
    - [Learning Short-Cut Connections for Object Counting (Daniel Oñoro-Rubio et al., 2018)](#learning-short-cut-connections-for-object-counting-daniel-oñoro-rubio-mathias-niepert-and-roberto-j-lópez-sastre-2018)
    - [Improving Object Counting with Heatmap Regulation (Shubhra Aich and Ian Stavness, 2018)](#improving-object-counting-with-heatmap-regulation-shubhra-aich-and-ian-stavness-2018)
    - [People, Penguins and Petri Dishes: Adapting Object Counting Models To New Visual Domains And Object Types Without Forgetting (Mark Marsden et al., 2017)](#people-penguins-and-petri-dishes-adapting-object-counting-models-to-new-visual-domains-and-object-types-without-forgetting-mark-marsden-et-al-2017)
    - [Generating High-Quality Crowd Density Maps using Contextual Pyramid CNNs (Vishwanath A. Sindagi and Vishal M. Patel, 2017)](#generating-high-quality-crowd-density-maps-using-contextual-pyramid-cnns-vishwanath-a-sindagi-and-vishal-m-patel-2017)
    - [Drone-based Object Counting by Spatially Regularized Regional Proposal Network (Meng-Ru Hsieh et al., 2017)](#drone-based-object-counting-by-spatially-regularized-regional-proposal-network-meng-ru-hsieh-yen-liang-lin-and-winston-h-hsu-2017)
    - [Count-ception: Counting by Fully Convolutional Redundant Counting (Joseph Paul Cohen, Yoshua Bengio et al., 2017)](#count-ception-counting-by-fully-convolutional-redundant-counting-joseph-paul-cohen-genevieve-boucher-craig-a-glastonbury-henry-z-lo-and-yoshua-bengio-2017)
    - [Microscopy cell counting and detection with fully convolutional regression networks (Weidi Xie, J. Alison Noble and Andrew Zisserman, 2016)](#microscopy-cell-counting-and-detection-with-fully-convolutional-regression-networks-weidi-xie-j-alison-noble-and-andrew-zisserman-2016)
    - [Counting in The Wild (Carlos Arteta, Victor Lempitsky and Andrew Zisserman, 2016)](#counting-in-the-wild-carlos-arteta-victor-lempitsky-and-andrew-zisserman-2016)
	- [CrowdNet: A Deep Convolutional Network for Dense Crowd Counting (Lokesh Boominathan et al., 2016)](#crowdnet-a-deep-convolutional-network-for-dense-crowd-counting-lokesh-boominathan-srinivas-s-s-kruthiventi-r-venkatesh-babu-2016)
    - [Learning to count with deep object features (Santi Seguí, Oriol Pujol and Jordi Vitrià, 2015)](#learning-to-count-with-deep-object-features-santi-seguí-oriol-pujol-and-jordi-vitrià-2015)
    - [Extremely Overlapping Vehicle Counting (Ricardo Guerrero-Gomez-Olmedo et al., 2015)](#extremely-overlapping-vehicle-counting-ricardo-guerrero-gomez-olmedo-et-al-2015)
    - [Learning To Count Objects in Images (Victor Lempitsky and Andrew Zisserman, 2010)](#learning-to-count-objects-in-images-victor-lempitsky-and-andrew-zisserman-2010)
    - [Detecting, localizing and grouping repeated scene elements from an image (Leung, T., Malik, J., 1996)](#detecting-localizing-and-grouping-repeated-scene-elements-from-an-image-leung-t-malik-j-1996)
6. [Datasets related to counting](#datasets-related-to-counting)
    - [Inria Aerial Image Labeling Dataset](#inria-aerial-image-labeling-dataset)
    - [Airbus Ship Detection Dataset](#airbus-ship-detection-dataset)
    - [NOAA Fisheries Steller Sea Lion Population Dataset](#noaa-fisheries-steller-sea-lion-population-dataset)
    - [VGG Cells Dataset](#vgg-cells-dataset)
    - [CARPK](#carpk)
    - [PUCPR+](#pucpr)
    - [TRANCOS](#trancos)
    - [ShanghaiTech Dataset](#shanghaitech-dataset)


# Counting approaches
1. __Supervised learning__
	- In the supervised case we know the location of the objects we learn to count.
	  I mean that the ground truth contains bounding boxes, dots or segmentation maps for objects of interest.
	- Methods:
        * Counting by detection;
        * Counting by regression;
        * Counting by segmentation;

2. __Weakly supervised learning__
    - Learning to count without giving locations of the objects. The system/method learns from the pair (image, number of objects).

3. __Semi-supervised learning__
	- A combination of supervised learning (few ground truth labels) and unsupervised learning (a lot of images without labels).

4. __Unsupervised learning__
    - Perform grouping based on self-similarities or motion similarities. No ground truth (labels) for objects of interest.


# Evaluation metrics
* __*MAE*__ - Mean Absolute Error;
* __*RMSE*__ - Root Mean Squared Error;	
* __%U__ - Underestimate;
* __%O__ - Overestimate;
* __%D__ - Difference;
* __GAME(L)__ - Grid Average Mean absolute Error (subdivide the image in 4^L non-overlapping regions,
                and compute MAE in each of these subregions);  
![evaluation_metrics](./images/evaluation_metrics.png)


# Applications
* Medicine: determine the quantity of red blood cells and white blood cells to infer the health of a patient;
* Biology: compute the cell concentration in molecular biology to adjust the amount of chemicals to be applied in an experiment;
* Surveillance: investigate crowds in different regions of a city;
* Monitoring: count vehicles in a traffic jam;
* Other applications: counting plants, trees, buildings from aerial images.


# Challenges for vision-based object counting
* overlapping;
* intra-class variation;
* object scale issues;
* perspective issues;
* visual occlusion;
* poor illumination;


# Papers related to counting
## [Class-Agnostic Counting (Erika Lu, Weidi Xie and Andrew Zisserman, 2018)](https://arxiv.org/abs/1811.00472)
* The authors recast object counting as a matching problem.
* They develop a Generic Matching Network (GMN) that learns a discriminative classifier to match instances of a given exemplar.
* The GMN acts as an excellent initialization for counting objects from unseen domains.
* They make use of detection video data to create a model (GMN) that can flexibly adapt to various domains, which is a form of few-shot learning.
* Observations:
  - matching within an image can be thought of as tracking within an image;
  - videos can be a natural data source for learning self-similarity;
  - object counting can have challenging appearance changes:
    * large degrees of rotation;
    * intra-class variation (in the case of cars, both color and shape);  
  
![gmn_2018](./images/gmn_2018.png)


## [CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes (Yuhong Li, Xiaofan Zhang and Deming Chen, 2018)](https://arxiv.org/abs/1802.10062)
* The authors propose CSRNet for crowd counting and high-quality density map generation.
* The architecture: VGG16 (first 10 conv layers) + 6 conv layers (dilation rate=1,2,4).
* They use the dilated convolutional layers to aggregate the multi-scale contextual information in the congested scenes.
* By using dilated convolutions CSRNet can expand the receptive field without losing resolution.  
![csr_net_2018](./images/csr_net_2018.png)


## [Object Counting with Small Datasets of Large Images (Shubhra Aich and Ian Stavness, 2018)](https://arxiv.org/abs/1805.11123)
* One-look regression model for counting: Conv layers + FC/GAP/GSP layers + FC1 (a scalar count).
* The main idea of the one-look regression models for object counting is to utilize weak ground truth information like object counts in the images.
* The authors introduce GSP (Global Sum Pooling) operation as a replacement of GAP (Global Average Pooling) or FC layers.
* They show that GSP exhibits the non-trivial property of generalization for counting objects over variable input shapes, which GAP does not.
* GSP helps to generate more localized activations on object regions (in their experiments, on 4 datasets) just when the model 
is trained with small sub-regions of the images. 
* When training with full-resolution images, both GAP and GSP models result in a less uniform distribution of activation among object regions and less localized activations inside object regions as compared to the GSP model trained with smaller patches.  
![gsp_2018](./images/gsp_2018.png)


## [Learning Short-Cut Connections for Object Counting (Daniel Oñoro-Rubio, Mathias Niepert and Roberto J. López-Sastre, 2018)](https://arxiv.org/abs/1805.02919)
* The authors follow a density estimation approach for counting.
* They propose a modified U-Net architecture with learnable skip-connections, called GU-Net (Gated U-Net).
* GU-Net outperforms the base U-Net, and achieves state-of-the-art performance on various counting datasets.
* The gating strategy leads to more robust models that produce better results.
* Gated short-cut units determine:
  - the amount of information which is passed to other layers;
  - the ways in which this information is combined with the input of these later layers.  
![gu_net_2018](./images/gu_net_2018.png)


## [Improving Object Counting with Heatmap Regulation (Shubhra Aich and Ian Stavness, 2018)](https://arxiv.org/abs/1803.05494)
* They propose to enhance one-look regression counting models by regulating activation maps from the final conv layer of the NN with coarse GT density maps.
* The authors use Smooth-L1 and L1 loss functions to compute CAM (class activation maps) and count errors, respectively.  
![object_counting_with_hr_2018](./images/object_counting_with_hr_2018.png)


## [People, Penguins and Petri Dishes: Adapting Object Counting Models To New Visual Domains And Object Types Without Forgetting (Mark Marsden _et al._, 2017)](https://arxiv.org/abs/1711.05586)
* The authors propose a new multi-domain object counting technique that employs parameter sharing and can be extended to new counting tasks while maintaining identical performance in all prior tasks.
* Each new counting task requires only 20,000 additional model parameters.
* The main idea is to train the network on a specific domain, and after that to retrain just module adapters (other layers are frozen) on different visual domain.
* In other words:  
  - at first stage (for prime visual domain) the network learns the 'domain distribution' and 'counting function';
  - at next stages (for other visual domains) the network learns just the 'domain distribution' (everything else is frozen).
* The residual adapter modules allow to adapt to the distinct statistical distributions of the various visual domains through domain-specific normalization and scaling.
* It is better to train first on a dataset with significant morphological variation among objects (e.g. DCC dataset) that will result in a broader set of learned features, and after that, retrain domain adapters for other classes.
* They also add a refinement network to address the issue of training a patch-based regressor that does not include wide scene context.  
![adapting_to_new_visual_domains_2017](./images/adapting_to_new_visual_domains_2017.png)


## [Generating High-Quality Crowd Density Maps using Contextual Pyramid CNNs (Vishwanath A. Sindagi and Vishal M. Patel, 2017)](https://arxiv.org/abs/1708.00953)
* 
* 
* 


## [Drone-based Object Counting by Spatially Regularized Regional Proposal Network (Meng-Ru Hsieh, Yen-Liang Lin and Winston H. Hsu, 2017)](https://arxiv.org/abs/1707.05972)
* They leverage the spatial layout information (e.g., cars often park regularly) and introduce these spatially regularized constraints into their network to improve the localization accuracy.
* The spatial layout information can be used to improve results of object counting tasks with regularized structures.
* They created the largest (in 2017) drone view dataset CARPK and modified PUCPR dataset to PUCPR+, which can be used for counting tasks.
> The regression-based methods can not generate precise object positions.  
  
![layout_proposal_network_2017](./images/layout_proposal_network_2017.png)


## [Count-ception: Counting by Fully Convolutional Redundant Counting (Joseph Paul Cohen, Genevieve Boucher, Craig A. Glastonbury, Henry Z. Lo and Yoshua Bengio, 2017)](https://arxiv.org/abs/1703.08710)
* Instead of predicting a density map, a redundant counting is proposed in order to average over the errors.
  The idea is to predict a count map which contains redundant counts based on the receptive field of a regression network.
* They also propose a new deep neural network for counting: Count-ception (adapted from the Inception family of networks).
* The Inception-like architecture allows them to obtain multi-scale feature representations and the small input size 32x32 prevents overfitting.
* Their approach results in an improvement (2.9 to 2.3 MAE) over the state-of-the-art method by Xie and Zisserman in 2016.
* Comparison with density map approach (_Learning to count objects in images (Lempitsky, 2015)_):
  - Using Gaussian density map forces the model to predict specific values on how far the object is from the center of the receptive field.
    This is a harder task than just predicting the existence of the object in the receptive field.
  - Redundant counts method is explicitly designed to tolerate the errors when predictions are made (summation over the output of the model).
* Limitations:
  - The predicted count map can localize the regions of the counted objects but not specific coordinates.
* Training details:
  - _Leaky ReLU_. The output can be pushed to zero and then recover to predict the correct count.
  - _Large convolutions_ instead of max_pooling and stride=2 convolutions.
    It is easier to calculate the receptive field of the network.
    Strides add a modulus to the calculation of the count map size.
  - _BatchNorm_ after every convolution.
  - _L1 loss_.  
![count_ception_2017](./images/count_ception_2017.png)
  

## [Microscopy cell counting and detection with fully convolutional regression networks (Weidi Xie, J. Alison Noble and Andrew Zisserman, 2016)](http://www.robots.ox.ac.uk/~vgg/publications/2016/Xie16/xie16.pdf)
* A very good paper in terms of clarity and coherence.
* They use fully convolutional NNs to regress a cell spatial density map that can be used for cell counting and detection.
* They show that FCRNs trained entirely on synthetic data are able to give excellent predictions on real microscopy images.
* The authors show that cell detection can be a side benefit of the cell counting task (based on density estimation, without the need for prior object detection and segmentation).
* An interesting experiment: inverting feature representations (given an encoding of an image, to what extent is it possible to reconstruct that image?)
  - when the networks get deeper, feature representations for cell clump become increasingly abstract (e.g., concavity information);
  - reconstruction quality decreases with the depth of the network, and only important information has been kept by deep layers.
* Details:
  - They pre-train FCRNs with 100x100 patches and fine-tune the parameters with whole images to smooth the estimated density maps.
  - When counting cells from large cell clumps, a larger receptive field is more important than being able to provide more detailed information over the receptive field.  
![fcrns_2016](./images/fcrns_2016.png)
![fcrns_inverting_feature_representations_2016](./images/fcrns_inverting_feature_representations_2016.png)


## [Counting in The Wild (Carlos Arteta, Victor Lempitsky and Andrew Zisserman, 2016)](https://www.robots.ox.ac.uk/~vgg/publications/2016/Arteta16/arteta16.pdf)
* 
* 
* 


## [CrowdNet: A Deep Convolutional Network for Dense Crowd Counting (Lokesh Boominathan, Srinivas S S Kruthiventi, R. Venkatesh Babu, 2016)](https://arxiv.org/abs/1608.06197)
* CrowdNet uses the combination of shallow and deep networks to acquire multi-scale information in density map approximation for crowd counting.
* 
* 


## [Learning to count with deep object features (Santi Seguí, Oriol Pujol and Jordi Vitrià, 2015)](https://arxiv.org/abs/1505.08082)
* Different from other approaches, the authors don't give any hint on the object they are counting besides the occurrence multiplicity.
  They follow a weakly supervised approach for object counting.
* They cast the object counting to a classification problem (Conv layers + FC layers) where the final FC layer has a fixed size: the maximum number of objects in an image.
  This is a drawback, because this method can be used only for images with few objects and the maximum number of objects has to be known a priori.
* Their experiments suggest that the task of object counting may be used as a surrogate for finding good representations for new tasks. 
> Classical regression functions are prone to errors when the input is high dimensional.  
  
![counting_as_classification_problem_weakly_supervised_2015](./images/counting_as_classification_problem_weakly_supervised_2015.png)


## [Extremely Overlapping Vehicle Counting (Ricardo Guerrero-Gomez-Olmedo _et al._, 2015)](http://agamenon.tsc.uah.es/Personales/rlopez/docs/ibpria2015-guerrero.pdf)
* 
*
* 


## [Learning To Count Objects in Images (Victor Lempitsky and Andrew Zisserman, 2010)](http://papers.nips.cc/paper/4043-learning-to-count-objects-in-images.pdf)
* Summary: http://www.robots.ox.ac.uk/~vgg/research/counting/index_new.html
* The authors propose to count objects in images through density estimation.
* The main idea is to estimate a continuous density function whose integral over any image region gives the count of objects within that region.
  In other words, each predicted object takes up a density of 1, so a sum of the density map will reveal the total number of objects in the image.
* Advantages of this approach:
  - avoids the hard task of learning to detect individual object instances;
  - is robust to crowding, overlap and size of the instances;
* Pipeline:
  1. Extract feature vectors (SIFT) at each pixel of the image;
  2. Learn a linear mapping from feature vector at each pixel to a density value, obtaining density function value in that pixel.  
  They use as loss the MESA (_Maximum Excess over SubArrays_) distance.  
  In their next paper (Interactive object counting, 2014), the loss is changed and the mapping coefficients are learned through a simple ridge regression.  
* In other words:
  - Density estimation with a supervised algorithm: 
  - __D(x) = c.T x phi(x)__  
    * D(x): ground-truth density map;  
	* phi(x): the local features;  
	* Parameters __c__ are learned by minimizing the error between the true and predicted density map with quadratic programming over all possible subwindows.  
* The MESA distance has the following (good/desirable) properties:
  - tolerates the local modifications (noise, jitter, change of Gaussian kernel) => robustness to the additive local perturbations;
  - reacts strongly to the change in the number of objects or their positions.  
![counting_through_density_estimation_2010](./images/counting_through_density_estimation_2010.png)


## [Detecting, localizing and grouping repeated scene elements from an image (Leung, T., Malik, J., 1996)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.36.2193&rep=rep1&type=pdf)
* An unsupervised approach for detecting and grouping repeated scene elements in an image.
* Their method can be seen like tracking an element to spatially neighboring locations in one image.
* Pipeline:
    1. _Detection of interesting windows_ (distinctive elements - possible candidates for the repeating elements) in the image.
	   A 2D candidate pattern/window is defined by the authors in a such way that first 2 eigenvalues of their 2nd moment matrix are large and comparable.
	2. _Finding matches_ and _estimating the affine transform_ between matching elements.
	   Two neighboring patches match if their sum of squared differences is small (<threshold).
	3. _Growing the pattern_ following the criterion that will decrease the matching error among the neighboring patches from step 2.
	4. _Grouping elements_ by looking at its 8 neighboring windows.  
![detecting_grouping_repeated_scene_elements_1996](./images/detecting_grouping_repeated_scene_elements_1996.png)


# Datasets related to counting
## Inria Aerial Image Labeling Dataset
* https://project.inria.fr/aerialimagelabeling/
* 180 tiff images (5000x5000) with ground truth (for 5 cities).
* ground truth: segmentation maps for buildings.  
![inria_aerial_dataset](./images/inria_aerial_dataset.png)


## Airbus Ship Detection Dataset
* https://www.kaggle.com/c/airbus-ship-detection/data
* 29 GB;
* 150,000 jpeg images (768x768) extracted from satellite imagery;
* images of tankers, commercial and fishing ships of various shapes and sizes;
* some images do not contain ships, but those that do may contain multiple ships;
* ground truth: oriented bounding boxes around the ships.  
![airbus_ship_detection_dataset](./images/airbus_ship_detection_dataset.png)


## NOAA Fisheries Steller Sea Lion Population Dataset
* https://www.kaggle.com/c/noaa-fisheries-steller-sea-lion-population-count/data
* 96 GB;
* aerial images of sea lions;
* ground truth: colored dots over the animals.  
![sea_lion_population_dataset](./images/sea_lion_population_dataset.png)


## VGG Cells Dataset
* http://academictorrents.com/details/b32305598175bb8e03c5f350e962d772a910641c
* a synthetic dataset;
* 200 png images (256x256) containing simulated bacterial cells from fluorescence-light microscopy;
* each image contains 174 +- 64 cells which overlap;
* ground truth: dot annotations.  
![vgg_cells_dataset](./images/vgg_cells_dataset.png)


## CARPK
* https://lafi.github.io/LPN/
* 1448 images (720x1280) (989 for train, 459 for test) of cars captured from different parking lots;
* 90,000 cars;
* maximum number of cars in a single scene: 188;
* ground truth: bounding boxes.  
![carpk_dataset](./images/carpk_dataset.png)


## PUCPR+
* https://lafi.github.io/LPN/
* modified annotations of PUCPR (initial dataset);
* 125 images (720x1280) (100 for train, 25 for test) of cars captured from a single parking lot, using fixed camera sensors that are placed in the same place;
* 17,000 cars;
* maximum number of cars in a single scene: 331;
* ground truth: bounding boxes.  
![pucpr_plus_dataset](./images/pucpr_plus_dataset.png)


## TRANCOS
* http://agamenon.tsc.uah.es/Personales/rlopez/data/trancos/
* benchmark for (extremely overlapping) vehicle counting in traffic congestion situations;
* 1,244 (480x640) images;
* 46,796 annotated vehicles;
* remark: motorcycles are also annotated;
* ground truth: dot annotations and masks that define regions of interest used for evaluation.  
![trancos_dataset](./images/trancos_dataset.png)


## ShanghaiTech Dataset
* https://github.com/desenzhou/ShanghaiTechDataset
* crowd counting dataset;
* part A: 482 (various sizes) images (300 for test, 182 for test).
* part B: 716 (768x1024) images (400 for train, 316 for test).
* ground truth: dot annotations.  
![shanghaitech_dataset](./images/shanghaitech_dataset.png)