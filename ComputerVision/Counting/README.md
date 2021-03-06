# Counting
1. [Counting approaches](#counting-approaches)
2. [Evaluation metrics](#evaluation-metrics)
3. [Applications](#applications)
4. [Challenges for vision-based object counting](#challenges-for-vision-based-object-counting)
5. [Papers related to counting](#papers-related-to-counting)  
    *  __2019__  
        - [W-Net: Reinforced U-Net for Density Map Estimation (Varun Kannadi Valloli and Kinal Mehta, 2019)](#w-net-reinforced-u-net-for-density-map-estimation-varun-kannadi-valloli-and-kinal-mehta-2019)
        - [Improving Dense Crowd Counting Convolutional Neural Networks using Inverse k-Nearest Neighbor Maps and Multiscale Upsampling (Greg Olmschenk et al., 2019)](#improving-dense-crowd-counting-convolutional-neural-networks-using-inverse-k-nearest-neighbor-maps-and-multiscale-upsampling-greg-olmschenk-hao-tang-and-zhigang-zhu-2019)
        - [Almost Unsupervised Learning for Dense Crowd Counting (Deepak Babu Sam et al., 2019)](#almost-unsupervised-learning-for-dense-crowd-counting-deepak-babu-sam-neeraj-n-sajjan-himanshu-maurya-r-venkatesh-babu-2019)  
    * __2018__
        - [Class-Agnostic Counting (Erika Lu, Weidi Xie and Andrew Zisserman, 2018)](#class-agnostic-counting-erika-lu-weidi-xie-and-andrew-zisserman-2018)
        - [Composition Loss for Counting, Density Map Estimation and Localization in Dense Crowds (Haroon Idrees et al., 2018)](#composition-loss-for-counting-density-map-estimation-and-localization-in-dense-crowds-haroon-idrees-muhmmad-tayyab-kishan-athrey-dong-zhang-somaya-al-maadeed-nasir-rajpoot-and-mubarak-shah-2018)
        - [Iterative Crowd Counting (Viresh Ranjan, Hieu Le and Minh Hoai, 2018)](#iterative-crowd-counting-viresh-ranjan-hieu-le-and-minh-hoai-2018)
        - [Leveraging Unlabeled Data for Crowd Counting by Learning to Rank (Xialei Liu et al., 2018)](#leveraging-unlabeled-data-for-crowd-counting-by-learning-to-rank-xialei-liu-joost-van-de-weijer-and-andrew-d-bagdanov-2018)
        - [CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes (Yuhong Li et al., 2018)](#csrnet-dilated-convolutional-neural-networks-for-understanding-the-highly-congested-scenes-yuhong-li-xiaofan-zhang-and-deming-chen-2018)
        - [Object Counting with Small Datasets of Large Images (Shubhra Aich and Ian Stavness, 2018)](#object-counting-with-small-datasets-of-large-images-shubhra-aich-and-ian-stavness-2018)
        - [Learning Short-Cut Connections for Object Counting (Daniel Oñoro-Rubio et al., 2018)](#learning-short-cut-connections-for-object-counting-daniel-oñoro-rubio-mathias-niepert-and-roberto-j-lópez-sastre-2018)
        - [Improving Object Counting with Heatmap Regulation (Shubhra Aich and Ian Stavness, 2018)](#improving-object-counting-with-heatmap-regulation-shubhra-aich-and-ian-stavness-2018)  
    * __2017__
        - [People, Penguins and Petri Dishes: Adapting Object Counting Models To New Visual Domains And Object Types Without Forgetting (Mark Marsden et al., 2017)](#people-penguins-and-petri-dishes-adapting-object-counting-models-to-new-visual-domains-and-object-types-without-forgetting-mark-marsden-et-al-2017)
        - [Generating High-Quality Crowd Density Maps using Contextual Pyramid CNNs (Vishwanath A. Sindagi and Vishal M. Patel, 2017)](#generating-high-quality-crowd-density-maps-using-contextual-pyramid-cnns-vishwanath-a-sindagi-and-vishal-m-patel-2017)
        - [Representation Learning by Learning to Count (Mehdi Noroozi, Hamed Pirsiavash and Paolo Favaro, 2017)](#representation-learning-by-learning-to-count-mehdi-noroozi-hamed-pirsiavash-and-paolo-favaro-2017)
        - [Drone-based Object Counting by Spatially Regularized Regional Proposal Network (Meng-Ru Hsieh et al., 2017)](#drone-based-object-counting-by-spatially-regularized-regional-proposal-network-meng-ru-hsieh-yen-liang-lin-and-winston-h-hsu-2017)
        - [Divide and Count: Generic Object Counting by Image Divisions (Tobias Stahl et al., 2017)](#divide-and-count-generic-object-counting-by-image-divisions-tobias-stahl-silvia-l-pintea-and-jan-c-van-gemert-2017)
        - [Count-ception: Counting by Fully Convolutional Redundant Counting (Joseph Paul Cohen, Yoshua Bengio et al., 2017)](#count-ception-counting-by-fully-convolutional-redundant-counting-joseph-paul-cohen-genevieve-boucher-craig-a-glastonbury-henry-z-lo-and-yoshua-bengio-2017)  
    * __2016__
        - [Microscopy cell counting and detection with fully convolutional regression networks (Weidi Xie, J. Alison Noble and Andrew Zisserman, 2016)](#microscopy-cell-counting-and-detection-with-fully-convolutional-regression-networks-weidi-xie-j-alison-noble-and-andrew-zisserman-2016)
        - [Counting in The Wild (Carlos Arteta, Victor Lempitsky and Andrew Zisserman, 2016)](#counting-in-the-wild-carlos-arteta-victor-lempitsky-and-andrew-zisserman-2016)
        - [CrowdNet: A Deep Convolutional Network for Dense Crowd Counting (Lokesh Boominathan et al., 2016)](#crowdnet-a-deep-convolutional-network-for-dense-crowd-counting-lokesh-boominathan-srinivas-s-s-kruthiventi-r-venkatesh-babu-2016)  
    * __2015__
        - [Learning to count with deep object features (Santi Seguí, Oriol Pujol and Jordi Vitrià, 2015)](#learning-to-count-with-deep-object-features-santi-seguí-oriol-pujol-and-jordi-vitrià-2015)
        - [Extremely Overlapping Vehicle Counting (Ricardo Guerrero-Gomez-Olmedo et al., 2015)](#extremely-overlapping-vehicle-counting-ricardo-guerrero-gomez-olmedo-et-al-2015)  
    * __2010__
        - [Learning To Count Objects in Images (Victor Lempitsky and Andrew Zisserman, 2010)](#learning-to-count-objects-in-images-victor-lempitsky-and-andrew-zisserman-2010)  
    * __1996__
        - [Detecting, localizing and grouping repeated scene elements from an image (Leung, T., Malik, J., 1996)](#detecting-localizing-and-grouping-repeated-scene-elements-from-an-image-leung-t-malik-j-1996)
6. [Datasets related to counting](#datasets-related-to-counting)
    - [Inria Aerial Image Labeling Dataset](#inria-aerial-image-labeling-dataset)
    - [Airbus Ship Detection Dataset](#airbus-ship-detection-dataset)
    - [NOAA Fisheries Steller Sea Lion Population Dataset](#noaa-fisheries-steller-sea-lion-population-dataset)
    - [VGG Cells Dataset](#vgg-cells-dataset)
    - [CARPK Dataset](#carpk)
    - [PUCPR+ Dataset](#pucpr)
    - [TRANCOS Dataset](#trancos)
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
* Urban planning;
* Behavior analysis in crowd scenes;
* Other applications: counting plants, trees, buildings from aerial images.


# Challenges for vision-based object counting
* overlapping;
* intra-class variation;
* object scale issues;
* perspective issues;
* visual occlusion;
* poor illumination;


# Papers related to counting
## [W-Net: Reinforced U-Net for Density Map Estimation (Varun Kannadi Valloli and Kinal Mehta, 2019)](https://arxiv.org/abs/1903.11249)
* They propose a U-Net inspired architecture (called W-Net) with 3 branches:
  - encoder branch: pre-trained VGG16bn for feature extraction;
  - decoder branch (1): density map estimation (DME) branch;
  - decoder branch (2): reinforcement branch that does binary classification in order to improve the DME and help the network to converge faster.
* They report state-of-the-art results on 3 crowd counting datasets: ShanghaiTech, UCF\_CC\_50, UCSD.
* Remark: Use nearest neighbor interpolation + convolutions for upsampling instead of transposed convolutions which produce checkerboard artifacts.  
![w_net_2019](./images/w_net_2019.png)


## [Improving Dense Crowd Counting Convolutional Neural Networks using Inverse k-Nearest Neighbor Maps and Multiscale Upsampling (Greg Olmschenk, Hao Tang and Zhigang Zhu, 2019)](https://arxiv.org/abs/1902.05379)
* The main ideas of this work:  
  1. Density map labeling is flawed. 
  2. The authors propose an alternative: inverse k-nearest neighbor (ikNN) labeling scheme.
  3. They show that by simply replacing density map training in a NN with their ikNN map training, the testing accuracy of the NN improves.
  4. A new network architecture is proposed: MUD-ikNN, which uses multi-scale upsampling with transposed convs to make effective use of ikNN labeling scheme.
* ikNN:
  - does not explicitly represent crowd density;
  - a single ikNN provides information similar to any arbitrary number of density maps with different Gaussian spreads, in a form which is better suited for NN training;
  - it provides a significant gradient spatially across the entire label and yet still provides precise location information of pedestrians.
* The final count prediction = the mean of all predicted crowd count from the regression modules and the output of the final DenseBlock.  
![iknn_2019](./images/iknn_2019.png)


## [Almost Unsupervised Learning for Dense Crowd Counting (Deepak Babu Sam, Neeraj N Sajjan, Himanshu Maurya, R. Venkatesh Babu, 2019)](http://val.serc.iisc.ernet.in/valweb/papers/AAAI_2019_WTACNN.pdf)
* The authors propose to train a counting CNN in an almost unsupervised manner:  
  a. (Unsupervised) They develop  Grid Winner-Take-All (GWTA) autoencoder to learn useful features from unlabeled images.  
The basic idea is to divide the conv layer in grid cells and within each cell, only the maximally activated neuron is allowed to update the filter.  
  b. (Supervised) Density map regression from the autoencoder features.
The layers trained in unsupervised manner are frozen and only the last 2 layers are tuned with labeled data.
* 99.9% of the parameters of the network are trained with unlabeled images, 0.1% with labeled images.
* Their unsupervised approach outperforms fully supervised training when available labeled data is less.
> The basic idea of winner-take-all (WTA) regularization for autoencoders is to selectively perform learning for neurons in the autoencoder.
This means not all neurons are allowed to update their weights at a particular iteration, creating a race among neurons to learn a feature and get specialized.
WTA autoencoders acquire better features than normal autoencoders.  
> The authors modify WTA training methodology and develop Grid WTA conv autoencoders to handle huge diversity in crowd scenes.  

![gwta_ccnn_2019](./images/gwta_ccnn_2019.png)


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


## [Composition Loss for Counting, Density Map Estimation and Localization in Dense Crowds (Haroon Idrees, Muhmmad Tayyab, Kishan Athrey, Dong Zhang, Somaya Al-Maadeed, Nasir Rajpoot and Mubarak Shah, 2018)](https://arxiv.org/abs/1808.01050)
* 
* 
* They introduce UCF-QNRF dataset, which contains 1201 + 334 dense crowd images with 1,251,642 dot annotations.


## [Iterative Crowd Counting (Viresh Ranjan, Hieu Le and Minh Hoai, 2018)](https://arxiv.org/abs/1807.09959)
* The authors propose iterative counting CNN (ic-CNN), a two-branch architecture for crowd counting using a density estimation approach.
* One branch estimates a low resolution density map and the other generates a high resolution density map.
* They have also proposed a multi-stage pipeline comprising multiple ic-CNNs, where each stage takes into account the predictions from the previous stages.
* Their method achieves the lowest MAE on ShanghaiTech, WorldExpo'10 and UCF datasets.  
![ic_cnn_2018](./images/ic_cnn_2018.png)


## [Leveraging Unlabeled Data for Crowd Counting by Learning to Rank (Xialei Liu, Joost van de Weijer and Andrew D. Bagdanov, 2018)](https://arxiv.org/abs/1803.03095)
* The authors propose an approach that leverages unlabeled crowd imagery in a learning-to-rank framework.  
To induce a ranking of cropped images, they use the following observation: a crowd image contains more or the same number of persons than its cropped sub-images.
* To enforce the ranking, they apply the pairwise ranking hinge loss.  
A typical implementation of the ranking loss would involve a Siamese network with 2 parallel branches that share the same parameters.
* They demonstrate how to efficiently learn from unlabeled datasets by incorporating learning-to-rank in a multi-task network which simultaneously ranks images and estimates crowd density maps.
* Alternating-task and multi-task learning is better than ranking plus fine-tuning:
  - If the network is trained first exclusively on the self-supervised task, you can not be sure it focuses on people.  
     This is probably caused by the poorly-defined nature of the self-supervised task.
  - By jointly learning both the self-supervised and crowd counting tasks, the self-supervised task is forced to focus on counting persons.
* They achieve state-of-the-art results on ShanghaiTech and UCF_CC_50 dataset.  
![learning_to_rank_2018](./images/learning_to_rank_2018.png)


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
* The authors present contextual pyramid of CNNs (CP-CNNs) that incorporates global and local contextual information.
* The global and local contexts are learned by classifying input images/patches into various density levels.
* The contextual information is concatenated (channel-wise) with the output of a multi-column DME and is used by a Fusion-CNN to generate qualitative density maps. 
* Addition of contextual information improves the count error and the quality of density maps.  
![cp_cnn_2017](./images/cp_cnn_2017.png)


## [Representation Learning by Learning to Count (Mehdi Noroozi, Hamed Pirsiavash and Paolo Favaro, 2017)](https://arxiv.org/abs/1708.06734)
* Remark: this paper is more related to representation learning than to counting.
* The authors introduce a novel method to learn representations from data that does not rely on manual annotations.
* Counting is used as a pretext task, which is formalized as a constraint that relates the 'counted' primitives in tiles of an image to those counted in its downsampled version.
This constraint is used to train a NN with a contrastive loss.
* In other words, the main observations used in this paper are the following:
  - (scaling) the number of visual primitives in an image is invariant to scale;
  - (tiling) the number of visual primitives in a whole image is equal to the total number of visual primitives from its sub-regions.
* I think that representations learned in this way can also be used later for semi-supervised counting tasks.
> One way to characterize a feature of interest is to describe how it should vary as a function of changes in the input data.  

![representation_learning_by_learning_to_count_2017](./images/representation_learning_by_learning_to_count_2017.png)


## [Drone-based Object Counting by Spatially Regularized Regional Proposal Network (Meng-Ru Hsieh, Yen-Liang Lin and Winston H. Hsu, 2017)](https://arxiv.org/abs/1707.05972)
* They leverage the spatial layout information (e.g., cars often park regularly) and introduce these spatially regularized constraints into their network to improve the localization accuracy.
* The spatial layout information can be used to improve results of object counting tasks with regularized structures.
* They created the largest (in 2017) drone view dataset CARPK and modified PUCPR dataset to PUCPR+, which can be used for counting tasks.
> The regression-based methods can not generate precise object positions.  
  
![layout_proposal_network_2017](./images/layout_proposal_network_2017.png)


## [Divide and Count: Generic Object Counting by Image Divisions (Tobias Stahl, Silvia L. Pintea and Jan C. van Gemert, 2017)](https://jvgemert.github.io/pub/stahlTIP18counting.pdf)
* The authors propose an end-to-end deep learning method for generic object counting with the following observations:
  - they make no assumption about the object class (does not use any prior category information);
  - their method is not dependent on the region proposal method;
  - the ground truth is represented by the image global count (without localization);
* They introduce a FC layer: Inclusion-Exclusion Principle (IEP) layer to perform the counting optimization independently per image division and to avoid over-counting for highly overlapping image regions.
* Learning over a hierarchy rather than a flat structure helps, as each hierarchy level is independently optimized in the L1 loss.  
![generic_object_counting_by_image_divisions_2017](./images/generic_object_counting_by_image_divisions_2017.png)


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
* The authors use a multi-task approach for learning to count penguins in images.
* The core of the CNN is the segmentation architecture FCN8, initialized from the VGG16, and adds skip and fusion layers for a finer prediction map.
* The proposed architecture includes: 
  - a foreground/background segmentation s(p);
  - an object density function lambda(p);
  - a prediction of the agreement between annotators u(p) as a measure of uncertainty.
* Motivation for the use of multi-task architecture:
  - improvement in generalization;
  - reuse the predicted segmentations to change the objective for other branches as learning progresses.
* They release a dataset that shows a high-degree of object occlusion and scale variation.
  Each image was annotated with dots by 20 different persons.
  The counting accuracy improves as the number of annotators per image increases.
* They also perform up-scale data augmentation (6 different scales), and use 700x700 patches for training.  
![counting_in_the_wild_2016](./images/counting_in_the_wild_2016.png)


## [CrowdNet: A Deep Convolutional Network for Dense Crowd Counting (Lokesh Boominathan, Srinivas S S Kruthiventi, R. Venkatesh Babu, 2016)](https://arxiv.org/abs/1608.06197)
* CrowdNet uses the combination of shallow and deep networks to acquire multi-scale information in density map approximation for crowd counting.
* The main idea is to capture both low-level features and high-level information.
* They also perform multi-scale data augmentation in order to learn scale invariant representations.  
![crowdnet_2016](./images/crowdnet_2016.png)


## [Learning to count with deep object features (Santi Seguí, Oriol Pujol and Jordi Vitrià, 2015)](https://arxiv.org/abs/1505.08082)
* Different from other approaches, the authors don't give any hint on the object they are counting besides the occurrence multiplicity.
  They follow a weakly supervised approach for object counting.
* They cast the object counting to a classification problem (Conv layers + FC layers) where the final FC layer has a fixed size: the maximum number of objects in an image.
  This is a drawback, because this method can be used only for images with few objects and the maximum number of objects has to be known a priori.
* Their experiments suggest that the task of object counting may be used as a surrogate for finding good representations for new tasks. 
> Classical regression functions are prone to errors when the input is high dimensional.  
  
![counting_as_classification_problem_weakly_supervised_2015](./images/counting_as_classification_problem_weakly_supervised_2015.png)


## [Extremely Overlapping Vehicle Counting (Ricardo Guerrero-Gomez-Olmedo _et al._, 2015)](http://agamenon.tsc.uah.es/Personales/rlopez/docs/ibpria2015-guerrero.pdf)
* The authors introduce TRANCOS, a counting dataset with extremely overlapping vehicles.
* They also propose a novel evaluation metric, the Grid Average Mean absolute Error (GAME), that represents the MAE of non-overlapping image regions.
The GAME metric is able to penalize those predictions with a good MAE but a wrong localization of the objects.
* They apply 3 different state-of-the-art (at that moment) counting methods on the TRANCOS dataset.
The results show that counting by regression strategies are more precise localizing and estimating the number of vehicles.  
![overlapping_vehicle_counting_2015](./images/overlapping_vehicle_counting_2015.png)


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


## CARPK Dataset
* https://lafi.github.io/LPN/
* 1448 images (720x1280) (989 for train, 459 for test) of cars captured from different parking lots;
* 90,000 cars;
* maximum number of cars in a single scene: 188;
* ground truth: bounding boxes.  
![carpk_dataset](./images/carpk_dataset.png)


## PUCPR+ Dataset
* https://lafi.github.io/LPN/
* modified annotations of PUCPR (initial dataset);
* 125 images (720x1280) (100 for train, 25 for test) of cars captured from a single parking lot, using fixed camera sensors that are placed in the same place;
* 17,000 cars;
* maximum number of cars in a single scene: 331;
* ground truth: bounding boxes.  
![pucpr_plus_dataset](./images/pucpr_plus_dataset.png)


## TRANCOS Dataset
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
* 330,165 annotated heads;
* part A: 482 (various sizes) images (300 for test, 182 for test) which are randomly crawled from the Internet.
* part B: 716 (768x1024) images (400 for train, 316 for test) which are taken from busy streets.
* ground truth: dot annotations.  
![shanghaitech_dataset](./images/shanghaitech_dataset.png)
