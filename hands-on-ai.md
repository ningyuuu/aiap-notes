# Hands-On-AI: Intel Academy

## Part 1: Create Applications with Powerful AI Capabilities
Building an AI solution is an elaborate process which requires a myriad of deep technology specialist skills. The process can be divided into 5 main processes:
- Idea: use case and business processes
- Technology: selection of technologies amongst many choices
- Data: how data is collected, analysed, preprocessed and annotated
- Model: build a model upon everything so far
- App: deploy the model into a usable application

## Part 2: Ideation
4 Main steps:
- Step 1: Understand users' needs, but especially considering the size of the company, as they influence the availability of data
- Step 2: Define a problem that can be solved through artificial intelligence
- Step 3: Search for existing projects or products that have used technologies similar to our defined problem
- Step 4: Talk to experts to evaluate the feasibility of the project

## Part 3: The Anatomy of an AI Team
We should build a team based on the needs of the project. Typically, we will need data scientists (especially people with deep learning expertise when necessary), engineering (data, front/back-end applications, devops), domain experts, and management (for larger teams).

Specialists could be found through internal or external channels. We should start the search with an internal process.

## Part 4: Project Planning
- __Formulation__ refers to process where we scope out the project clearly, moving from fuzzy ideas to clear, quantitative operational definitions of our goals and processes. Typically, this involves the formalisation of data labels, training process, and the end product.
- __Methods__: we can consider a hierarchical decomposition of big tasks into smaller parts, what-if analyses and user-journey simulation to discover challenges of users of the product. These processes should be done for both the front and back ends of the application/product

## Part 5: Select a Deep Learning Framework
Out of a large selection of possible frameworks, we choose Tensorflow because 1) it has scripted models with a model zoo, 2) it has a vibrant community, 3) it supports multi-cpu used by Intel, 4) it is easily deployable through Tensorflow Serving.

## Part 6: Select an AI Computing Infrastructure
Between cloud and physical hardware, hardware becomes more cost-friendly as a project's timeline scales (or if they hardware can be used again for future projects), while cloud gives infinite flexibility and scalability at the expense of cost. For projects which largely rely on pre-trained models however, cloud may be more accessible and cheaper as it is largely a short-term project.

## Part 7: Augment AI with Human Intelligence Using Amazon Mechanical Turk
We can use the Amazon Mechanical Turk to basically label data, or create training data that can develop AI. To ensure data validity, we should create redundancy, ask verification questions, ensure qualifications, conduct time cutoffs, or filter by ratings.

## Part 8: Crowdsourcing Word Selection for Image Search
Hands on for part 7.

## Part 9: Data Annotation Techniques
For supervised learning, we would first need a set of labelled data. If the data population required is small, we can label is manually. Otherwise, we can make use of existing APIs, data-driven approaches, or crowdsourcing, all of which may cost more.

We could label data through offline annotation (one shot do all), active learning (based on the needs of the model at that point in time), or weakly supervised, in which we start with weaker models before moving on to strong ones.

## Part 10: Set Up a Portable Experimental Environment for Deep Learning with Docker
We can use docker to deploy an existing environment built for machine learning. Since we are using tensorflow, we should use the tensorflow's docker. This is a hands-on session.

## Part 11: Image Dataset Search
Existing image databases have challenges of bias, scale, copyright and limited applicability. Hence, we decided to manually find a collection of pictures, and find a collection of labels. While pictures were scraped, the keywords for these pictures were condensed through mechanical turks to a set of 20 words for each major emotion. Ultimately, a dataset of images were collected for the key words through Flickr.

## Part 12: Image Data Collection
A Flickr API was used to call for images, but the images proved to be largely unusuable. Hence, the team chose to the GAPED and OASIS datasets, which we deeemed to be higher in quality.

## Part 13: Image Data Exploration
Images were split into 4 main categories: animals, humans, scenes and objects, with the remaining dumped into miscellaneous. In the process of exploration, images were labelled into negative, neutral and positive images.

## Part 14: Image Data Preprocessing and Augmentation
__Preprocessing__: image preprocessing includes things like rescaling (from 0:255 to 0:1), greyscaling, samplewise centering, samplewise std dev.
__Augmentations__: includes rotations (45 degrees), H shift, V shift, shearing (slants), and zooming as well as flips (both V and H). A combination of these modifications produce a larger sample of augmented images.

## Part 15: Overview of Convolutional Neural Networks for Image Classification
In traditional ml, we use statistical models on top of hand-crafted features. However, the engineering of such features require the input of a subject matter expert, which is costly to train or hire. 

Instead, we use conv nets, which use kernels to discover features of images. After many such layers, it then use a dense layer (softmax) to make an effort to conduct the classification.

## Part 16: Modern Deep Neural Network Architectures for Image Classification
- __AlexNet__: the first breakthrough of parallelisation, because it split the process into 2 relatively weak GPUs. It greatly reduced the error rate, but also introduced the ReLU function. It also made use of dropout.
- __ZFNet__: apart from a sigificant improvement in error rates, it helped to visualise kernels, weights, and hidden representations of images for future improvement of CNNs.
- __VGG__: simple filters and simple poolings, but with significant increase in depth. In other words, simple building blocks, but more building blocks.
- __GoogLeNet__: Introduced the inception module, which processes layers in parallel instead of in series to speed up calculation. It also introduced the 1x1 convolution, which is basically a form of dimensionality reduction without decreasing resolution
- __ResNet__: the concept of residuals of a previous layer being added to the next layer, in other words, F(x) => relu => __F(x) + x__ => relu. This approach reduced errors to 3.6%, a huge improvement once again. This structure had 152 layers.

## Part 17: Emotion Recognition from Images Baseline Model
A preliminary look at the data structure reveals that images would be difficult to differentiate. A realistic target for our model would be perhaps an accuracy of about 80%.

A train/validation split can be conducted, so that the validation data can be used to evaluate the model. We do this in the ratio of 4:1. We then implement the preprocessing and augmentation process, before implementing the VGG architecture. We trained the model over 10 epochs, and the validation gave us an accuracy of about 50%, with a AUC of 0.57. The AUC shows us that our model is only slightly better than random.

## Part 18: Emotion Recognition from Images Model Tuning and Hyperparameters
We tackle insufficient data through 4 approaches: resampling, unsupervised learning, data augmentation and transfer learning. 

__Transfer learning__ means using large pretrained models on a different dataset of similar nature. We first select a pretrained model (vgg), then remove its dense layers using the `include_top=False` optional parameter. We then "freeze" the remaining layers by using the cached result of the net,

We use a concept called *global average pooling* to take an average across the spatial dimensions, which eliminates the need for additional weights to condense the layers into dense layers. Using the adam optimiser, we now have to optimise for the learning rate, which we use 0.0001. This gives us an ROC of 0.82, a significant improvement, which also meets our target.









