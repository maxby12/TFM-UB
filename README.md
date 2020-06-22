# Fundamental Principles of Data Science Master’s thesis

Thesis title: Human explainability through an auxiliary Neural Network

This repository stores the final code implemented for the research project as well as its memory, the presentation and all the disclosable datasets.

Code implemented with Python 3, Tensorflow v2.0, Keras v2.3.0 and PyTorch v1.5.0 and the Google Collaboratory environment.

Birds dataset source: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html

# Abstract

Explainability in Deep Learning has become a hot topic in the recent years due to the need of insights and justifications for predictions. Although this field has an extensive range of different approaches, this thesis explores the feasibility of a new methodology that seeks to provide human-interpretable explanations for each sample being processed by a Neural Network. The term black box is often used in the Explainability field, meaning that there is a lack in transparency within the model when processing data. The explored approach tries to deal with the black box by using the outputs of the hidden layers of a Neural Network as inputs for the model responsible for the explanations. This model is another Neural Network that can be seen as an auxiliary Neural Network to the main task. The predicted explanations are formed by a subset of a list of human-designed justifications for the possible outcomes of the main task. Using the predictions from both networks a cross comparison process is also performed in order to build confidence on the main predictions. Results successfully show how a significant proportion of incorrect outputs are questioned thanks to the predicted explanations.
