# Data Aware Neural Architecture Search
<img width="1278" alt="A graphical representation of the relationship between Data Aware NAS, AutoML, and regular NAS" src="https://user-images.githubusercontent.com/22741414/212304892-cddd05cc-18ac-42c6-b0f5-9cfc31e2bc77.png">

This is a repository containing a simple Data Aware Neural Architecture Search (Data Aware NAS).

## Concept
The central idea of a Data Aware NAS is to include input data granularity in the search space of a regular Neural Architecture Search.
The term “data granularity” refers to the concept that data can be input into an ML model at different levels of granularity.
E.g., an audio sample can be input to an ML model at various sample rates. 
In this situation, we would say that an audio sample given at a sample rate of 24 kHz will be of a higher data granularity than the same audio sample given at 12 kHz. 
Similarly, an image can be input into an ML model at different resolutions.
The number and type of sensors (e.g., mono vs stereo audio) is another example of data granularity.

## Implementation
<img width="840" alt="A diagram showing the architecture of the Data Aware NAS implementation. Simply visualises what is written below." src="https://user-images.githubusercontent.com/22741414/212305113-56bc7bab-1ba1-4d94-bb62-4a60e767502d.png">
The implementation of the Data Aware NAS hosted in this repository uses a Genetic Algorithm to generate combinations of data granularities and neural network architectures.
To estimate performance, the implementation trains the combination for a configurable amount of epochs to record the accuracy, precision, recall, and model size.
These metrics are combined for a total score of a combination.

## Publication
A paper with a more detailed description of the Data Aware NAS concept and this implementation is currently under review and will be linked here after publication.
