# Data Aware Neural Architecture Search
![A graphical representation of the relationship between AutoML, Data Aware Neural Architecture Search, and regular Neural Architecture Search](url "a title")
This is a repository containing a simple Data Aware Neural Architecture Search (Data Aware NAS).

## Concept
The central idea of a Data Aware NAS is to include input data granularity in the search space of a regular Neural Architecture Search.
The term “data granularity” refers to the concept that data can be input into an ML model at different levels of granularity.
E.g., an audio sample can be input to an ML model at various sample rates. 
In this situation, we would say that an audio sample given at a sample rate of 24 kHz will be of a higher data granularity than the same audio sample given at 12 kHz. 
Similarly, an image can be input into an ML model at different resolutions.
The number and type of sensors (e.g., mono vs stereo audio) is another example of data granularity.

## Implementation
The implementation of the Data Aware NAS hosted in this repository uses a Genetic Algorithm to generate combinations of data granularities and neural network architectures.
To estimate performance, the implementation trains the combination for a configurable amount of epochs to record the accuracy, precision, recall, and model size.
These metrics are combined for a total score of a combination.

## Publication
A paper with a more detailed description of the Data Aware NAS concept and this implementation is currently under review and will be linked here after publication.