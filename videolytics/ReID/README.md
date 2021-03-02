# Module for re-identification of object from camera streams

For the user guide please refer the thesis. 

The main files in the project are:

* `annotator.py` -- file containing all annotators (generators of feature vectors)
* `explore.py` -- annotation tool
* `trajectory_generator` -- module responsible for creating identities and trajectories  
* `best_neural_network.h5` -- best neural network we trained during our research
  it can be used for example via `external-neural-network-annotator`
* `evaluation_final.py`, `generate_roc_features.py`, `generate_roc_identities.py` -- 
  auxiliary scripts used for evaluations
  
Aside from the main files the project contains additional files with various utilities
needed in main modules.