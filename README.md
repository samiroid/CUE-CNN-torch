# CUE-CNN Torch with user embeddings
## Scripts
### Training the Model
- main.sh trains the model for a given set of hyper-parameter

### Hyper-Paramter Search
- param_search.sh searches for the best set of hyper-parameter and stores the performance of various sets of hyper-parameters in a txt file

### Pre-training User Embeddings 
- train_u2v.sh trains U2V for one set of given user hyper-parameters
- hyp_store_u2v uses different sets of hyper-parameters to train U2V embeddings and stores them in a separate folder that will be used by the following code.
- hyperopt_u2v.sh searches for the best set of hyper-parameters to train user embeddings with. It stores the performance of various sets of hyper-parameters in a txt file