# MHGNN_2020
Meta-path based heterogeneous neural network
We used same datasets as HetGNN to test the performances of our algorithms, see https://github.com/chuxuzhang/KDD2019_HetGNN for dataset and basic usage. Please download data and change file directory before run the code.
We save several different models in model_save,  use model.load_state_dict(torch.load("./model_name.pt")) to load trained models, remember to change args such as embedding size in args.py
Use main.py to run HAN modified from https://github.com/dmlc/dgl/tree/master/examples/pytorch/han.
Use MHGNN.py to run MHGNN and HetGNN modified from https://github.com/chuxuzhang/KDD2019_HetGNN.
Use application.py to run link prediction, venue recommendation, classification and clustering.
