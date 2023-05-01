# gpt_experiments
playing around with transformers


Requirements: 
1. Poetry Installed (`brew install poetry` on mac) 
2. MPS-enabled PyTorch, (can change the device to cpu/cuda in the relevant script if wanting to run on non apple m1/m2 based systems. 

Project's using poetry for dependency management. 

to setup: run `poetry install` 


Main GPT Demo: 
Runnning a simple transformer implementation on a custom stack exchange dataset. 

Reqs for GPT: 
1. `mkdir data`
2. add data science stack exchange dataset or other stackexchange dataset from https://archive.org/details/stackexchange

Can run the demo in the experimentation ipython notebook. 

