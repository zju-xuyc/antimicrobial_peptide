# Readme

## Data preparation

### First step:  

> Use sequence_generated.py in ./sequence_generated to generate the sequence for your own searching space.

### Second step:  

> Use cal_pep_des.py in ./featured_data_generated to generate structual data for Classification and Ranking stage from the sequences derived in last step.

## Model Training  

### First step:

> Use train.py to get all the params for the three models(Classifcation/Ranking/Regressing). You can use customized training data or data provided in ./data

### Second step: 

> Use lstm_fine_tune.py for incremental learning. You can use customized data validated in other wet-lab settings.

## Searching antimicrobal sequences

> Use predict.py to get the final searching result. For a vast searching space, you may use 'chunk' mechanism to avoid RAM shortage.

