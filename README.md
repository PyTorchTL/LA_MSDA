# LA_MSDA
Code release for "Label-based Alignment Multi-Source Domain Adaptation for Cross-subject EEG Fatigue Mental State Evaluation"

The code has been tested on Python 3.7 + PyTorch 1.7. To run the training and testing code, using the following script:

## Setup
* Install PyTorch and dependencies from http://pytorch.org.
* Install Torch vision from the source.
* Install deepsort requirements:
```
pip install -r requirements.txt
```

## Data Loading
./MSDA_DataLoader.py in folder ./DataLoader is used to load the dataset, and split the training and testing data by configuring the auxiliary training data rate in the target domain 

The data and the corresponding labels are returned together by: 
```
./DataLoader/MSDA_LoadTensorData.py
```

## Training
Multi-source domain adaptation training can be launched by running:
```
./main_mutil_sources_LAMSDA.py
```
You can change the 'root_path' and 'domain_list' in 'main_mutil_sources_LAMSDA.py' to set different tasks.
```
The documentation for this task:
optional arguments:
  --batch_size            size of the batches (default:64)
  --iteration             num of the iteration
  --lr                    learning rate
  --class_num             number of classes
  --test_rate             auxiliary training data rate in the target domain
  --namelist              the subjects list of dataset
  --shape                 the shape of each sample, (channel, frequencies)
  --num_source            number of source domains
  --source_name_list      the samples of multi-source domains
  --target_name_list      the samples of target domain
```
