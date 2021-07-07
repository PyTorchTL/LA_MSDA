# LA_MSDA
Code release for "Label-based Alignment Multi-Source Domain Adaptation for Cross-subject EEG Fatigue Mental State Evaluation"

The code has been tested on Python 3.7 + PyTorch 1.7. To run the training and testing code, using the following script:

## Setup
1. Download source code from GitHub
   ```
   git clone https://github.com/PyTorchTL/LA_MSDA
   ```
2. Create [conda](https://docs.conda.io/en/latest/miniconda.html) virtual-environment
   ```
   conda create --name LA-MSDA python=3
   ```
3. Activate conda environment
   ```
   source activate LA-MSDA
   ```
4. Install deepsort requirements:
   ```
   pip install -r requirements.txt
   ```

## Data Loading
```./MSDA_DataLoader.py``` in folder ./DataLoader is used to load the dataset, and split the training and testing data by configuring the auxiliary training data rate in the target domain 

The data and the corresponding labels are returned together by: 
```./DataLoader/MSDA_LoadTensorData.py```

EEG data:
1. The EEG data format: ```.mat```
2. The shape of the EEG data: \[61, 27] ```the channel unmber is 61, and the frequency points are 27```
3. Split N-1 subjects of N as the N-1 source domains, in which one subject is regarded as a source domain.


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
It provides the network structure of LA-MSDA, which mainly including stage1 (feature extraction), Stage 2 (local label-based alignment), and stage 3 (global optimization)
```
./LAMSDA_Modules/mutil_sources_modules.py
```
The files of ```./LAMSDA_Modules/multi_sources_LLMMD.py``` and ```./LAMSDA_Modules/multi_sources_LocalWeight.py``` are the modules of Stage 2.
