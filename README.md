# A Statistical Model for Predicting Generalization in Few-Shot Classification

Official implementation for "A Statistical Model for Predicting Generalization in Few-Shot Classification" accepted at EUSIPCO 2023.

To run the code for different datasets, first download the features. We use the features proposed in the article https://arxiv.org/pdf/2201.09699.pdf which can be downloaded from the following [link](https://drive.google.com/drive/folders/1fALYAfzStWXasI-DTl6qi9moNuWbFA-j) and can be put in the the folder "features".

Then, run the following Bash script: 
```
./bash_scripts/run_mini.sh
```
If one is interested in changing elements of the runs, you can specify the parameters and run the following commands:

```
SAVE_PATH="results";
FEATURES_PATH="features"

# validation set
VALIDATION_DATASET="miniimagenet_validation";
VALIDATION_FEATURES="mini11miniimagenet_validation_features";

# Test set
TEST_DATASET="miniimagenet_test";
TEST_FEATURES="mini11miniimagenet_test_features";

N_RUNS=1000; #Number of few-shot problems
N_WAYS=5; #Number of classes
MAXK=50; #Max number of samples
UNBALANCED="False";
MDS="True";

# First run the validation split
python  main_bias_estimate.py --save-folder $SAVE_PATH --maxK $MAXK --features-path $FEATURES_PATH/$VALIDATION_FEATURES.pt --dataset $VALIDATION_DATASET --validation --n-ways $N_WAYS --n-runs $N_RUNS --mds $MDS; 

# Run on the test set
python  main_bias_estimate.py --save-folder $SAVE_PATH --maxK $MAXK --features-path $FEATURES_PATH/$TEST_FEATURES.pt --dataset $TEST_DATASET --config-validation $SAVE_PATH/$VALIDATION_DATASET"/nruns"$N_RUNS"_c"$N_WAYS"_unbalanced"$UNBALANCED"_filename_"$VALIDATION_FEATURES.pt --n-ways $N_WAYS --n-runs $N_RUNS --mds $MDS;  
````
