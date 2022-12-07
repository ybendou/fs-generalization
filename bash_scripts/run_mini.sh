cd ..;

SAVE_PATH="results";
FEATURES_PATH="features"

# validation set
VALIDATION_DATASET="miniimagenet_validation";
VALIDATION_FEATURES="mini11miniimagenet_validation_features";

# Test set
TEST_DATASET="miniimagenet_test";
TEST_FEATURES="mini11miniimagenet_test_features";

N_RUNS=1000; # Number of runs
N_WAYS=5; # Number of classes
MAXK=50; # Max number of samples for the few-shot problems.
UNBALANCED="False";

# Run the validation set (different classes)
echo python  main_bias_estimate.py --save-folder $SAVE_PATH --maxK $MAXK --features-path $FEATURES_PATH/$VALIDATION_FEATURES.pt --dataset $VALIDATION_DATASET --validation --n-ways $N_WAYS --n-runs $N_RUNS; 
python  main_bias_estimate.py --save-folder $SAVE_PATH --maxK $MAXK --features-path $FEATURES_PATH/$VALIDATION_FEATURES.pt --dataset $VALIDATION_DATASET --validation --n-ways $N_WAYS --n-runs $N_RUNS; 

# Run the test set
echo python  main_bias_estimate.py --save-folder $SAVE_PATH --maxK $MAXK --features-path $FEATURES_PATH/$TEST_FEATURES.pt --dataset $TEST_DATASET --config-validation $SAVE_PATH/$VALIDATION_DATASET"/nruns"$N_RUNS"_c"$N_WAYS"_unbalanced"$UNBALANCED"_filename_"$VALIDATION_FEATURES.pt --n-ways $N_WAYS --n-runs $N_RUNS; 
python  main_bias_estimate.py --save-folder $SAVE_PATH --maxK $MAXK --features-path $FEATURES_PATH/$TEST_FEATURES.pt --dataset $TEST_DATASET --config-validation $SAVE_PATH/$VALIDATION_DATASET"/nruns"$N_RUNS"_c"$N_WAYS"_unbalanced"$UNBALANCED"_filename_"$VALIDATION_FEATURES.pt --n-ways $N_WAYS --n-runs $N_RUNS; 