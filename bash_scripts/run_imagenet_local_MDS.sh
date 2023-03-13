cd ..;

# SAVE_PATH="/ssd2/Experiments/generalization/debug";#"/users/local/y17bendo/data/generalization/results";
# FEATURES_PATH="/ssd2/Experiments/metadatasets/features";#"/users/local/y17bendo/data/features"
SAVE_PATH="/users/local/y17bendo/generalization/results";
FEATURES_PATH="/users/local/y17bendo/generalization/features"
MDS="True";
DATASET="metadataset_imagenet";

VALIDATION_DATASET=$DATASET"_validation_with_full_cov_mds"$MDS; #"metadataset_imagenet_validation";#
VALIDATION_FEATURES="no_ME"$DATASET$"_validation_features";#"no_MEminiimagenet_validation";no_MEmetadataset_imagenet_test_features
TEST_DATASET=$DATASET"_test_with_full_cov_mds"$MDS;#"miniimagenet_test";metadataset_vgg_flower_test
TEST_FEATURES="no_ME"$DATASET"_test_features";#"no_MEminiimagenet_test";no_MEmetadataset_vgg_flower_test_features
DEVICE="cpu"
N_RUNS=1000;
N_WAYS=5;
MAXK=50;
UNBALANCED="False";

echo python  main.py --save-folder $SAVE_PATH --maxK $MAXK --features-path $FEATURES_PATH/$VALIDATION_FEATURES.pt --dataset $VALIDATION_DATASET --validation --n-ways $N_WAYS --n-runs $N_RUNS --mds $MDS --device $DEVICE; 
python main.py --save-folder $SAVE_PATH --maxK $MAXK --features-path $FEATURES_PATH/$VALIDATION_FEATURES.pt --dataset $VALIDATION_DATASET --validation --n-ways $N_WAYS --n-runs $N_RUNS --mds $MDS --device $DEVICE; 

echo python  main.py --save-folder $SAVE_PATH --maxK $MAXK --features-path $FEATURES_PATH/$TEST_FEATURES.pt --dataset $TEST_DATASET --config-validation $SAVE_PATH/$VALIDATION_DATASET"/nruns"$N_RUNS"_c"$N_WAYS"_unbalanced"$UNBALANCED"_filename_"$VALIDATION_FEATURES.pt --n-ways $N_WAYS --n-runs $N_RUNS --mds $MDS --device $DEVICE; 
python  main.py --save-folder $SAVE_PATH --maxK $MAXK --features-path $FEATURES_PATH/$TEST_FEATURES.pt --dataset $TEST_DATASET --config-validation $SAVE_PATH/$VALIDATION_DATASET"/nruns"$N_RUNS"_c"$N_WAYS"_unbalanced"$UNBALANCED"_filename_"$VALIDATION_FEATURES.pt --n-ways $N_WAYS --n-runs $N_RUNS --mds $MDS --device $DEVICE; 
#SL6