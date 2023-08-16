cd ..
output_dir=saved_models
mkdir -p $output_dir
CUDA_VISIBLE_DEVICES=0  python3 -m baseline.run --output_dir $output_dir --config baseline/config.yml 2>&1| tee -a  $output_dir/train.log