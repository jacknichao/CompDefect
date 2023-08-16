output_dir=saved_models
mkdir -p $output_dir
CUDA_VISIBLE_DEVICES=0 python3 run.py --output_dir $output_dir --config config_all.yml 2>&1| tee -a $output_dir/train.log
