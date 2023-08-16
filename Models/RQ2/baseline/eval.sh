cd ..
output_dir='test_log'
mkdir -p $output_dir
CUDA_VISIBLE_DEVICES=2 python3 -m baseline.run --output_dir $output_dir --config baseline/eval.yml
