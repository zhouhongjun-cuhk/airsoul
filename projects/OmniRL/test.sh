export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=$PYTHONPATH:/home/wangfan/fangdong/airsoul
export PYTHONPATH="$PYTHONPATH:/home/shaopt/code/main_branch/foudation_model/l3c_baselines"
export PYTHONPATH="$PYTHONPATH:/home/shaopt/code/main_branch/foudation_model"
python generate.py config_test.yaml