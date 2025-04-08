export CUDA_VISIBLE_DEVICES=7
export PYTHONPATH="$PYTHONPATH: /home/wangfan/cassel/airsoul/"
export PYTHONPATH="$PYTHONPATH:/home/wangfan/cassel/airsoul/airsoul"
python3 train.py config_test.yaml
