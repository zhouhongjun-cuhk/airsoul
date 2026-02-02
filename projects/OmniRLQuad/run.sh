export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export PYTHONPATH="$PYTHONPATH: /home/wangfan/cassel/airsoul/"
#export PYTHONPATH="$PYTHONPATH:/home/wangfan/cassel/airsoul/airsoul"
python3 train.py config_test.yaml
