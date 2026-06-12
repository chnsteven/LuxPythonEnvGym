conda activate luxai
cd ..
cd .\e1\
python train.py
python train.py --reward_policy

conda activate luxai
cd ..
cd .\e2\
python train.py

conda activate luxai
cd ..
cd .\kaggle_submissions\
lux-ai-2021 agent1/main.py agent2/main.py --maxtime 10000
python evaluate.py

conda activate luxai
tensorboard --logdir lux_tensorboard