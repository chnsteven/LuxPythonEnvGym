conda activate luxai
cd ..
cd .\e1\
python train.py

conda activate luxai
cd ..
cd .\e2\
python train.py

conda activate luxai
cd ..
cd .\kaggle_submissions\
lux-ai-2021 --seed=100 agent1/main.py agent2/main.py --maxtime 10000

conda activate luxai
tensorboard --logdir lux_tensorboard