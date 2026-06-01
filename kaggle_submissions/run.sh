conda activate luxai
cd ..
cd .\examples\
python train.py

conda activate luxai
cd ..
cd .\kaggle_submissions\
lux-ai-2021 --seed=100 agent1/main.py agent2/main.py --maxtime 10000

tensorboard --logdir lux_tensorboard