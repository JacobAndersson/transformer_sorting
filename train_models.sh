echo "Training models..."
echo "MODEL: list length 5"
python3 train.py --device="cuda" --batch_size=100 --num_samples=100000 --pth="model-5-2.pkl" --num_epochs=20 --run_name='length-5' --use_wandb

echo "MODEL: list length 10"
python3 train.py --device="cuda" --batch_size=100 --num_samples=200000 --list_length=10 --pth="model-10-2.pkl" --num_epochs=40 --run_name='length-10' --use_wandb

echo "MODEL: list length 5, variable length"
python3 train.py --device="cuda" --batch_size=100 --num_samples=200000 --list_length=10 --pth="model-10-var2.pkl" --num_epochs=150 --var_length --run_name='var-length-10' --use_wandb
