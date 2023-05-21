echo "Training models..."
echo "MODEL: list length 5"
python3 train.py --device="cuda" --batch_size=100 --num_samples=100000 --pth="model-5.pkl" --num_epochs=20

echo "MODEL: list length 10"
python3 train.py --device="cuda" --batch_size=100 --num_samples=200000 --list_length=10 --pth="model-10.pkl" --num_epochs=20

echo "MODEL: list length 5, variable length"
python3 train.py --device="cuda" --batch_size=100 --num_samples=200000 --list_length=10 --pth="model-10-var.pkl" --num_epochs=20 --var_length
