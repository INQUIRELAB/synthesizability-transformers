import os
import argparse
from src.training.train import main

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model for materials synthesizability prediction')
    parser.add_argument('--config', type=str, default='configs/default.yaml', help='Path to config file')
    args = parser.parse_args()
    
    # Suppress warnings
    os.environ["PYTHONWARNINGS"] = "ignore"  # Suppress Python warnings
    
    main(args.config) 