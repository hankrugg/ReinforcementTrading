# Reinforcement Learning Trading Bot  

## Overview  
This project is a reinforcement learning-based trading bot designed to make automated trading decisions in financial markets. Using a Proximal Policy Optimization (PPO) model, the bot continuously learns from market data to optimize its trading strategy over time. The primary objective is to maximize portfolio returns while managing risk.  

## Features  
- **Reinforcement Learning Model:** Utilizes PPO for decision-making.  
- **Action Space:** Supports buy, sell, and hold actions.  
- **Reward Function:** Based on portfolio value changes to encourage profitable trades.  
- **Live:** Can be deployed for live trading.  
- **Customizable Strategy Parameters:** Allows fine-tuning of hyperparameters and model architecture.  

## Prerequisites  
Before running the project, ensure you have the following installed:  
- [Anaconda](https://www.anaconda.com/)  

## Installation  
### 1. Clone the Repository  
```sh
git clone https://github.com/hankrugg/ReinforcementTrading.git
cd ReinforcementTrading 
```

### 2. Create and Activate a Conda Environment  
```sh
conda create --name rl-trading python=3.8 -y  
conda activate rl-trading  
```

### 3. Install Dependencies  
```sh
pip install -r requirements.txt  
```

## Usage  

### 1. Train the Model  
Run the training script:  
```sh
python TrainShortBot.py  
```
This will process market data and update the RL model over time.  


### 2. Deploy for Live Trading  
To run the bot in a live market, ensure you have API access to a brokerage and execute:  
```sh
python RunBot.py  
```
This can be set up using cron jobs in a Linux environment to fully automate all aspects.

## Customization  
- Modify the reward function in to experiment with different trading objectives.  
- Integrate additional technical indicators for improved feature engineering.  

## Future Improvements  
- Enhance risk management techniques to minimize drawdowns.  
- Add multi-asset trading support.  

