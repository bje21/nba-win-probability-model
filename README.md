# NBA Game Outcome Prediction

## Overview
This project constructs a pre-game NBA win probability model using team-level historical information available before tip-off. The primary goal was not to maximize prediction accuracy but to investigate how much predictive signal can be extracted from simple, interpretable features like prior game outcomes, rest, and point differential. 

A logistic regression model is trained to estimate the probability that the home team wins, and its performance is determined by evaluating both accuracy and probabilistic metrics. Again, the focus is not to maximize the accuracy of the prediction model. The idea is to understand the limits of using team-level data for single-game predictions. 

## Data
- Source: Kaggle NBA dataset (game-level data only)
- Scope: All games from 1946-present
- Target variable: home_win (0 for loss, 1 for win)

Each row represents a single NBA game with team identifiers, game date, and final scores.

## Feature Engineering
All features are constructed to avoid data leakage by only using data available before each game.
Engineered features include:
- Win percentage prior to game
 - Home and away team win percentages computed using only previous games
 - First game initialized to default value (0.5)
- Rest days
 - Days since each team's previous game
 - First game initialized to default value (0)
- Point differential prior to game
 - Cumlative point differential from previous games
 - Computed separately for home and away teams
- Differential features
 - Win percentage difference (home - away)
 - Rest days difference (home - away)
 - Point differential difference (home - away)
- Home indicator
 - Binary indicator capturing baseline home-court advantage 

## Model
- Model: Logistic regression
- Rationale:
 - Logistic regression is ideal for binary outcomes and provides calibrated probability estimates, making it well-suited for modeling win-probability
- Train/test split:
A time-based split is used to preserve the chronological structure of the data and to prevent data leakage into training from future games:
 - Early games (first 80%) used for training
 - Later games (last 20%) used for testing

## Evaluation
Model performance is evaluated against a baseline that always predicts a home win. 

Metrics used:
- Accuracy
- Log loss (primary)

While accuracy remains close to the baseline, the model consistently improves log loss, indicating better probability estimates.

Two visualizations are used to better understand model behavior:
- Distribution of predicted probabilities
 - Illustrates that the model assigns a range of win probabilities rather than just predicting a home win.
- Accuracy vs. predicted probability (binned)
 - Demonstrates that accuracy increases with predicted confidence in the main data region, while extreme bins on either side are noisy due to limited sample size. 

These figures show that the model meaningfully ranks games by likelihood even when classification accuracy is limited.

## Key Takeaways
- While it is limited, team-level historical features contain some valuable predictive signal
- Probability calibration improves even when accuracy is limited
- Higher predictive confidence generally corresponds to higher empirical accuracy
- NBA game outcomes remain inherently "noisy" at a single-game level

## Limitations
- Model only uses team-level aggregates (no player data)
- Injuries, rotations, travel distance, and matchup effects are not used
- Predictive ability is limited by the inherent randomness of NBA games

## Next Steps
- Incorporating player-level data or factoring in injuries
- Comparing model probabilities to betting odds
- Exploring different types of models while preserving leakage safety

## Requirements
- Python
- pandas
- numpy
- scikit-learn
- matplotlib

## How to Run
- Clone repository
- Install requirements
- Run main.py to train model and generate visualizations

