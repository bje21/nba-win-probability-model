# NBA Game Outcome Prediction

## Overview

This project constructs a pre-game NBA win probability model using team-level historical information available before tip-off. The primary goal is not to maximize prediction accuracy, but to investigate how much predictive signal can be extracted from simple, interpretable features such as prior game outcomes, rest, and point differential.

A logistic regression model is trained to estimate the probability that the home team wins, and its performance is evaluated using both accuracy and probabilistic metrics. The focus is on understanding the limits of team-level data for single-game NBA prediction rather than producing a highly accurate classifier.

## Data

* **Source:** Kaggle NBA dataset (game-level data only)
* **Scope:** All NBA games from 1946–present
* **Target variable:** `home_win` (0 = loss, 1 = win)

Each row represents a single NBA game with team identifiers, game date, and final scores.

## Feature Engineering

All features are constructed using only information available prior to each game.

Engineered features include:

* **Win percentage prior to game**

  * Home and away team win percentages computed using only previous games
  * First game initialized to a neutral value (0.5)

* **Rest days**

  * Days since each team’s previous game
  * First game initialized to 0 rest days

* **Point differential prior to game**

  * Cumulative point differential from previous games
  * Computed separately for home and away teams

* **Differential features**

  * Win percentage difference (home − away)
  * Rest days difference (home − away)
  * Point differential difference (home − away)

* **Home indicator**

  * Binary indicator capturing baseline home-court advantage

## Model

* **Model:** Logistic Regression

* **Rationale:**

  * Logistic regression is well-suited for binary outcomes and produces calibrated probability estimates, making it appropriate for win probability modeling.

* **Train/test split:**
  A time-based split is used to preserve the chronological structure of the data:

  * Early games (first 80%) used for training
  * Later games (last 20%) used for testing

## Evaluation

Model performance is evaluated against a baseline that always predicts a home team win.

Metrics used:

* Accuracy
* Log loss (primary metric)

While overall accuracy remains close to the baseline, the model consistently improves log loss, indicating better-calibrated probability estimates.

Two visualizations are used to analyze model behavior:

* **Distribution of predicted probabilities**

  * Shows that the model assigns a range of win probabilities rather than predicting a constant value.

* **Accuracy vs. predicted probability (binned)**

  * Demonstrates that accuracy generally increases with predicted confidence in the main data region, while extreme bins are noisy due to limited sample size.

These visualizations show that the model meaningfully ranks games by likelihood even when classification accuracy is limited.

## Key Takeaways

* Team-level historical features contain limited but real predictive signal
* Probability calibration improves even when overall accuracy plateaus
* Higher predicted confidence generally corresponds to higher empirical accuracy
* NBA game outcomes remain inherently noisy at the single-game level

## Limitations

* Model uses only team-level aggregates (no player-level data)
* Injuries, rotations, travel distance, and matchup effects are not included
* Predictive power is constrained by the inherent randomness of NBA games

## Next Steps

* Incorporating player-level information or injury indicators
* Comparing model probabilities to betting market odds
* Exploring alternative model types while preserving pre-game information constraints

## Requirements

* Python
* pandas
* numpy
* scikit-learn
* matplotlib

## How to Run

* Clone the repository
* Install requirements
* Run `main.py` to train the model and generate visualizations
