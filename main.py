import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import numpy as np
import matplotlib.pyplot as plt

games = pd.read_csv("data/raw/game.csv")

games["home_win"] = (games["pts_home"] > games["pts_away"]).astype(int)

model_df = games[
    [
        "game_id",
        "game_date",
        "season_id",
        "team_id_home",
        "team_id_away",
        "home_win",
        "pts_home",
        "pts_away"
    ]
].copy()

model_df["game_date"] = pd.to_datetime(model_df["game_date"])
model_df = model_df.sort_values("game_date")

model_df["home_win_pct_so_far"] = 0.5

for team in model_df["team_id_home"].unique():
    team_games = model_df[model_df["team_id_home"] == team].copy()
    team_games = team_games.sort_values("game_date")

    team_games["wins_before"] = team_games["home_win"].shift(1).fillna(0)
    team_games["total_wins_before"] = team_games["wins_before"].cumsum()
    team_games["games_played_so_far"] = range(len(team_games))

    team_games["win_pct_so_far"] = (
        team_games["total_wins_before"] / team_games["games_played_so_far"]
    ).fillna(0.5)

    model_df.loc[team_games.index, "home_win_pct_so_far"] = team_games["win_pct_so_far"]

model_df["away_win"] = 1 - model_df["home_win"]
model_df["away_win_pct_so_far"] = 0.5

for team in model_df["team_id_away"].unique():
    team_games = model_df[model_df["team_id_away"] == team].copy()
    team_games = team_games.sort_values("game_date")

    team_games["wins_before"] = team_games["away_win"].shift(1).fillna(0)
    team_games["total_wins_before"] = team_games["wins_before"].cumsum()
    team_games["games_played_so_far"] = range(len(team_games))

    team_games["win_pct_so_far"] = (
        team_games["total_wins_before"] / team_games["games_played_so_far"]
    ).fillna(0.5)

    model_df.loc[team_games.index, "away_win_pct_so_far"] = team_games["win_pct_so_far"]

model_df["win_pct_diff"] = model_df["home_win_pct_so_far"] - model_df["away_win_pct_so_far"]
model_df["home_indicator"] = 1

model_df["home_rest_days"] = 0

for team in model_df["team_id_home"].unique():
    team_games = model_df[model_df["team_id_home"] == team].copy()
    team_games = team_games.sort_values("game_date")

    team_games["rest_days"] = (team_games["game_date"] - team_games["game_date"].shift(1)).dt.days
    team_games["rest_days"] = team_games["rest_days"].fillna(0)

    model_df.loc[team_games.index, "home_rest_days"] = team_games["rest_days"]

model_df["away_rest_days"] = 0

for team in model_df["team_id_away"].unique():
    team_games = model_df[model_df["team_id_away"] == team].copy()
    team_games = team_games.sort_values("game_date")

    team_games["rest_days"] = (team_games["game_date"] - team_games["game_date"].shift(1)).dt.days
    team_games["rest_days"] = team_games["rest_days"].fillna(0)

    model_df.loc[team_games.index, "away_rest_days"] = team_games["rest_days"]

model_df["rest_days_diff"] = model_df["home_rest_days"] - model_df["away_rest_days"]

model_df["home_pt_diff_so_far"] = 0

for team in model_df["team_id_home"].unique():
    team_games = model_df[model_df["team_id_home"] == team].copy()
    team_games = team_games.sort_values("game_date")

    team_games["pt_diff_before"] = (team_games["pts_home"] - team_games["pts_away"]).shift(1).fillna(0)
    team_games["total_pdiff_before"] = team_games["pt_diff_before"].cumsum()

    model_df.loc[team_games.index, "home_pt_diff_so_far"] = team_games["total_pdiff_before"]

model_df["away_pt_diff_so_far"] = 0

for team in model_df["team_id_away"].unique():
    team_games = model_df[model_df["team_id_away"] == team].copy()
    team_games = team_games.sort_values("game_date")

    team_games["pt_diff_before"] = (team_games["pts_away"] - team_games["pts_home"]).shift(1).fillna(0)
    team_games["total_pdiff_before"] = team_games["pt_diff_before"].cumsum()

    model_df.loc[team_games.index, "away_pt_diff_so_far"] = team_games["total_pdiff_before"]

model_df["pt_diff_diff"] = model_df["home_pt_diff_so_far"] - model_df["away_pt_diff_so_far"]

X = model_df[["win_pct_diff", "rest_days_diff", "pt_diff_diff", "home_indicator"]]
y = model_df["home_win"]

split_index = int(len(model_df) * 0.8)
X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

model = LogisticRegression()
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("Test accuracy:", accuracy)

print("Coefficient:", model.coef_)
print("Intercept:", model.intercept_)

baseline_accuracy = y_test.mean()
print("Baseline accuracy (always predit home win):", baseline_accuracy)

probs = model.predict_proba(X_test)[:, 1]
print("Log loss:", log_loss(y_test, probs))

baseline_probs = np.full(len(y_test), y_train.mean())
print("Baseline log loss:", log_loss(y_test, baseline_probs))

p = pd.Series(probs, index=y_test.index)
edges = np.arange(0.45, 0.76, 0.05)
bins = pd.cut(p, bins=edges, include_lowest=True)

df = pd.DataFrame({
    "prob":p,
    "actual":y_test
})
df["bin"] = bins

df["prediction"] = (df["prob"] > 0.5).astype(int)
df["correct"] = (df["prediction"] == df["actual"])

summary = df.groupby("bin")["correct"].agg(["mean", "count"])
summary = summary.reset_index()
summary["mid"] = summary["bin"].apply(lambda x: x.mid)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 8))

ax1.hist(probs, bins=30)
ax1.set_title("Distribution of predicted home win probabilities")
ax1.set_xlabel("Predicted home win probability")
ax1.set_ylabel("Number of games")

ax2.plot(summary["mid"], summary["mean"], marker="o", label="Accuracy in bin")
ax2.axhline(baseline_accuracy, linestyle="--", label="Baseline accuracy")
ax2.set_xlabel("Predicted home win probability (bin midpoint)")
ax2.set_ylabel("Accuracy")
ax2.set_title("Accuracy vs predicted probability (binned)")

ax2b = ax2.twinx()
ax2b.bar(
    summary["mid"],
    summary["count"],
    width=0.045,
    alpha=0.3,
    label="Games in bin"
)
ax2b.set_ylabel("Number of games")

lines, labels = ax2.get_legend_handles_labels()
bars, bar_labels = ax2b.get_legend_handles_labels()
ax2.legend(lines + bars, labels + bar_labels, loc="upper right")

plt.tight_layout()
plt.savefig("figures/model_evaluation.png", dpi=150, bbox_inches="tight")
plt.show()