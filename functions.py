# -*- coding: utf-8 -*-
"""
PARTS 2-4: Data Processing and Model Training

this file handles everything after the webscraping step:
  - Part 2: takes the raw kaggle tournament results and attaches actual team names
  - Part 3: combines the tournament data with our scraped stats to build two datasets
             (a mega dataset with winner/loser framing, and a model dataset with Team1/Team2)
  - Part 4: trains all 4 models -- logistic regression, random forest, gradient boosting, neural net

all training is kicked off from MarchMadness.ipynb which imports these functions.
input files needed: MMHistoricResults.csv, MTeams.csv, RegSeasonStats.csv
"""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# file paths -- you can change these if your csvs are somewhere differnt
GAMES_PATH  = "MMHistoricResults.csv"
TEAMS_PATH  = "MTeams.csv"
STATS_PATH  = "RegSeasonStats.csv"
MEGA_OUT    = "MarchMadness_MegaDataset_2010_2025.csv"
MODEL_OUT   = "MarchMadness_ModelDataset_2010_2025.csv"


# ===========
# part 2: attach team names to the kaggle tournament game records
# ========

def build_historic_results():
    """
    the raw kaggle tournament file only has numeric team IDs, not names.
    this function joins in the team name from MTeams.csv for both the
    winner and loser of each game, then saves the updated file.

    it returns the games dataframe with WTeamName and LTeamName columns added.
    """
    games = pd.read_csv(GAMES_PATH)
    teams = pd.read_csv(TEAMS_PATH)

    # only keep 2010 onward to match our scraped stats range
    games = games[(games["Season"] >= 2010) & (games["Season"] <= 2026)]

    # merge in the winner name
    games = games.merge(
        teams[["TeamID", "TeamName"]], how="left", left_on="WTeamID", right_on="TeamID"
    )
    games = games.rename(columns={"TeamName": "WTeamName"}).drop(columns=["TeamID"])

    # merge in the loser name (same process, different column)
    games = games.merge(
        teams[["TeamID", "TeamName"]], how="left", left_on="LTeamID", right_on="TeamID"
    )
    games = games.rename(columns={"TeamName": "LTeamName"}).drop(columns=["TeamID"])

    games.to_csv("MMHistoricResultsNEW.csv", index=False)
    print("Saved MMHistoricResultsNEW.csv")
    return games


# ============
# part 3: build the actual datasets we use for modeling
# ======================

def strip_seed_from_team(s):
    """
    KenPom sometimes appends a seed number with an asterisk to the team name
    like 'Duke 1*' for tournament teams. This just strips that off so we
    get a clean team name we can match against other sources.
    """
    s = str(s).strip()
    parts = s.split()
    last_part = parts[-1]
    # check if the last token is a number with an asterisk (e.g. '1*', '14*')
    if last_part.replace("*", "").isdigit() and "*" in last_part:
        parts = parts[:-1]
    return " ".join(parts)


def base_norm(s):
    """
    lowercases and strips punctuation from a team name so we can compare
    across datasets that use slightly different formatting. For example
    'St. Mary's' and 'St Marys' should match after this.
    """
    s = strip_seed_from_team(s).lower().strip()
    s = s.replace("&", "and")
    s = re.sub(r"[\.\'']", "", s)   # remove periods and apostrophes
    s = s.replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


# manually built lookup table for team names that still don't match after normalization.
# these were found by going through mismatches and fixing them one by one 
# mostly abbreviations that kenpom uses that kaggle doesn't (e.g. 'MTSU' vs 'Middle Tennessee')
ALIAS = {
    "ark pine bluff": "arkansas pine bluff",
    "st marys ca": "saint marys",
    "ut san antonio": "utsa",
    "wku": "western kentucky",
    "st louis": "saint louis",
    "nc aandt": "north carolina aandt",
    "fgcu": "florida gulf coast",
    "suny albany": "albany",
    "n dakota st": "north dakota st",
    "sf austin": "stephen f austin",
    "mt st marys": "mount st marys",
    "mtsu": "middle tennessee",
    "st josephs pa": "saint josephs",
    "tx southern": "texas southern",
    "f dickinson": "fairleigh dickinson",
    "abilene chr": "abilene christian",
    "tam c christi": "texas aandm corpus chris",
    "fl atlantic": "florida atlantic",
    "grambling": "grambling st",
    "mcneese st": "mcneese",
    "etsu": "east tennessee st",
    "ark little rock": "arkansas little rock",
    "n colorado": "northern colorado",
    "boston univ": "boston university",
    "st peters": "saint peters",
    "ms valley st": "mississippi valley st",
    "s dakota st": "south dakota st",
    "southern univ": "southern",
    "northwestern la": "northwestern st",
    "w michigan": "western michigan",
    "wi milwaukee": "milwaukee",
    "american univ": "american",
    "nc central": "north carolina central",
    "e kentucky": "eastern kentucky",
    "g washington": "george washington",
    "coastal car": "coastal carolina",
    "e washington": "eastern washington",
    "cs bakersfield": "cal st bakersfield",
    "wi green bay": "green bay",
    "n kentucky": "northern kentucky",
    "kent": "kent st",
    "prairie view": "prairie view aandm",
    "cs fullerton": "cal st fullerton",
    "se missouri st": "southeast missouri st",
    "col charleston": "charleston",
    "college of charleston": "charleston",
    "kennesaw": "kennesaw st",
    "st francis pa": "saint francis",
    "ne omaha": "nebraska omaha",
    "louisiana lafayette": "louisiana",
}


def team_key(s):
    """
    combines base_norm and the alias table into one lookup.
    this is what we actually call on every team name before joining datasets.
    this returns the standardized name we use as the join key.
    """
    k = base_norm(s)
    return ALIAS.get(k, k)  # return alias if one exists, otherwise the normalized name


def build_datasets(games):
    """
    builds two datasets from the tournament game records + scraped regular season stats:

    MEGA dataset: one row per tournament game, winner/loser framing preserved,
       with each team's regular season stats attached and difference columns added.
       Good for analysis and exploration.

    Model dataset: same games but re-framed as Team1 vs Team2 (lower TeamID = Team1)
       so the labeling is neutral -- Team1Win is our target variable.
       This is what we actually train the models on.

    The difference columns (e.g. barthag_diff = Team1_barthag - Team2_barthag) are
    the key features for the ML models since they capture relative team strength.

    this returns the model dataframe and the list of base stat column names.
    """
    stats = pd.read_csv(STATS_PATH)

    # only keep seasons we actually have tournament results for
    stats = stats[stats["Season"].isin(games["Season"].unique())].copy()

    # build a clean join key for every team in both datasets
    stats["team_clean"] = stats["team"].apply(strip_seed_from_team)
    stats["team_key"]   = stats["team_clean"].apply(team_key)
    games["W_team_key"] = games["WTeamName"].apply(team_key)
    games["L_team_key"] = games["LTeamName"].apply(team_key)

    # figure out which columns are actual stat features vs identifiers
    non_feature = {"Season", "year", "team_key", "team_clean", "team", "conf", "w_l", "record"}
    base_cols = [c for c in stats.columns if c not in non_feature]

    #  build the mega dataset (winner/loser framing) ----

    # prefix all stat columns with W_ or L_ then merge onto each game
    Wstats = stats.add_prefix("W_")
    Lstats = stats.add_prefix("L_")

    mega = games.merge(Wstats, left_on=["Season", "W_team_key"], right_on=["W_Season", "W_team_key"], how="left")
    mega = mega.merge(Lstats, left_on=["Season", "L_team_key"], right_on=["L_Season", "L_team_key"], how="left")

    # add winner loser difference columns for each stat
    # these are useful for seeing how much better the winner was in each category
    winner_data, loser_data = {}, {}
    for col in base_cols:
        wc = "W_" + col
        if wc in mega.columns:
            winner_data[col] = pd.to_numeric(mega[wc], errors="coerce")
    for col in base_cols:
        lc = "L_" + col
        if lc in mega.columns:
            loser_data[col] = pd.to_numeric(mega[lc], errors="coerce")

    for col in base_cols:
        if col in winner_data and col in loser_data:
            # only create the diff if both teams actually have data for this stat
            if winner_data[col].notna().any() and loser_data[col].notna().any():
                mega[col + "_diff"] = winner_data[col] - loser_data[col]

    mega["Winner"] = mega["WTeamName"]
    mega["Loser"]  = mega["LTeamName"]
    mega["winner_first"] = 1  # flag so we remember winner is always listed first

    # reorder columns so the important ones are upfront
    front = ["Season", "DayNum", "WTeamID", "WTeamName", "WScore",
             "LTeamID", "LTeamName", "LScore", "Winner", "Loser", "WLoc", "NumOT"]
    front = [c for c in front if c in mega.columns]
    mega = mega[front + [c for c in mega.columns if c not in front]]
    mega.to_csv(MEGA_OUT, index=False)
    print(f"Mega dataset saved: {MEGA_OUT} {mega.shape}")

    # ---- build the model dataset (Team1/Team2 neutral framing) ----
    # we assign Team1 = the team with the lower TeamID so the framing is
    # consistent and not biased toward winners always being "Team1"

    model_games = games.copy()
    t1_is_winner = model_games["WTeamID"] < model_games["LTeamID"]

    model_games["Team1Name"]  = np.where(t1_is_winner, model_games["WTeamName"], model_games["LTeamName"])
    model_games["Team2Name"]  = np.where(t1_is_winner, model_games["LTeamName"], model_games["WTeamName"])
    model_games["Team1ID"]    = np.where(t1_is_winner, model_games["WTeamID"],   model_games["LTeamID"])
    model_games["Team2ID"]    = np.where(t1_is_winner, model_games["LTeamID"],   model_games["WTeamID"])
    model_games["Team1Score"] = np.where(t1_is_winner, model_games["WScore"],    model_games["LScore"])
    model_games["Team2Score"] = np.where(t1_is_winner, model_games["LScore"],    model_games["WScore"])
    model_games["Team1_key"]  = model_games["Team1Name"].apply(team_key)
    model_games["Team2_key"]  = model_games["Team2Name"].apply(team_key)
    model_games["Team1Win"]   = np.where(t1_is_winner, 1, 0)  # target variable

    # attach stats for each team
    T1 = stats.add_prefix("T1_")
    T2 = stats.add_prefix("T2_")
    model_df = model_games.merge(T1, left_on=["Season", "Team1_key"], right_on=["T1_Season", "T1_team_key"], how="left")
    model_df = model_df.merge(T2, left_on=["Season", "Team2_key"], right_on=["T2_Season", "T2_team_key"], how="left")

    # add Team1 - Team2 difference columns -- these are the actual features the models use
    for c in base_cols:
        t1c, t2c = "T1_" + c, "T2_" + c
        if t1c in model_df.columns and t2c in model_df.columns:
            n1 = pd.to_numeric(model_df[t1c], errors="coerce")
            n2 = pd.to_numeric(model_df[t2c], errors="coerce")
            if n1.notna().any() and n2.notna().any():
                model_df[c + "_diff"] = n1 - n2

    # reorder so key columns are at the front
    front2 = ["Season", "DayNum", "Team1ID", "Team1Name", "Team1Score",
              "Team2ID", "Team2Name", "Team2Score", "Team1Win",
              "WTeamName", "LTeamName", "WScore", "LScore"]
    model_df = model_df[front2 + [c for c in model_df.columns if c not in front2]]
    model_df.to_csv(MODEL_OUT, index=False)
    print(f"Model dataset saved: {MODEL_OUT} {model_df.shape}")

    return model_df, base_cols


# ============
# part 4: model training
# =======================

def run_baselines(df):
    """
    run three simple seed based baselines before we touch any ML
    good for checking and gives us a floor to beat

    1. seed only logistic regression: just uses the seed difference as one feature
    2. deterministic lower seed wins: always picks the better seed, no model needed
    3. empirical matchup rates: looks up historical win rates for each seed matchup
       (e.g. how often does a 5 seed beat a 12 seed historically)

    The empirical matchup baseline is the strongest of the three and is what
    we use as our main comparison point against the ML models.
    """
    data = df.copy()

    # seeds come in as strings/floats sometimes so force them to int
    data["T1_seed"] = pd.to_numeric(data["T1_seed"], errors="coerce")
    data["T2_seed"] = pd.to_numeric(data["T2_seed"], errors="coerce")
    data = data.dropna(subset=["T1_seed", "T2_seed", "Team1Win"])
    data["T1_seed"] = data["T1_seed"].astype(int)
    data["T2_seed"] = data["T2_seed"].astype(int)

    data["seed_diff"] = data["T1_seed"] - data["T2_seed"]
    data["lo_seed"]   = data[["T1_seed", "T2_seed"]].min(axis=1)
    data["hi_seed"]   = data[["T1_seed", "T2_seed"]].max(axis=1)
    # lo_won = 1 if the lower seed won, used to build the empirical rates table
    data["lo_won"] = np.where(
        data["T1_seed"] < data["T2_seed"], data["Team1Win"], 1 - data["Team1Win"]
    )

    train, test = train_test_split(data, test_size=0.2, random_state=42, stratify=data["Team1Win"])

    # baseline 1: logistic regression with just seed difference
    seed_model = LogisticRegression(max_iter=1000)
    seed_model.fit(train[["seed_diff"]], train["Team1Win"])
    seed_preds = seed_model.predict(test[["seed_diff"]])
    print(f"Seed Only Accuracy:          {accuracy_score(test['Team1Win'], seed_preds):.2%}")

    # baseline 2: just always pick the lower seed number (better seed)
    test_no_ties = test[test["seed_diff"] != 0].copy()  # remove equal seed matchups
    det_preds = (test_no_ties["seed_diff"] < 0).astype(int)
    print(f"Lower-seed-wins baseline:    {accuracy_score(test_no_ties['Team1Win'], det_preds):.2%}")

    # baseline 3: use historical win rates by seed matchup
    # only built on training data to avoid leakage
    rates = (
        train.groupby(["lo_seed", "hi_seed"])
        .agg(games=("lo_won", "count"), lo_wins=("lo_won", "sum"))
        .reset_index()
    )
    rates["win_rate"] = rates["lo_wins"] / rates["games"]

    def predict_matchup(row):
        lower_seed  = min(row["T1_seed"], row["T2_seed"])
        higher_seed = max(row["T1_seed"], row["T2_seed"])
        matchup = rates[(rates["lo_seed"] == lower_seed) & (rates["hi_seed"] == higher_seed)]
        # if we've seen fewer than 20 historical games for this matchup, just pick the better seed
        if matchup.empty or matchup["games"].values[0] < 20:
            prob = 1 if row["T1_seed"] < row["T2_seed"] else 0
        else:
            lwr = matchup["win_rate"].values[0]
            prob = lwr if row["T1_seed"] < row["T2_seed"] else 1 - lwr
        return int(prob > 0.5)

    test_no_ties["matchup_pred"] = test_no_ties.apply(predict_matchup, axis=1)
    print(f"Empirical matchup baseline:  {accuracy_score(test_no_ties['Team1Win'], test_no_ties['matchup_pred']):.2%}")

    return train, test


def prepare_features(df):
    """
    pulls out the feature columns we actually use for ML.
    we use only the _diff columns (Team1 stat minus Team2 stat) and
    explicitly exclude seed difference -- we want the model to learn
    from actual performance stats, not just seeding.

    this returns X (features), y (target), and the list of column names used.
    """
    diff_cols = [col for col in df.columns if col.endswith("_diff")]
    # drop seed diff -- we don't want the model to just learn to pick lower seeds
    diff_cols_no_seed = [c for c in diff_cols if "seed" not in c.lower()]
    X = df[diff_cols_no_seed].copy()
    y = df["Team1Win"].copy()
    return X, y, diff_cols_no_seed


def train_logistic(X_train, X_test, y_train, y_test):
    """
    trains a logistic regression model on the diff features.
    max_iter bumped up to 5000 because the default 100 doesn't always converge
    with this many features. Prints accuracy and top 20 coefficients by magnitude
    so we can see which stats matter most.
    """
    model = LogisticRegression(max_iter=5000, random_state=42).fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"Logistic Regression accuracy: {accuracy_score(y_test, y_pred):.2%}")

    # print coefficients sorted by absolute value -- bigger = more influential
    coef_df = pd.DataFrame({"Feature": X_train.columns, "Coefficient": model.coef_[0]})
    coef_df["AbsValue"] = coef_df["Coefficient"].abs()
    print(coef_df.sort_values("AbsValue", ascending=False).head(20).to_string(index=False))
    return model


def train_random_forest(X_train, X_test, y_train, y_test):
    """
    trains a random forest classifier. Key hyperparameter choices:
    - max_depth=5: we found deeper trees overfit pretty badly on this dataset
    - n_estimators=500: more trees = more stable, diminishing returns past this
    - class_weight='balanced': games aren't perfectly 50/50 so this helps
    - min_samples_split=10: prevents splits on tiny subsets

    prints both train and test accuracy so we can check for overfitting.
    """
    rf = RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        max_depth=5,          # 5 seemed to be the sweet spot -- deeper overfit
        min_samples_split=10,
        class_weight="balanced"
    ).fit(X_train, y_train)

    rf_acc    = accuracy_score(y_test, rf.predict(X_test))
    train_acc = accuracy_score(y_train, rf.predict(X_train))
    print(f"Random Forest train accuracy: {train_acc:.2%}")
    print(f"Random Forest test  accuracy: {rf_acc:.2%}")

    # feature importances from the forest
    imp = pd.DataFrame({"Feature": X_train.columns, "Importance": rf.feature_importances_})
    print("Top 10 RF features:")
    print(imp.sort_values("Importance", ascending=False).head(10).to_string(index=False))
    return rf


def train_gradient_boosting(X_train, X_test, y_train, y_test):
    """
    trains a gradient boosting classifier. We kept the settings conservative
    to avoid overfitting -- low learning rate, shallow trees, and subsampling:
    - learning_rate=0.05: slow learner but generalizes better
    - max_depth=2: very shallow stumps, reduces overfitting
    - subsample=0.8: uses 80% of data per tree, adds some randomness
    - min_samples_leaf=5: stops splits that only cover a few samples

    also prints AUC since accuracy alone can be misleading for this kind of problem.
    """
    gb = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,          # shallow trees generalize better here
        min_samples_leaf=5,
        subsample=0.8,        # stochastic gradient boosting
        random_state=42
    ).fit(X_train, y_train)

    gb_acc    = accuracy_score(y_test, gb.predict(X_test))
    train_acc = accuracy_score(y_train, gb.predict(X_train))
    gb_auc    = roc_auc_score(y_test, gb.predict_proba(X_test)[:, 1])
    print(f"Gradient Boosting train accuracy: {train_acc:.2%}")
    print(f"Gradient Boosting test  accuracy: {gb_acc:.2%}")
    print(f"Gradient Boosting AUC:            {gb_auc:.3f}")

    imp = pd.DataFrame({"Feature": X_train.columns, "Importance": gb.feature_importances_})
    print("Top 10 GB features:")
    print(imp.sort_values("Importance", ascending=False).head(10).to_string(index=False))
    return gb


# ---- neural network architecture ----

class Net(nn.Module):
    """
    simple feedforward network for binary classification.
    architecture: input -> 64 -> 32 -> 1 with ReLU activations and dropout.
    dropout at 0.3 on both hidden layers to reduce overfitting.
    sigmoid on the output so we get a probability between 0 and 1.
    """
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32),        nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1),         nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze()


def train_neural_network(X_train, X_test, y_train, y_test):
    """
    trains the neural network. A few things worth noting:
    - Features are standardized first (StandardScaler) since the diff columns
      are on very different scales and NNs are sensitive to that
    - Adam optimizer with lr=1e-3, stepped down by half every 50 epochs
    - 40 epochs with batch size 32 -- enough to converge without overfitting
    - Seeds are set for reproducibility

    this returns the trained model, the fitted scaler (needed to transform new data),
    the test tensor, and the raw output probabilities on the test set.
    """
    # standardize features -- important for NNs since features are on different scales
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)  # fit only on train to avoid leakage

    # convert to pytorch tensors
    X_train_t = torch.tensor(X_tr_s, dtype=torch.float32)
    X_test_t  = torch.tensor(X_te_s, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values if hasattr(y_train, "values") else y_train, dtype=torch.float32)
    y_test_t  = torch.tensor(y_test.values  if hasattr(y_test,  "values") else y_test,  dtype=torch.float32)

    loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)

    # set seeds so results are reproducible
    torch.manual_seed(42)
    np.random.seed(42)

    model_nn  = Net(X_tr_s.shape[1])
    optimizer = optim.Adam(model_nn.parameters(), lr=1e-3)
    criterion = nn.BCELoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)  # halve lr every 50 epochs

    for epoch in range(40):
        model_nn.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model_nn(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # check validation accuracy every 5 epochs
        model_nn.eval()
        with torch.no_grad():
            val_acc = ((model_nn(X_test_t) > 0.5).float() == y_test_t).float().mean().item()

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d} -- val accuracy: {val_acc:.2%}")

    # get final predictions
    model_nn.eval()
    with torch.no_grad():
        nn_probs = model_nn(X_test_t).numpy()
        nn_preds = (nn_probs > 0.5).astype(int)

    print(f"\nNeural Network accuracy: {accuracy_score(y_test, nn_preds):.2%}")
    return model_nn, scaler, X_test_t, nn_probs


# ========
# functions.py only defines functions -- all training is run from MarchMadness.ipynb
# =======================