# -*- coding: utf-8 -*-
"""
marchmadness.py

Module containing all functions and classes used by the March Madness
prediction project. This module is imported from the project notebook
(MarchMadness.ipynb) so that the notebook itself contains no function or
class definitions, per project requirements.

Sections:
    1. KenPom / Barttorvik scraping and fuzzy merging
    2. Tournament results processing (Kaggle data)
    3. Mega / model dataset construction
    4. Baseline seed models
    5. ML model training helpers (Logistic Regression, Random Forest,
       Gradient Boosting, Neural Network)
    6. Visualization / interpretation helpers
"""

import re
import time

import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from rapidfuzz import process, fuzz

import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    auc,
)


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

# HTTP header used for all scraping requests so that sites do not reject
# the request as coming from a non-browser client.
HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
        '(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    )
}

# Selection Sunday date for each season. Used as the cutoff date for both
# KenPom and Barttorvik scrapes so that only regular-season data (no
# tournament games) is included, preventing data leakage.
SELECTION_SUNDAY = {
    2026: '20260315',
    2025: '20250316',
    2024: '20240317',
    2023: '20230312',
    2022: '20220313',
    2021: '20210314',
    # 2020 skipped — no tournament
    2019: '20190317',
    2018: '20180311',
    2017: '20170312',
    2016: '20160313',
    2015: '20150315',
    2014: '20140316',
    2013: '20130317',
    2012: '20120311',
    2011: '20110313',
    2010: '20100314',
}

# Column index map used to pull specific stats out of each KenPom table row.
KENPOM_COLUMN_MAP = {
    'rank': 0, 'team': 1, 'conf': 2, 'w_l': 3, 'net_rtg': 4,
    'ORtg': 5, 'ORtg_rank': 6, 'DRtg': 7, 'DRtg_rank': 8,
    'Adjusted_Tempo': 9, 'tempo_rank': 10, 'luck': 11, 'luck_rank': 12,
    'NetRTg_sos': 13, 'ORtg_sos': 14, 'Drtg_sos': 15, 'ncsos_rank': 16,
}

# Columns from the merged KenPom snapshot that are dropped from the final
# regular-season stats CSV (kept only Barttorvik-derived columns plus seed).
KENPOM_DROP_COLS = [
    'w_l', 'net_rtg', 'ORtg', 'ORtg_rank', 'DRtg', 'DRtg_rank',
    'Adjusted_Tempo', 'tempo_rank', 'luck', 'luck_rank', 'NetRTg_sos',
    'ORtg_sos', 'Drtg_sos', 'ncsos_rank',
]

# Manual alias map used to reconcile team-name spelling differences between
# the KenPom/Barttorvik stats and the Kaggle tournament results.
TEAM_ALIASES = {
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

# Display colors used consistently across all plots for each model.
MODEL_COLORS = {
    "Baseline (Empirical) Matchup": "darkgray",
    "Logistic Regression": "orange",
    "Random Forest": "green",
    "Gradient Boosting": "blue",
    "Neural Network": "red",
}


# --------------------------------------------------------------------------
# Section 1: KenPom / Barttorvik scraping and fuzzy merging
# --------------------------------------------------------------------------

def extract_td_by_index(row, index):
    """
    Return the stripped text of the <td> element at `index` within a
    BeautifulSoup table row.

    Args:
        row: A BeautifulSoup <tr> element.
        index (int): Zero-based index of the <td> to extract.

    Returns:
        str or None: The cell's text, or None if the cell is missing/empty.
    """
    tds = row.find_all('td')
    if index >= len(tds):
        return None
    value = tds[index].text.strip()
    return value if value else None


def extract_team_and_seed(team_str):
    """
    Split a KenPom team string of the form "Team Name 12" into the team
    name and its tournament seed.

    Args:
        team_str (str or None): Raw team cell text, e.g. "Duke 1".

    Returns:
        tuple: (team_name, seed) where seed is a string digit or None if
        the team did not make the tournament (no seed listed).
    """
    if team_str is None:
        return None, None
    parts = team_str.strip().rsplit(' ', 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0], parts[1]
    return team_str, None


def extract_kenpom(year, cutoff_date):
    """
    Scrape KenPom's ratings table for a given season as of a given date.

    Args:
        year (int): Season year (e.g. 2024).
        cutoff_date (str): Date string 'YYYYMMDD' (typically Selection
            Sunday) used so that only regular-season ratings are returned.

    Returns:
        pd.DataFrame: One row per team with KenPom rating columns, team
        name, tournament seed (if any), and the season year.
    """
    url = f'https://kenpom.com/index.php?y={year}&d={cutoff_date}'
    response = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(response.text, 'html.parser')

    records = []
    for row in soup.find_all('tr'):
        tds = row.find_all('td')
        if len(tds) < 5:
            continue
        record = {col: extract_td_by_index(row, idx) for col, idx in KENPOM_COLUMN_MAP.items()}
        if record['rank'] and record['rank'].isdigit():
            record['team'], record['seed'] = extract_team_and_seed(record.get('team'))
            records.append(record)

    df = pd.DataFrame(records)
    df['year'] = year
    return df


def extract_barttorvik(year, cutoff_date):
    """
    Pull Barttorvik's advanced team stats JSON endpoint for a given season,
    restricted to games played before `cutoff_date`.

    Args:
        year (int): Season year (e.g. 2024).
        cutoff_date (str): Date string 'YYYYMMDD' (typically Selection
            Sunday) used as the end of the date range so tournament games
            are excluded.

    Returns:
        pd.DataFrame: One row per team with Barttorvik advanced metrics
        and the season year.
    """
    url = (
        f'https://barttorvik.com/teamslicejson.php'
        f'?year={year}&json=1&type=All'
        f'&begin={year-1}1101&end={cutoff_date}'
    )
    r = requests.get(url, headers=HEADERS)
    data = r.json()

    records = []
    for row in data:
        records.append({
            'team':      row[0],
            'adjoe':     row[1],
            'adjde':     row[2],
            'barthag':   row[3],
            'record':    row[4],
            'wins':      row[5],
            'losses':    row[6],
            'efg':       row[7],
            'efgd':      row[8],
            'ftr':       row[9],
            'ftrd':      row[10],
            'tor':       row[11],
            'tord':      row[12],
            'orb':       row[13],
            'drb':       row[14],
            '2p':        row[16],
            '2pd':       row[17],
            '3p':        row[18],
            '3pd':       row[19],
            '3pr':       row[24],
            '3prd':      row[25],
            'adj_tempo': row[26],
            'wab':       row[34],
            'year':      year,
        })
    return pd.DataFrame(records)


def merge_year(year, cutoff_date):
    """
    Scrape and fuzzy-merge KenPom and Barttorvik data for a single season.

    KenPom team names and Barttorvik team names are spelled differently
    (e.g. "UNC" vs "North Carolina"), so each KenPom team name is matched
    to its closest Barttorvik counterpart using token-sort fuzzy matching.

    Args:
        year (int): Season year.
        cutoff_date (str): Selection Sunday date 'YYYYMMDD' for this season.

    Returns:
        pd.DataFrame: Merged KenPom + Barttorvik stats for every team,
        with suffixes '_kenpom' / '_bart' applied to overlapping columns.
    """
    print(f"  KenPom snapshot @ {cutoff_date} ...")
    kenpom = extract_kenpom(year, cutoff_date)
    time.sleep(3)

    print(f"  Barttorvik end @ {cutoff_date} ...")
    bart = extract_barttorvik(year, cutoff_date)
    time.sleep(3)

    bart_teams = bart['team'].tolist()
    kenpom['team_matched'] = kenpom['team'].apply(
        lambda name: process.extractOne(name, bart_teams, scorer=fuzz.token_sort_ratio)[0]
        if name is not None else None
    )
    bart = bart.rename(columns={'team': 'team_matched'})
    merged = kenpom.merge(bart, on='team_matched', how='left', suffixes=('_kenpom', '_bart'))
    merged = merged.drop(columns=['team_matched'])
    return merged


def build_regular_season_stats(selection_sunday=SELECTION_SUNDAY,
                                 kenpom_drop_cols=KENPOM_DROP_COLS,
                                 out_path='RegSeasonStats.csv'):
    """
    Scrape and merge KenPom + Barttorvik data for every season in
    `selection_sunday`, then write the combined regular-season stats to
    a CSV file.

    This is the top-level driver for Part 1 of the pipeline. Note that
    this function makes many web requests (with short sleeps between
    them) and can take several minutes to run.

    Args:
        selection_sunday (dict): Mapping of season year -> Selection
            Sunday date string 'YYYYMMDD'.
        kenpom_drop_cols (list): KenPom-only columns to drop from the
            final output (kept lean for downstream modeling).
        out_path (str): Path to write the resulting CSV file to.

    Returns:
        pd.DataFrame: The final regular-season stats table (same data
        that was written to `out_path`).
    """
    all_years = []
    for year, cutoff in sorted(selection_sunday.items()):
        print(f"\n=== {year} (cutoff: {cutoff}) ===")
        try:
            df = merge_year(year, cutoff)
            all_years.append(df)
            print(f"  OK — {len(df)} teams")
        except Exception as e:
            print(f"  ERROR: {e}")

    final = pd.concat(all_years, ignore_index=True)
    final.to_csv(out_path, index=False)
    print(f"\nDone! {len(final)} rows → {out_path}")

    # Rename the Barttorvik year column to "Season" so it matches the
    # Kaggle tournament results' season column name.
    final = final.rename(columns={'year_bart': 'Season'})

    # Drop the raw KenPom rating columns; only Barttorvik metrics + seed
    # are used for modeling.
    final = final.drop(columns=[c for c in kenpom_drop_cols if c in final.columns])

    final.to_csv(out_path, index=False)
    return final


# --------------------------------------------------------------------------
# Section 2: Tournament results processing (Kaggle data)
# --------------------------------------------------------------------------

def build_tournament_results(games_path='MMHistoricResults.csv',
                               teams_path='MTeams.csv',
                               out_path='MMHistoricResultsNEW.csv',
                               season_start=2010,
                               season_end=2026):
    """
    Load the Kaggle historical tournament results and team-ID lookup
    table, attach winner/loser team names to each game, and write the
    enriched results to a new CSV.

    Args:
        games_path (str): Path to the Kaggle MMHistoricResults.csv file.
        teams_path (str): Path to the Kaggle MTeams.csv file.
        out_path (str): Path to write the enriched results CSV to.
        season_start (int): Earliest season (inclusive) to keep.
        season_end (int): Latest season (inclusive) to keep.

    Returns:
        pd.DataFrame: Tournament games with 'WTeamName' and 'LTeamName'
        columns added.
    """
    games = pd.read_csv(games_path)
    teams = pd.read_csv(teams_path)

    # Restrict to the seasons we have stats for.
    games = games[(games["Season"] >= season_start) & (games["Season"] <= season_end)]

    # Attach the winning team's name.
    games = games.merge(
        teams[["TeamID", "TeamName"]], how="left", left_on="WTeamID", right_on="TeamID"
    )
    games = games.rename(columns={"TeamName": "WTeamName"})
    games = games.drop(columns=["TeamID"])

    # Attach the losing team's name.
    games = games.merge(
        teams[["TeamID", "TeamName"]], how="left", left_on="LTeamID", right_on="TeamID"
    )
    games = games.rename(columns={"TeamName": "LTeamName"})
    games = games.drop(columns=["TeamID"])

    games.to_csv(out_path, index=False)
    return games


# --------------------------------------------------------------------------
# Section 3: Mega / model dataset construction
# --------------------------------------------------------------------------

def strip_seed_from_team(s):
    """
    Remove a trailing seed marker (e.g. "Duke 1*") from a KenPom team name.

    Args:
        s: Raw team name, possibly with a trailing "<digits>*" seed token.

    Returns:
        str: The team name with the seed token removed, if present.
    """
    s = str(s).strip()
    parts = s.split()

    last_part = parts[-1]

    if last_part.replace("*", "").isdigit() and "*" in last_part:
        parts = parts[:-1]

    return " ".join(parts)


def base_norm(s):
    """
    Normalize a team name to a simple lowercase, punctuation-free form so
    that the same team is represented identically across data sources.

    Args:
        s: Raw team name.

    Returns:
        str: Normalized team name (lowercase, no periods/apostrophes,
        hyphens replaced with spaces, collapsed whitespace).
    """
    s = strip_seed_from_team(s).lower().strip()
    s = s.replace("&", "and")
    s = re.sub(r"[\.\'’]", "", s)
    s = s.replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def team_key(s, alias=TEAM_ALIASES):
    """
    Produce a canonical "team key" for joining records across data
    sources: normalize the name, then apply the manual alias map for
    known spelling/abbreviation discrepancies.

    Args:
        s: Raw team name.
        alias (dict): Mapping of normalized name -> canonical name.

    Returns:
        str: Canonical team key.
    """
    k = base_norm(s)
    return alias.get(k, k)


def build_datasets(games_path='MMHistoricResultsNEW.csv',
                    stats_path='RegSeasonStats.csv',
                    mega_out='MarchMadness_MegaDataset_2010_2025.csv',
                    model_out='MarchMadness_ModelDataset_2010_2025.csv'):
    """
    Merge tournament game results with each team's regular-season stats
    to build two datasets:

    1. The "mega" dataset: one row per tournament game, with the winning
       team's stats prefixed 'W_' and the losing team's stats prefixed
       'L_', plus per-stat winner-minus-loser difference columns.
    2. The "model" dataset: the same games re-labeled as Team1/Team2 (with
       Team1 assigned by lower TeamID, independent of who won), with
       Team1-minus-Team2 difference columns and a 'Team1Win' target.

    Args:
        games_path (str): Path to the enriched tournament results CSV
            (output of `build_tournament_results`).
        stats_path (str): Path to the regular-season stats CSV (output of
            `build_regular_season_stats`).
        mega_out (str): Output path for the mega dataset CSV.
        model_out (str): Output path for the model dataset CSV.

    Returns:
        tuple: (mega_df, model_df) — the two resulting DataFrames.
    """
    games = pd.read_csv(games_path)
    stats = pd.read_csv(stats_path)

    # Keep only seasons that actually have tournament game results.
    stats = stats[stats["Season"].isin(games["Season"].unique())].copy()

    # Build canonical team keys for joining.
    stats["team_clean"] = stats["team"].apply(strip_seed_from_team)
    stats["team_key"] = stats["team_clean"].apply(team_key)
    games["W_team_key"] = games["WTeamName"].apply(team_key)
    games["L_team_key"] = games["LTeamName"].apply(team_key)

    # ---- Mega dataset: winner/loser stats side by side ----
    Wstats = stats.add_prefix("W_")
    Lstats = stats.add_prefix("L_")
    mega = games.merge(
        Wstats,
        left_on=["Season", "W_team_key"],
        right_on=["W_Season", "W_team_key"],
        how="left"
    )
    mega = mega.merge(
        Lstats,
        left_on=["Season", "L_team_key"],
        right_on=["L_Season", "L_team_key"],
        how="left"
    )

    # Columns that are identifiers/metadata rather than numeric features.
    non_feature = {"Season", "year", "team_key", "team_clean", "team", "conf", "w_l", "record"}
    base_cols = [c for c in stats.columns if c not in non_feature]

    winner_data, loser_data = {}, {}
    for column in base_cols:
        winner_column = "W_" + column
        if winner_column in mega.columns:
            winner_data[column] = pd.to_numeric(mega[winner_column], errors="coerce")

    for column in base_cols:
        loser_column = "L_" + column
        if loser_column in mega.columns:
            loser_data[column] = pd.to_numeric(mega[loser_column], errors="coerce")

    # Winner-minus-loser difference columns (key features for modeling).
    for column in base_cols:
        if column in winner_data and column in loser_data:
            if winner_data[column].notna().any() and loser_data[column].notna().any():
                mega[column + "_diff"] = winner_data[column] - loser_data[column]

    mega["Winner"] = mega["WTeamName"]
    mega["Loser"] = mega["LTeamName"]
    mega["winner_first"] = 1

    front = ["Season", "DayNum", "WTeamID", "WTeamName", "WScore", "LTeamID",
             "LTeamName", "LScore", "Winner", "Loser", "WLoc", "NumOT"]
    front = [c for c in front if c in mega.columns]
    mega = mega[front + [c for c in mega.columns if c not in front]]
    mega.to_csv(mega_out, index=False)

    # ---- Model dataset: Team1/Team2 framing (order independent of result) ----
    model_games = games.copy()
    team1_is_winner = model_games["WTeamID"] < model_games["LTeamID"]

    model_games["Team1Name"] = np.where(team1_is_winner, model_games["WTeamName"], model_games["LTeamName"])
    model_games["Team2Name"] = np.where(team1_is_winner, model_games["LTeamName"], model_games["WTeamName"])
    model_games["Team1ID"] = np.where(team1_is_winner, model_games["WTeamID"], model_games["LTeamID"])
    model_games["Team2ID"] = np.where(team1_is_winner, model_games["LTeamID"], model_games["WTeamID"])
    model_games["Team1Score"] = np.where(team1_is_winner, model_games["WScore"], model_games["LScore"])
    model_games["Team2Score"] = np.where(team1_is_winner, model_games["LScore"], model_games["WScore"])
    model_games["Team1_key"] = model_games["Team1Name"].apply(team_key)
    model_games["Team2_key"] = model_games["Team2Name"].apply(team_key)
    model_games["Team1Win"] = np.where(team1_is_winner, 1, 0)

    T1 = stats.add_prefix("T1_")
    T2 = stats.add_prefix("T2_")
    model = model_games.merge(T1, left_on=["Season", "Team1_key"], right_on=["T1_Season", "T1_team_key"], how="left")
    model = model.merge(T2, left_on=["Season", "Team2_key"], right_on=["T2_Season", "T2_team_key"], how="left")

    for c in base_cols:
        t1, t2 = "T1_" + c, "T2_" + c
        if t1 in model.columns and t2 in model.columns:
            n1 = pd.to_numeric(model[t1], errors="coerce")
            n2 = pd.to_numeric(model[t2], errors="coerce")
            if n1.notna().any() and n2.notna().any():
                model[c + "_diff"] = n1 - n2

    front2 = ["Season", "DayNum", "Team1ID", "Team1Name", "Team1Score", "Team2ID",
              "Team2Name", "Team2Score", "Team1Win", "WTeamName", "LTeamName", "WScore", "LScore"]
    model = model[front2 + [c for c in model.columns if c not in front2]]
    model.to_csv(model_out, index=False)

    print("Mega dataset saved:", mega_out, mega.shape)
    print("Model dataset saved:", model_out, model.shape)

    return mega, model


# --------------------------------------------------------------------------
# Section 4: Baseline seed models
# --------------------------------------------------------------------------

def prepare_seed_data(model_out='MarchMadness_ModelDataset_2010_2025.csv'):
    """
    Load the model dataset and compute seed-derived columns used by the
    baseline models.

    Args:
        model_out (str): Path to the model dataset CSV.

    Returns:
        pd.DataFrame: The model dataset with added 'seed_diff', 'lo_seed',
        'hi_seed', and 'lo_won' columns, restricted to rows with valid
        seed and outcome data.
    """
    df = pd.read_csv(model_out)

    data = df.copy()
    data["T1_seed"] = pd.to_numeric(data["T1_seed"], errors="coerce")
    data["T2_seed"] = pd.to_numeric(data["T2_seed"], errors="coerce")
    data = data.dropna(subset=["T1_seed", "T2_seed", "Team1Win"])
    data["T1_seed"] = data["T1_seed"].astype(int)
    data["T2_seed"] = data["T2_seed"].astype(int)
    data["seed_diff"] = data["T1_seed"] - data["T2_seed"]
    data["lo_seed"] = data[["T1_seed", "T2_seed"]].min(axis=1)
    data["hi_seed"] = data[["T1_seed", "T2_seed"]].max(axis=1)
    data["lo_won"] = np.where(
        data["T1_seed"] < data["T2_seed"],
        data["Team1Win"],
        1 - data["Team1Win"]
    )
    return data


def run_seed_baselines(data, test_size=0.2, random_state=42, min_games=20):
    """
    Train and evaluate the three seed-based baseline models:

    1. A logistic regression on seed_diff alone.
    2. A deterministic "lower seed always wins" rule.
    3. An empirical historical win-rate lookup by seed matchup.

    Args:
        data (pd.DataFrame): Output of `prepare_seed_data`.
        test_size (float): Fraction of data held out for testing.
        random_state (int): Random seed for the train/test split.
        min_games (int): Minimum number of historical games required for
            a seed matchup before falling back to "better seed wins".

    Returns:
        dict: Accuracy scores for each baseline, keyed by name:
        'seed_logreg', 'lower_seed_wins', 'empirical_matchup'. Also
        returns the train/test split via 'train' and 'test' keys.
    """
    train, test = train_test_split(
        data, test_size=test_size, random_state=random_state, stratify=data["Team1Win"]
    )

    # 1. Seed-only logistic regression.
    seed_model = LogisticRegression(max_iter=1000)
    seed_model.fit(train[["seed_diff"]], train["Team1Win"])
    seed_preds = seed_model.predict(test[["seed_diff"]])
    seed_logreg_acc = accuracy_score(test["Team1Win"], seed_preds)

    # 2. Deterministic baseline: lower seed always wins.
    test_no_ties = test[test["seed_diff"] != 0].copy()
    det_preds = (test_no_ties["seed_diff"] < 0).astype(int)
    det_acc = accuracy_score(test_no_ties["Team1Win"], det_preds)

    # 3. Empirical matchup rates (computed on train only).
    rates = (
        train.groupby(["lo_seed", "hi_seed"])
        .agg(games=("lo_won", "count"), lo_wins=("lo_won", "sum"))
        .reset_index()
    )
    rates["win_rate"] = rates["lo_wins"] / rates["games"]

    def predict_matchup(row):
        team1_seed, team2_seed = row["T1_seed"], row["T2_seed"]
        lower_seed, higher_seed = min(team1_seed, team2_seed), max(team1_seed, team2_seed)

        matchup = rates[(rates["lo_seed"] == lower_seed) & (rates["hi_seed"] == higher_seed)]
        if matchup.empty or matchup["games"].values[0] < min_games:
            team1_probability = 1 if team1_seed < team2_seed else 0
        else:
            lower_seed_win_rate = matchup["win_rate"].values[0]
            team1_probability = lower_seed_win_rate if team1_seed < team2_seed else 1 - lower_seed_win_rate

        return int(team1_probability > 0.5)

    test_no_ties = test[test["seed_diff"] != 0].copy()
    test_no_ties["matchup_pred"] = test_no_ties.apply(predict_matchup, axis=1)
    matchup_acc = accuracy_score(test_no_ties["Team1Win"], test_no_ties["matchup_pred"])

    return {
        "seed_logreg": seed_logreg_acc,
        "lower_seed_wins": det_acc,
        "empirical_matchup": matchup_acc,
        "train": train,
        "test": test,
        "rates": rates,
    }


# --------------------------------------------------------------------------
# Section 5: ML model training helpers
# --------------------------------------------------------------------------

def get_feature_columns(df):
    """
    Identify the feature columns used for modeling: all '_diff' columns
    except the seed difference (seed is reserved for the baseline models).

    Args:
        df (pd.DataFrame): The model dataset.

    Returns:
        list: Column names of difference features, excluding seed.
    """
    diff_cols = [col for col in df.columns if col.endswith("_diff")]
    return [col for col in diff_cols if "seed" not in col.lower()]


def split_features(df, feature_cols, test_size=0.2, random_state=42):
    """
    Build the feature matrix X and target vector y from the model dataset
    and split into train/test sets.

    Args:
        df (pd.DataFrame): The model dataset.
        feature_cols (list): Feature column names (e.g. from
            `get_feature_columns`).
        test_size (float): Fraction of data held out for testing.
        random_state (int): Random seed for the split.

    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X = df[feature_cols].copy()
    y = df["Team1Win"].copy()
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def train_logistic_regression(X_train, y_train, X_test, y_test, max_iter=5000, random_state=42):
    """
    Fit a logistic regression model and report test accuracy plus the
    largest-magnitude coefficients.

    Args:
        X_train, y_train: Training features/target.
        X_test, y_test: Test features/target.
        max_iter (int): Maximum solver iterations.
        random_state (int): Random seed.

    Returns:
        tuple: (fitted_model, accuracy, coefficients_df) where
        coefficients_df has columns 'Feature', 'Coefficient', 'AbsValue',
        sorted by absolute coefficient size (descending).
    """
    model = LogisticRegression(max_iter=max_iter, random_state=random_state).fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    coefficients = pd.DataFrame({
        "Feature": X_train.columns,
        "Coefficient": model.coef_[0],
    })
    coefficients["AbsValue"] = coefficients["Coefficient"].abs()
    coefficients = coefficients.sort_values(by="AbsValue", ascending=False)

    return model, accuracy, coefficients


def train_random_forest(X_train, y_train, X_test, y_test,
                         n_estimators=500, max_depth=5, min_samples_split=10, random_state=42):
    """
    Fit a Random Forest classifier and report train/test accuracy plus
    feature importances.

    Args:
        X_train, y_train: Training features/target.
        X_test, y_test: Test features/target.
        n_estimators (int): Number of trees.
        max_depth (int): Maximum tree depth.
        min_samples_split (int): Minimum samples required to split a node.
        random_state (int): Random seed.

    Returns:
        tuple: (fitted_model, test_accuracy, train_accuracy, importances_df)
        where importances_df has columns 'Feature', 'Importance', sorted
        descending.
    """
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        class_weight="balanced",
    )
    rf_model.fit(X_train, y_train)

    test_accuracy = accuracy_score(y_test, rf_model.predict(X_test))
    train_accuracy = accuracy_score(y_train, rf_model.predict(X_train))

    importances = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": rf_model.feature_importances_,
    }).sort_values(by="Importance", ascending=False)

    return rf_model, test_accuracy, train_accuracy, importances


def train_gradient_boosting(X_train, y_train, X_test, y_test,
                             n_estimators=100, learning_rate=0.05, max_depth=2,
                             min_samples_leaf=5, subsample=0.8, random_state=42):
    """
    Fit a Gradient Boosting classifier and report train/test accuracy,
    AUC, and feature importances.

    Args:
        X_train, y_train: Training features/target.
        X_test, y_test: Test features/target.
        n_estimators, learning_rate, max_depth, min_samples_leaf, subsample:
            GradientBoostingClassifier hyperparameters.
        random_state (int): Random seed.

    Returns:
        tuple: (fitted_model, test_accuracy, train_accuracy, auc_score,
        importances_df) where importances_df has columns 'Feature',
        'Importance', sorted descending.
    """
    gb_model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        subsample=subsample,
        random_state=random_state,
    )
    gb_model.fit(X_train, y_train)

    y_pred = gb_model.predict(X_test)
    y_prob = gb_model.predict_proba(X_test)[:, 1]

    test_accuracy = accuracy_score(y_test, y_pred)
    train_accuracy = accuracy_score(y_train, gb_model.predict(X_train))
    auc_score = roc_auc_score(y_test, y_prob)

    importances = pd.DataFrame({
        "Feature": X_train.columns,
        "Importance": gb_model.feature_importances_,
    }).sort_values(by="Importance", ascending=False)

    return gb_model, test_accuracy, train_accuracy, auc_score, importances


class WinPredictorNet(nn.Module):
    """
    A small feedforward neural network for binary win/loss prediction.

    Architecture: input -> 64 (ReLU, Dropout 0.3) -> 32 (ReLU, Dropout 0.3)
    -> 1 (Sigmoid).
    """

    def __init__(self, input_dim):
        """
        Args:
            input_dim (int): Number of input features.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Run the forward pass.

        Args:
            x (torch.Tensor): Input batch of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Predicted win probabilities, shape (batch_size,).
        """
        return self.net(x).squeeze()


def train_neural_network(X_train, y_train, X_test, y_test,
                          epochs=40, batch_size=32, lr=1e-3,
                          step_size=50, gamma=0.5, seed=42, print_every=5):
    """
    Scale features, train a `WinPredictorNet`, and report test accuracy.

    Args:
        X_train, y_train: Training features/target (DataFrames/Series or
            arrays).
        X_test, y_test: Test features/target.
        epochs (int): Number of training epochs.
        batch_size (int): Mini-batch size.
        lr (float): Adam learning rate.
        step_size (int): LR scheduler step size (in epochs).
        gamma (float): LR scheduler decay factor.
        seed (int): Random seed for reproducibility.
        print_every (int): Print validation accuracy every N epochs.

    Returns:
        tuple: (model, scaler, test_accuracy, test_probs, epoch_accuracies)
        where test_probs is a numpy array of predicted probabilities for
        the test set, and epoch_accuracies is a list of per-epoch
        validation accuracies.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values if hasattr(y_train, "values") else y_train, dtype=torch.float32)
    y_test_t = torch.tensor(y_test.values if hasattr(y_test, "values") else y_test, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = WinPredictorNet(X_train_scaled.shape[1])
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    epoch_accs = []
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_acc = ((model(X_test_t) > 0.5).float() == y_test_t).float().mean().item()
        epoch_accs.append(val_acc)

        if (epoch + 1) % print_every == 0:
            print(f"Epoch {epoch+1:3d} — val accuracy: {val_acc:.2%}")

    model.eval()
    with torch.no_grad():
        test_probs = model(X_test_t).numpy()

    test_preds = (test_probs > 0.5).astype(int)
    test_accuracy = accuracy_score(y_test, test_preds)

    return model, scaler, test_accuracy, test_probs, epoch_accs


# --------------------------------------------------------------------------
# Section 6: Visualization / interpretation helpers
# --------------------------------------------------------------------------

def plot_accuracy_comparison(model_results, baseline_line=0.7609, colors=MODEL_COLORS):
    """
    Plot a bar chart comparing the accuracy of each model, with a
    horizontal reference line for the deterministic seed baseline.

    Args:
        model_results (list[dict]): List of {'Model': name, 'Accuracy':
            value} dicts.
        baseline_line (float): Y-position of the horizontal reference line
            (e.g. the "lower seed wins" accuracy).
        colors (dict): Mapping of model name -> bar color.

    Returns:
        pd.DataFrame: The results sorted by accuracy descending (the data
        used for the plot).
    """
    performance_df = pd.DataFrame(model_results)
    accuracy_plot = performance_df.sort_values("Accuracy", ascending=False)
    plot_colors = [colors.get(m, "black") for m in accuracy_plot["Model"]]

    plt.figure(figsize=(11, 6))
    plt.bar(accuracy_plot["Model"], accuracy_plot["Accuracy"], color=plot_colors, edgecolor="black")
    plt.axhline(y=baseline_line, color="gray", linestyle="--", linewidth=1.24)
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.3)

    for index, accuracy in enumerate(accuracy_plot["Accuracy"]):
        plt.text(index, accuracy + 0.01, f"{accuracy:.2%}", ha="center")

    plt.tight_layout()
    plt.show()

    return accuracy_plot


def plot_permutation_importance(models, X_test, y_test, top_n=8, colors=MODEL_COLORS):
    """
    Compute permutation importance for each fitted model and plot the
    top features as a horizontal bar chart, alongside a summary table.

    Args:
        models (dict): Mapping of model name -> fitted estimator (must
            support `.predict` for permutation_importance).
        X_test, y_test: Test features/target.
        top_n (int): Number of top features (by average importance
            across models) to display.
        colors (dict): Mapping of model name -> bar color.

    Returns:
        pd.DataFrame: Top `top_n + 2` features and their average
        permutation importance, sorted descending.
    """
    importance_tables = []
    for model_name, fitted_model in models.items():
        result = permutation_importance(fitted_model, X_test, y_test, n_repeats=10, random_state=42, scoring="accuracy")
        importance_tables.append(pd.DataFrame({
            "Feature": X_test.columns,
            "Model": model_name,
            "Importance": result.importances_mean,
        }))

    all_importances = pd.concat(importance_tables, ignore_index=True)
    all_importances["Importance"] = all_importances["Importance"].clip(lower=0)

    top_features = (
        all_importances.groupby("Feature")["Importance"]
        .mean().sort_values(ascending=False).head(top_n).index
    )

    importance_pivot = (
        all_importances[all_importances["Feature"].isin(top_features)]
        .pivot_table(index="Feature", columns="Model", values="Importance")
        .fillna(0)
    )
    importance_pivot["Average"] = importance_pivot.mean(axis=1)
    importance_pivot = importance_pivot.sort_values("Average", ascending=True).drop(columns="Average")

    plot_colors = [colors.get(m, "black") for m in importance_pivot.columns]
    importance_pivot.plot(kind="barh", figsize=(11, 7), color=plot_colors, edgecolor="black")
    plt.title("Most Important Predictors Across Models")
    plt.xlabel("Permutation Importance")
    plt.ylabel("Feature")
    plt.grid(axis="x", alpha=0.3)
    plt.legend(title="Model")
    plt.tight_layout()
    plt.show()

    top_feature_table = (
        all_importances.groupby("Feature")["Importance"]
        .mean().sort_values(ascending=False).head(top_n + 2).reset_index()
    )
    return top_feature_table


def plot_confusion_matrices(preds, y_test):
    """
    Plot side-by-side confusion matrices for each model's predictions.

    Args:
        preds (dict): Mapping of model name -> predicted labels array.
        y_test: True test labels.
    """
    fig, axes = plt.subplots(1, len(preds), figsize=(18, 4))

    for ax, (name, pred) in zip(axes, preds.items()):
        cm = confusion_matrix(y_test, pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["T2 Wins", "T1 Wins"],
                    yticklabels=["T2 Wins", "T1 Wins"])
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.suptitle("Confusion Matrices", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_roc_curves(roc_models, y_test, colors=MODEL_COLORS):
    """
    Plot ROC curves for each model on a single set of axes, with AUC
    values in the legend.

    Args:
        roc_models (dict): Mapping of model name -> predicted positive-
            class probabilities array.
        y_test: True test labels.
        colors (dict): Mapping of model name -> line color.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, probs in roc_models.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.3f})", color=colors.get(name, "black"))

    ax.plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_probability_distributions(roc_models, y_test):
    """
    Plot histograms of predicted win probabilities for each model, split
    by actual outcome (Team1 won vs. lost).

    Args:
        roc_models (dict): Mapping of model name -> predicted positive-
            class probabilities array.
        y_test: True test labels.
    """
    fig, axes = plt.subplots(1, len(roc_models), figsize=(18, 4))

    for ax, (name, probs) in zip(axes, roc_models.items()):
        for outcome, label, color in [(1, "Team1 Won", "steelblue"), (0, "Team1 Lost", "tomato")]:
            mask = np.array(y_test) == outcome
            ax.hist(probs[mask], bins=20, alpha=0.6, label=label, color=color)
        ax.set_title(name)
        ax.set_xlabel("Predicted P(Team1 Win)")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    plt.suptitle("Predicted Probability Distributions", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_logistic_coefficients(model, feature_names):
    """
    Plot a horizontal bar chart of logistic regression coefficients,
    colored by sign.

    Args:
        model: Fitted LogisticRegression model.
        feature_names: Iterable of feature column names corresponding to
            `model.coef_[0]`.
    """
    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": model.coef_[0],
    }).sort_values("Coefficient")

    colors_coef = ["tomato" if c < 0 else "steelblue" for c in coef_df["Coefficient"]]

    plt.figure(figsize=(10, 7))
    plt.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors_coef, edgecolor="black")
    plt.axvline(x=0, color="black", linewidth=0.8)
    plt.title("Logistic Regression Coefficients")
    plt.xlabel("Coefficient Value")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()
