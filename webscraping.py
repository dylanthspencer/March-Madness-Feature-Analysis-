# -*- coding: utf-8 -*-
"""
PART 1: KenPom and Barttorvik Webscraping

Scrapes regular-season stats up to Selection Sunday for each year,
fuzzy-merges the two sources by team name, and writes RegSeasonStats.csv.

NOTE: This scraper is not always reliable — KenPom and Barttorvik periodically
block automated requests. The output CSV (RegSeasonStats.csv) was saved during
a window when access was available and is used directly by the rest of the pipeline.

Requires: pip install requests beautifulsoup4 pandas rapidfuzz

"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
from rapidfuzz import process, fuzz
import time

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

#spoof a browser user agent to reduce the chance of being blocked
headers = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/58.0.3029.110 Safari/537.3'
    )
}

#maps each season year to its selection sunday date (YYYYMMDD format)
#kenpom and barttorvik are searched with this date as the cutoff so that
#only regular-season stats are captured — no tournament games are included
#2020 is omitted because the tournament was cancelled due to covid
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

# ---------------------------------------------------------------------------
# KenPom helper functions
# ---------------------------------------------------------------------------

def extract_td_by_index(row, index):
    """
    pulls the text content of a specific <td> cell from a BeautifulSoup
    table row by its column index.

    parameters
    row: bs4.element.Tag, a <tr> element from the parsed kenpom HTML table.
    index : int, the zero-based column index of the cell to extract.

    returns
    str or None
        The stripped text of the cell or none if the index is out of range
        or the cell is empty.
    """
    tds = row.find_all('td')
    if index >= len(tds):
        return None
    value = tds[index].text.strip()
    return value if value else None


def extract_team_and_seed(team_str):
    """
    splits a kenpom team string into the team name and tournament seed.

    kenpom appends the seed as a trailing integer when a team has been
    selected for the tournament, like 'Duke 1' or 'Gonzaga 11'. This
    function separates the two so the seed can be stored in its own column.

    parameterrs
    team_str: str or None, raw team string from the KenPom table, possibly including a seed.

    returns a tuple (str or None, str or None), (team_name, seed) where seed is None if no seed was found.
    """
    if team_str is None:
        return None, None
    parts = team_str.strip().rsplit(' ', 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0], parts[1]
    return team_str, None


def extract_kenpom(year, cutoff_date):
    """
    scrapes the kenpom team ratings table for a single season, snapshotted
    on the given cutoff date (selection sunday) so only regular-season
    performance is captured.

    columns scraped: rank, team, conference, win/loss, net rating,
    offensive/defensive ratings and ranks, adjusted tempo, luck,
    and strength-of-schedule metrics.

    parametrs
    year: int, the season year (e.g. 2024 for the 2023-24 season)
    cutoff_date: str, selection sunday in YYYYMMDD format, used as the kenpom date parameter.

    returns pd.DataFrame
        one row per team with all scraped kenpom columns plus a 'year' column.
    """
    #maps column names to their zero-based <td> index in the table
    COLUMN_MAP = {
        'rank': 0, 'team': 1, 'conf': 2, 'w_l': 3, 'net_rtg': 4,
        'ORtg': 5, 'ORtg_rank': 6, 'DRtg': 7, 'DRtg_rank': 8,
        'Adjusted_Tempo': 9, 'tempo_rank': 10, 'luck': 11, 'luck_rank': 12,
        'NetRTg_sos': 13, 'ORtg_sos': 14, 'Drtg_sos': 15, 'ncsos_rank': 16,
    }

    # the d parameter locks the snapshot to Selection Sunday
    url = f'https://kenpom.com/index.php?y={year}&d={cutoff_date}'
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    records = []
    for row in soup.find_all('tr'):
        tds = row.find_all('td')
        #skip header rows and rows with too few cells
        if len(tds) < 5:
            continue
        record = {col: extract_td_by_index(row, idx) for col, idx in COLUMN_MAP.items()}
        #only keep actual team rows (rank column contains a digit)
        if record['rank'] and record['rank'].isdigit():
            record['team'], record['seed'] = extract_team_and_seed(record.get('team'))
            records.append(record)

    df = pd.DataFrame(records)
    df['year'] = year
    return df


# ---------------------------------------------------------------------------
# Barttorvik
# ---------------------------------------------------------------------------

def extract_barttorvik(year, cutoff_date):
    """
    fetches team stats from the Barttorvik JSON API for a single season,
    using the cutoff date to exclude any tournament games from the averages.

    The API returns a list of lists; each inner list is a team row with
    stats at fixed positional indices. only the columns relevant to our
    analysis are extracted (shooting efficiency, turnover rates, rebounding,
    tempo, and wins-above-bubble).

    parmeters
    year: int, the season year ( 2024 for the 2023-24 season).
    cutoff_date: str, selection Sunday in YYYYMMDD format, used as the 'end' parameter so
        that stats only reflect regular-season play.

    returns
    pd.DataFrame
        one row per team with the selected Barttorvik stat columns plus 'year'.
    """
    # begin is set to November 1 of the prior year to get the full reg season
    url = (
        f'https://barttorvik.com/teamslicejson.php'
        f'?year={year}&json=1&type=All'
        f'&begin={year-1}1101&end={cutoff_date}'
    )
    r = requests.get(url, headers=headers)
    data = r.json()  # returns a list of lists, one per team

    records = []
    for row in data:
        # positional indexes are fixed by the Barttorvik API response format
        records.append({
            'team':      row[0],   # Team name
            'adjoe':     row[1],   # Adjusted offensive efficiency
            'adjde':     row[2],   # Adjusted defensive efficiency
            'barthag':   row[3],   # Power rating (win probability vs average team)
            'record':    row[4],   # Win-loss record string
            'wins':      row[5],
            'losses':    row[6],
            'efg':       row[7],   # Effective field goal % (offense)
            'efgd':      row[8],   # Effective field goal % allowed (defense)
            'ftr':       row[9],   # Free throw rate (offense)
            'ftrd':      row[10],  # Free throw rate allowed (defense)
            'tor':       row[11],  # Turnover rate (offense)
            'tord':      row[12],  # Turnover rate forced (defense)
            'orb':       row[13],  # Offensive rebound rate
            'drb':       row[14],  # Defensive rebound rate
            '2p':        row[16],  # 2-point shooting % (offense)
            '2pd':       row[17],  # 2-point shooting % allowed (defense)
            '3p':        row[18],  # 3-point shooting % (offense)
            '3pd':       row[19],  # 3-point shooting % allowed (defense)
            '3pr':       row[24],  # 3-point attempt rate (offense)
            '3prd':      row[25],  # 3-point attempt rate allowed (defense)
            'adj_tempo': row[26],  # Adjusted pace (possessions per 40 minutes)
            'wab':       row[34],  # Wins above bubble
            'year':      year,
        })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Fuzzy merge
# ---------------------------------------------------------------------------

def merge_year(year, cutoff_date):
    """
    scrapes both kenpom and bartorvik for a single season and merges them
    into one DataFrame on team name using fuzzy string matching.

    fuzzy matching is necessary because the two sites use slightly different
    team name conventions (e.g. 'UConn' vs 'Connecticut'). the
    token_sort_ratio scorer from rapidfuzz handles word order differences
    and common abbreviations well enough for this dataset.

    A 3 second sleep is used after each request to avoid rate limiting.

    parameters
    year: int, the season year.
    cutoff_date : str, selection Sunday in YYYYMMDD format.

    returns
    pd.DataFrame
        merged dataFrame with all kenpom and bartorvik columns for every
        team that year. teams present in kenpom but not matched in bartorvik
        will have NaN for bartorvik columns.
    """
    print(f"  KenPom snapshot @ {cutoff_date} ...")
    kenpom = extract_kenpom(year, cutoff_date)
    time.sleep(3)  

    print(f"  Barttorvik end @ {cutoff_date} ...")
    bart = extract_barttorvik(year, cutoff_date)
    time.sleep(3)  

    #for each kenpom team name, find the closest match in the Barttorvik list
    bart_teams = bart['team'].tolist()
    kenpom['team_matched'] = kenpom['team'].apply(
        lambda name: process.extractOne(name, bart_teams, scorer=fuzz.token_sort_ratio)[0]
        if name is not None else None
    )

    #merge on the fuzzy matched name, then drop the temporary key column
    bart = bart.rename(columns={'team': 'team_matched'})
    merged = kenpom.merge(bart, on='team_matched', how='left', suffixes=('_kenpom', '_bart'))
    merged = merged.drop(columns=['team_matched'])
    return merged


# ---------------------------------------------------------------------------
# Main — scrape all years and save to CSV
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    all_years = []

    for year, cutoff in sorted(SELECTION_SUNDAY.items()):
        print(f"\n=== {year} (cutoff: {cutoff}) ===")
        try:
            df = merge_year(year, cutoff)
            all_years.append(df)
            print(f"  OK — {len(df)} teams")
        except Exception as e:
            # log the failure and continue, one bad year shouldn't stop the rest
            print(f"  ERROR: {e}")

    #concatenate all successfully scraped seasons into one DataFrame
    final = pd.concat(all_years, ignore_index=True)
    final.to_csv('RegSeasonStats.csv', index=False)
    print(f"\nDone! {len(final)} rows → RegSeasonStats.csv")

    #rename year_bart -> season to match the column name expected downstream
    rss = pd.read_csv("RegSeasonStats.csv")
    rss = rss.rename(columns={'year_bart': 'Season'})

    # drop kenpom-only columns that are redundant with Barttorvik equivalents
    # (barttorvik values are preferred as they are more granular)
    kenpom_cols = [
        'w_l', 'net_rtg', 'ORtg', 'ORtg_rank', 'DRtg', 'DRtg_rank',
        'Adjusted_Tempo', 'tempo_rank', 'luck', 'luck_rank',
        'NetRTg_sos', 'ORtg_sos', 'Drtg_sos', 'ncsos_rank'
    ]
    rss = rss.drop(columns=[c for c in kenpom_cols if c in rss.columns])
    rss.to_csv("RegSeasonStats.csv", index=False)

    print(rss.columns.tolist())
    print(rss.iloc[0])
    print("\nPart 1 complete")
