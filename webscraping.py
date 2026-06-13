# -*- coding: utf-8 -*-
"""
PART 1: KenPom and Barttorvik Webscraping
Scrapes regular-season stats up to Selection Sunday for each year,
fuzzy-merges the two sources, and writes RegSeasonStats.csv.

Run: python webscraping.py
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

headers = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/58.0.3029.110 Safari/537.3'
    )
}

# Cutoff = Selection Sunday for each year (no tournament games included)
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
# KenPom
# ---------------------------------------------------------------------------

def extract_td_by_index(row, index):
    tds = row.find_all('td')
    if index >= len(tds):
        return None
    value = tds[index].text.strip()
    return value if value else None


def extract_team_and_seed(team_str):
    if team_str is None:
        return None, None
    parts = team_str.strip().rsplit(' ', 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0], parts[1]
    return team_str, None


def extract_kenpom(year, cutoff_date):
    COLUMN_MAP = {
        'rank': 0, 'team': 1, 'conf': 2, 'w_l': 3, 'net_rtg': 4,
        'ORtg': 5, 'ORtg_rank': 6, 'DRtg': 7, 'DRtg_rank': 8,
        'Adjusted_Tempo': 9, 'tempo_rank': 10, 'luck': 11, 'luck_rank': 12,
        'NetRTg_sos': 13, 'ORtg_sos': 14, 'Drtg_sos': 15, 'ncsos_rank': 16,
    }
    # Snapshot on Selection Sunday — stats reflect reg season only
    url = f'https://kenpom.com/index.php?y={year}&d={cutoff_date}'
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')

    records = []
    for row in soup.find_all('tr'):
        tds = row.find_all('td')
        if len(tds) < 5:
            continue
        record = {col: extract_td_by_index(row, idx) for col, idx in COLUMN_MAP.items()}
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
    # end=<Selection Sunday> — excludes tournament games
    url = (
        f'https://barttorvik.com/teamslicejson.php'
        f'?year={year}&json=1&type=All'
        f'&begin={year-1}1101&end={cutoff_date}'
    )
    r = requests.get(url, headers=headers)
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

# ---------------------------------------------------------------------------
# Fuzzy merge
# ---------------------------------------------------------------------------

def merge_year(year, cutoff_date):
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

# ---------------------------------------------------------------------------
# Main
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
            print(f"  ERROR: {e}")

    final = pd.concat(all_years, ignore_index=True)
    final.to_csv('RegSeasonStats.csv', index=False)
    print(f"\nDone! {len(final)} rows → RegSeasonStats.csv")

    # Rename year_bart → Season and drop redundant KenPom columns
    rss = pd.read_csv("RegSeasonStats.csv")
    rss = rss.rename(columns={'year_bart': 'Season'})

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
