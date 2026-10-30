"""V13 national weekly predictions. No new model tuning; optional AP top-25 rankings CSV."""
import argparse, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import cfb_asu_v13 as v13

def read_zip(z, suffix):
    paths=[p for p in z.namelist() if p.endswith('/'+suffix)]
    if len(paths)!=1: raise ValueError(f'Expected one {suffix}, got {paths}')
    return pd.read_csv(z.open(paths[0]))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--zip',required=True,help='V2 ZIP')
    ap.add_argument('--stats-zip',required=True,help='Muse verified data ZIP')
    ap.add_argument('--rankings',help='Optional CSV columns: rank,team_id (ESPN ID). Rank is AP poll.')
    ap.add_argument('--start',default='2026-10-30')
    ap.add_argument('--end',default='2026-11-02',help='Exclusive UTC date; covers Oct 30-Nov 1')
    ap.add_argument('--output',default='top25_week_predictions.csv')
    args=ap.parse_args()
    with zipfile.ZipFile(args.zip) as z:
        games=v13.load(z,'games.csv')
    with zipfile.ZipFile(args.stats_zip) as z:
        stats=read_zip(z,'historical_team_game_stats.csv')
        schedule=read_zip(z,'remaining_fbs_schedule_2026.csv')
    stats=stats[(stats.fbs_vs_fbs==1)&(stats.score_flag.fillna('')!='completed_0_0')].drop_duplicates(['espn_game_id','espn_team_id'])
    cutoff=pd.Timestamp('2026-10-30T00:00:00Z')
    games=games[~((games.home_score==0)&(games.away_score==0)&(games.is_completed==1))].copy()
    schedule=schedule[(schedule.fbs_vs_fbs==1)&(schedule.status=='pre')].copy()
    schedule['kickoff']=pd.to_datetime(schedule.date,utc=True)
    schedule=schedule[schedule.kickoff>=cutoff].drop_duplicates('espn_game_id')
    schedule=schedule.rename(columns={'home_team_id':'home_team_id','away_team_id':'away_team_id'})
    schedule['season']=2026
    history,future=v13.build(games,schedule,cutoff,stats,retention=.65,mov_strength=.25)
    train=history[(history.season<=2025)&(history.season!=2023)]
    val=history[history.season==2023]
    v13.FEATURE_MODE='full'
    _,p=v13.model('nn',train,val,future)
    out=schedule.merge(pd.DataFrame({'espn_game_id':future.espn_game_id,'p_home':np.clip(p,.01,.99)}),on='espn_game_id',validate='one_to_one')
    out=out[(out.kickoff>=pd.Timestamp(args.start,tz='UTC'))&(out.kickoff<pd.Timestamp(args.end,tz='UTC'))].copy()
    out['predicted_winner']=np.where(out.p_home>=.5,out.home_team,out.away_team)
    out['win_probability']=np.maximum(out.p_home,1-out.p_home)
    out['home_rank']=pd.NA;out['away_rank']=pd.NA
    if args.rankings:
        ranks=pd.read_csv(args.rankings)
        if 'team_id' not in ranks.columns and 'espn_team_id' in ranks.columns: ranks=ranks.rename(columns={'espn_team_id':'team_id'})
        if not {'rank','team_id'}.issubset(ranks.columns):raise ValueError('Rankings CSV requires rank,team_id or espn_team_id')
        if ranks.team_id.duplicated().any():raise ValueError('Duplicate team_id')
        if not ranks['rank'].between(1,25).all():raise ValueError('Ranks must be 1..25')
        r=dict(zip(ranks.team_id.astype(int),ranks['rank'].astype(int)))
        out['home_rank']=out.home_team_id.map(r)
        out['away_rank']=out.away_team_id.map(r)
        out=out[out.home_rank.notna()|out.away_rank.notna()].copy()
        out['sort_rank']=out[['home_rank','away_rank']].min(axis=1)
        out=out.sort_values(['sort_rank','kickoff'])
    else:
        out=out.sort_values(['kickoff','home_team'])
    cols=['date','home_team','away_team','home_rank','away_rank','predicted_winner','win_probability','p_home','espn_game_id','source_url']
    out[cols].to_csv(args.output,index=False)
    print(f'Wrote {len(out)} games to {args.output}')
    if not args.rankings:print('NOTE: No AP rankings supplied; this is ALL FBS-vs-FBS games in the date window, not a Top-25-only list.')
    print(out[['home_team','away_team','predicted_winner','win_probability']].to_string(index=False,formatters={'win_probability':'{:.1%}'.format}))

if __name__=='__main__':main()
