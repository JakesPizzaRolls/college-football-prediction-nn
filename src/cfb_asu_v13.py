"""V12: chronological, leakage-resistant Elo + opponent-adjusted form forecast.
Usage: python cfb_asu_v13.py --zip ~/Downloads/asu-big12-cfb-dataset-v2.zip
Only uses scores and dates from the supplied V2 dataset. No player data.
"""
import argparse, zipfile
from collections import defaultdict, deque
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
import torch
from torch import nn

SEED=42
BASE_COLS=['elo','win_pct','margin','last5','adjusted','games','rest']
STAT_COLS=['off_ypp','def_ypp','off_yards','def_yards','off_third','def_third','turnover_diff','off_points','def_points']
COLS=BASE_COLS+STAT_COLS
def load(z,name):
    paths=[n for n in z.namelist() if n.endswith('/data/'+name)]
    if len(paths)!=1: raise ValueError('Missing/ambiguous '+name)
    return pd.read_csv(z.open(paths[0]))
def state():
    return {'elo':1500.,'season':None,'margins':[],'wins':[],'adjusted':[],
            'last':None,'stats':[]}
def reset(s,season,retention):
    if s['season']!=season:
        if s['season'] is not None:
            s['elo']=1500+retention*(s['elo']-1500)
        s.update(season=season,margins=[],wins=[],adjusted=[],last=None,stats=[])
def snapshot(s,date):
    m=s['margins'];a=s['adjusted']
    rest=np.nan if s['last'] is None else min(35.,max(0.,(date-s['last']).total_seconds()/86400))
    z={}
    for k in STAT_COLS:
        values=[v[k] for v in s['stats'][-8:] if np.isfinite(v.get(k,np.nan))]
        prior={'off_ypp':5.5,'def_ypp':5.5,'off_yards':380,'def_yards':380,'off_third':.4,'def_third':.4,'turnover_diff':0,'off_points':27,'def_points':27}[k]
        z[k]=(sum(values)+5*prior)/(len(values)+5)
    return dict(**z,elo=s['elo'],win_pct=np.mean(s['wins']) if m else .5,
                margin=np.mean(m) if m else 0.,last5=np.mean(m[-5:]) if m else 0.,
                adjusted=(len(a)/(len(a)+5))*np.mean(a[-8:]) if a else 0.,games=len(m),rest=rest)
def build(games,remaining,cutoff,stats,retention=.75,mov_strength=0.):
    teams=defaultdict(state); rows=[]; remaining_rows=[]
    stat_index=stats.set_index(['espn_game_id','espn_team_id'],verify_integrity=True)
    # Treat games sharing kickoff timestamp as one batch; never update until all pregame snapshots taken.
    completed=games[(games.is_completed==1)&(games.fbs_vs_fbs==1)&
                    (pd.to_datetime(games.date,utc=True)<cutoff)].copy()
    completed['kickoff']=pd.to_datetime(completed.date,utc=True)
    completed=completed.sort_values(['kickoff','espn_game_id'])
    remaining=remaining.copy();remaining['kickoff']=pd.to_datetime(remaining.date,utc=True)
    for date, group in completed.groupby('kickoff',sort=True):
        batch=[]
        for r in group.itertuples():
            h=int(r.home_team_id);a=int(r.away_team_id);season=int(r.season)
            for t in (h,a):reset(teams[t],season,retention)
            sh=snapshot(teams[h],date);sa=snapshot(teams[a],date)
            record={'espn_game_id':r.espn_game_id,'date':r.date,'season':season,
                    'home_team':r.home_team,'away_team':r.away_team,
                    'y':int(r.home_score>r.away_score),'neutral':int(bool(r.neutral_site))}
            record.update({'home_'+k:v for k,v in sh.items()})
            record.update({'away_'+k:v for k,v in sa.items()})
            rows.append(record)
            batch.append((h,a,float(r.home_score),float(r.away_score),sh,sa,bool(r.neutral_site),r.espn_game_id))
        for h,a,hs,as_,sh,sa,neutral,gid in batch:
            mh=hs-as_; ma=-mh
            eh=sh['elo'];ea=sa['elo']
            advantage=0 if neutral else 55
            p=1/(1+10**(-((eh-ea+advantage)/400)))
            # Margin multiplier = 1 for baseline, grows gently for large margins.
            multiplier=1+mov_strength*(np.log1p(abs(mh))-np.log1p(10))
            multiplier=float(np.clip(multiplier,.5,2.0))
            teams[h]['elo']+=20*multiplier*(float(mh>0)-p)
            teams[a]['elo']+=20*multiplier*(float(ma>0)-(1-p))
            # Opponent's PRE-game season margin, avoiding same-game outcome leakage.
            # Elo-adjusted scoring residual: compare actual margin with a capped expected margin.
            # No current/future game information enters this expectation.
            expected_margin=np.clip((eh-ea+advantage)/28.,-21.,21.)
            teams[h]['adjusted'].append(mh-expected_margin)
            teams[a]['adjusted'].append(ma+expected_margin)
            def getstat(tid):
                try: return stat_index.loc[(gid,tid)]
                except KeyError: return None
            home_stats=getstat(h);away_stats=getstat(a)
            def val(rec,k):
                if rec is None: return np.nan
                v=rec.get(k,np.nan)
                return float(v) if pd.notna(v) else np.nan
            def ratio(rec,num,den):
                n,d=val(rec,num),val(rec,den)
                return n/d if np.isfinite(n) and np.isfinite(d) and d>0 else np.nan
            def add_stats(tid,own,opp,own_pre,opp_pre,pts_for,pts_against):
                if own is None or opp is None or not bool(own.get('stats_available',False)) or not bool(opp.get('stats_available',False)):return
                own_ypp=val(own,'computed_yards_per_play');opp_ypp=val(opp,'computed_yards_per_play')
                own_yards=val(own,'total_yards');opp_yards=val(opp,'total_yards')
                own_third=ratio(own,'third_down_made','third_down_attempted')
                opp_third=ratio(opp,'third_down_made','third_down_attempted')
                # Opponent-adjusted residuals against PRE-game defensive/offensive priors.
                # For example, a 7 YPP offense against a defense allowing 6 YPP gets +1 YPP.
                v={'off_ypp':own_ypp-opp_pre['def_ypp']+5.5,
                   'def_ypp':opp_ypp-opp_pre['off_ypp']+5.5,
                   'off_yards':own_yards-opp_pre['def_yards']+380,
                   'def_yards':opp_yards-opp_pre['off_yards']+380,
                   'off_third':own_third-opp_pre['def_third']+.4,
                   'def_third':opp_third-opp_pre['off_third']+.4,
                   'turnover_diff':val(opp,'turnovers')-val(own,'turnovers'),
                   'off_points':pts_for-opp_pre['def_points']+27,
                   'def_points':pts_against-opp_pre['off_points']+27}
                teams[tid]['stats'].append(v)
            add_stats(h,home_stats,away_stats,sh,sa,hs,as_)
            add_stats(a,away_stats,home_stats,sa,sh,as_,hs)
            for tid,m in ((h,mh),(a,ma)):
                teams[tid]['margins'].append(m)
                teams[tid]['wins'].append(int(m>0))
                teams[tid]['last']=date
    # Final current-season team snapshots: all games through cutoff processed.
    for r in remaining.sort_values('kickoff').itertuples():
        h=int(r.home_team_id);a=int(r.away_team_id)
        for t in (h,a):reset(teams[t],int(r.season),retention)
        sh=snapshot(teams[h],r.kickoff);sa=snapshot(teams[a],r.kickoff)
        d={'espn_game_id':r.espn_game_id,'date':r.date,'home_team':r.home_team,
           'away_team':r.away_team,'neutral':int(bool(r.neutral_site))}
        d.update({'home_'+k:v for k,v in sh.items()})
        d.update({'away_'+k:v for k,v in sa.items()})
        remaining_rows.append(d)
    return pd.DataFrame(rows),pd.DataFrame(remaining_rows)
FEATURE_MODE='full'
def matrix(df):
    keys=['elo'] if FEATURE_MODE=='elo' else (BASE_COLS if FEATURE_MODE=='baseline' else COLS)
    x=np.array([df['home_'+k].to_numpy(dtype=float)-df['away_'+k].to_numpy(dtype=float) for k in keys]+[df.neutral.to_numpy(dtype=float)]).T
    return x
def model(kind,train,val,pred):
    imp=SimpleImputer(strategy='median');scale=StandardScaler()
    x=scale.fit_transform(imp.fit_transform(matrix(train)))
    xv=scale.transform(imp.transform(matrix(val)))
    xp=scale.transform(imp.transform(matrix(pred)))
    y=train.y.to_numpy(dtype=float)
    if kind=='logistic':
        m=LogisticRegression(max_iter=1500,random_state=SEED)
        m.fit(x,y)
        return m.predict_proba(xv)[:,1],m.predict_proba(xp)[:,1]
    torch.manual_seed(SEED)
    m=nn.Sequential(nn.Linear(x.shape[1],64),nn.ReLU(),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
    opt=torch.optim.Adam(m.parameters(),lr=.001,weight_decay=.003)
    xt=torch.tensor(x,dtype=torch.float32);yt=torch.tensor(y[:,None],dtype=torch.float32)
    vx=torch.tensor(xv,dtype=torch.float32);vy=torch.tensor(val.y.to_numpy(dtype=float)[:,None],dtype=torch.float32)
    best=float('inf');wait=0;best_state=None
    for epoch in range(400):
        m.train();opt.zero_grad()
        loss=nn.functional.binary_cross_entropy_with_logits(m(xt),yt)
        loss.backward();opt.step()
        m.eval()
        with torch.no_grad():v=nn.functional.binary_cross_entropy_with_logits(m(vx),vy).item()
        if v<best-1e-5:
            best=v;wait=0;best_state={k:t.clone() for k,t in m.state_dict().items()}
        else:wait+=1
        if wait==35:break
    m.load_state_dict(best_state);m.eval()
    with torch.no_grad():
        return (torch.sigmoid(m(vx)).numpy().ravel(),
                torch.sigmoid(m(torch.tensor(xp,dtype=torch.float32))).numpy().ravel())
def simulate(rem,standings,n):
    rng=np.random.default_rng(SEED)
    names=standings.team.tolist();idx={name:i for i,name in enumerate(names)}
    w=np.broadcast_to(standings.conference_wins.to_numpy(dtype=np.int16),(n,len(names))).copy()
    for r in rem.itertuples():
        h=idx[r.home_team];a=idx[r.away_team]
        result=rng.random(n)<r.p_home
        w[:,h]+=result;w[:,a]+=~result
    ai=idx['Arizona State Sun Devils']
    greater=(w>w[:,ai,None]).sum(axis=1)
    equal=(w==w[:,ai,None]).sum(axis=1)
    certain=(greater+equal<=2).mean()
    possible=(greater<2).mean()
    heuristic=np.mean(np.clip((2-greater)/equal,0,1))
    return certain,heuristic,possible
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--zip',required=True)
    p.add_argument('--stats-zip',required=True,help='Muse verified team stats ZIP')
    p.add_argument('--simulations',type=int,default=100000)
    p.add_argument('--retention',type=float,default=.65)
    p.add_argument('--mov-strength',type=float,default=.25)
    args=p.parse_args()
    with zipfile.ZipFile(args.zip) as z:
        games=load(z,'games.csv')
        remaining=load(z,'big12_2026_remaining_matchups.csv')
        standings=load(z,'big12_2026_standings.csv')
    with zipfile.ZipFile(args.stats_zip) as z:
        candidates=[n for n in z.namelist() if n.endswith('/historical_team_game_stats.csv')]
        if len(candidates)!=1:raise ValueError('Missing historical stats')
        stats=pd.read_csv(z.open(candidates[0]))
    stats=stats[(stats.fbs_vs_fbs==1)&(stats.score_flag.fillna('')!='completed_0_0')].copy()
    stats=stats.drop_duplicates(['espn_game_id','espn_team_id'],keep='first')
    cutoff=pd.Timestamp('2026-10-30T00:00:00Z')
    if len(remaining)!=39 or remaining.espn_game_id.duplicated().any():
        raise ValueError('Remaining conference schedule count or IDs invalid')
    if not (remaining.conference_game==True).all():
        raise ValueError('Nonconference game in remaining schedule')
    if not 0<=args.retention<=1 or not 0<=args.mov_strength<=1:
        raise ValueError('Retention and margin strength must be between 0 and 1')
    # Remove suspicious 0-0 games; ensure score labels are valid.
    games=games[~((games.home_score==0)&(games.away_score==0)&(games.is_completed==1))].copy()
    history,future=build(games,remaining,cutoff,stats,args.retention,args.mov_strength)
    # Source audit: raw scores used for each 2026 FBS-vs-FBS team game.
    audit=games[(games.season==2026)&(games.is_completed==1)&(games.fbs_vs_fbs==1)].copy()
    audit.to_csv('cfb_v13_2026_games_audit.csv',index=False)
    history.to_csv('cfb_v13_rebuilt_pregame_features.csv',index=False)
    future.to_csv('cfb_v13_future_features.csv',index=False)
    train=history[(history.season>=2014)&(history.season<=2022)]
    val=history[history.season==2023]
    test=history[history.season.isin([2024,2025])]
    print('Parameters: retention=',args.retention,'MOV strength=',args.mov_strength)
    print('Historical stat rows:',len(stats),'coverage:',stats.stats_available.mean())
    print('Splits:',len(train),len(val),len(test))
    metrics={}
    global FEATURE_MODE
    for kind in ('elo_logistic','baseline_logistic','logistic','nn'):
        FEATURE_MODE='elo' if kind=='elo_logistic' else ('baseline' if kind=='baseline_logistic' else 'full')
        algorithm='logistic' if kind in ('elo_logistic','baseline_logistic') else kind
        pv,pt=model(algorithm,train,val,test)
        metrics[kind]=log_loss(val.y,pv)
        print(f'{kind}: validation logloss={metrics[kind]:.4f}; '
              f'test acc={accuracy_score(test.y,pt>=.5):.4f}, '
              f'AUC={roc_auc_score(test.y,pt):.4f}, logloss={log_loss(test.y,pt):.4f}')
    selected=min(metrics,key=metrics.get)
    print('Selected by validation:',selected)
    FEATURE_MODE='elo' if selected=='elo_logistic' else ('baseline' if selected=='baseline_logistic' else 'full')
    # Historical training through 2025; 2023 retained as early-stopping validation for NN.
    final_train=history[(history.season<=2025)&(history.season!=2023)]
    _,p_future=model('logistic' if selected in ('elo_logistic','baseline_logistic') else selected,final_train,val,future)
    rem=remaining.copy();rem['p_home']=np.clip(p_future,.01,.99)
    rem[['espn_game_id','date','home_team','away_team','p_home']].to_csv('cfb_v13_remaining_probabilities.csv',index=False)
    asu=rem[(rem.home_team.str.contains('Arizona State'))|(rem.away_team.str.contains('Arizona State'))].copy()
    asu['p_ASU']=np.where(asu.home_team.str.contains('Arizona State'),asu.p_home,1-asu.p_home)
    print('\nASU win probabilities:\n',asu[['date','home_team','away_team','p_ASU']].to_string(index=False))
    for label,frame in [('Unconditional',rem)]:
        low,mid,high=simulate(frame,standings,args.simulations)
        print(f'{label}: qualification {low:.2%} to {high:.2%}; equal-random heuristic {mid:.2%}')
    print('CAUTION: official tiebreakers not implemented. Equal-random is not an official probability.')
    print('Future probabilities use static strength snapshots; no in-simulation Elo updates.')
    print('All 2026 predictions are exploratory until data and model are audited.')
    print('AUDIT: Verify 2026 results in cfb_v13_2026_games_audit.csv against an independent source.')
    print('NOTE: no independent source verification or official tiebreaker logic is provided.')
    print('WARNING: V13 parameters are exploratory and were not selected on an independent holdout.')
if __name__=='__main__':main()
