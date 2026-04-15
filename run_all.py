"""
═══════════════════════════════════════════════════════════════════════════
  EARNINGS FACTOR DECOMPOSITION — FULL PIPELINE
═══════════════════════════════════════════════════════════════════════════

  !pip install wrds statsmodels scipy -q
  %run /content/drive/MyDrive/Suresh1.github/run_all.py
═══════════════════════════════════════════════════════════════════════════
"""
import os, sys, getpass, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = '/content/drive/MyDrive/Suresh1.github'
sys.path.insert(0, REPO)
for d in ['inputs','outputs','outputs/figures','outputs/tables']:
    os.makedirs(os.path.join(REPO, d), exist_ok=True)

print("=" * 70)
print("  EARNINGS FACTOR DECOMPOSITION")
print("=" * 70)

# ── WRDS: ask once ──────────────────────────────────────────────────────
username = input("WRDS username: ")
password = getpass.getpass("WRDS password: ")
import wrds
db = wrds.Connection(wrds_username=username, wrds_password=password)
print("✓ Connected\n")

import src.data
src.data._db = db
src.data.get_connection = lambda: db

from src.config import *
from src.data import pull_full_sample
from src.abnormal_returns import build_event_panel
from src.api import compute_all_20_specs, compute_quintile_api, compute_spread, compute_api
from src.inference import bootstrap_spread, cross_sectional_regression, spearman_monotonicity
from src.backtest import run_all_strategies, strategy_binary_pead, compute_performance, ffc_regression
from src.transaction_costs import transaction_cost_analysis, compute_net_returns
from src.plotting import *
plt.rcParams.update(PLOT_PARAMS)

# ═══════════════════════════════════════════════════════════════════════
# 1. DATA
# ═══════════════════════════════════════════════════════════════════════
print("=" * 70); print("  STEP 1: DATA"); print("=" * 70)
data = pull_full_sample(fiscal_year=PRIMARY_FY)
print(f"\n  {len(data['ibes'])} IBES → {len(data['linked'])} linked → {len(data['msf']):,} stock-months\n")

# ═══════════════════════════════════════════════════════════════════════
# 2. EVENT PANELS
# ═══════════════════════════════════════════════════════════════════════
print("=" * 70); print("  STEP 2: EVENT PANELS"); print("=" * 70)
ep_fye = build_event_panel(data['linked'], data['msf'], data['mkt'],
    event_col='fiscal_ye', cache_name='ep_fye', fiscal_year=PRIMARY_FY)
ep_ann = build_event_panel(data['linked'], data['msf'], data['mkt'],
    event_col='event_date', cache_name='ep_ann', fiscal_year=PRIMARY_FY)
ep_ann_ext = build_event_panel(data['linked'], data['msf'], data['mkt'],
    event_col='event_date', w_end=PEAD_WINDOW_END,
    cache_name='ep_ann_ext', fiscal_year=PRIMARY_FY)
print(f"  FYE: {len(ep_fye):,} | Ann: {len(ep_ann):,} | Ext: {len(ep_ann_ext):,}\n")

# ═══════════════════════════════════════════════════════════════════════
# 3. TABLE 2 — 20 SPECIFICATIONS
# ═══════════════════════════════════════════════════════════════════════
print("=" * 70); print("  STEP 3: TABLE 2"); print("=" * 70)
all_api_df, table2 = compute_all_20_specs(ep_fye, ep_ann)
print(f"\n  {'Evt':<5} {'Model':<14} {'Surp':<5} {'Good':>8} {'Bad':>8} {'Spread':>9}")
print("  " + "-" * 52)
for _, r in table2.iterrows():
    s = "***" if abs(r['spread_pp'])>10 else("**" if abs(r['spread_pp'])>5 else("*" if abs(r['spread_pp'])>2 else ""))
    print(f"  {r['event']:<5} {r['model']:<14} {r['surprise']:<5} "
          f"{r['good_api']:>8.4f} {r['bad_api']:>8.4f} {r['spread_pp']:>7.2f}pp {s}")
table2.to_csv(os.path.join(TABLE_DIR, 'table2_spreads.csv'), index=False)
all_api_df.to_csv(os.path.join(TABLE_DIR, 'api_all_20.csv'), index=False)

# ═══════════════════════════════════════════════════════════════════════
# 4. FIGURES 1-2
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70); print("  STEP 4: FIGURES"); print("=" * 70)
fig1_ball_brown(all_api_df); plt.close(); print("  → fig1")
fig2_attenuation(all_api_df); plt.close(); print("  → fig2")

# ═══════════════════════════════════════════════════════════════════════
# 5. TABLE 3 — BOOTSTRAP CIs
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70); print("  STEP 5: TABLE 3 — BOOTSTRAP"); print("=" * 70)
specs = [
    (ep_fye,'ar_simple_ew','news_tsue','B&B Exact'),
    (ep_fye,'ar_simple_ew','news_sue','Simple EW+SUE+FYE'),
    (ep_ann,'ar_simple_ew','news_sue','Simple EW+SUE+Ann'),
    (ep_ann,'ar_mm_vw','news_sue','MM VW+SUE+Ann'),
    (ep_ann,'ar_ffc','news_sue','FFC+SUE+Ann'),
    (ep_ann,'ar_ffc','news_tsue','FFC+TSUE+Ann'),
]
print(f"\n  {'Spec':<25} {'Mean':>5} {'95% CI':>14} {'Sig':>4}")
print("  " + "-" * 52)
brows = []
for ep, ac, nc, lab in specs:
    b = bootstrap_spread(ep, ac, nc)
    sig = "Yes" if (b['ci_lo']>0 or b['ci_hi']<0) else "No"
    print(f"  {lab:<25} {b['mean']:>5.2f} [{b['ci_lo']:>5.2f},{b['ci_hi']:>5.2f}] {sig:>4}")
    brows.append({**b, 'spec': lab, 'significant': sig})
pd.DataFrame(brows).to_csv(os.path.join(TABLE_DIR, 'table3_bootstrap.csv'), index=False)

# ═══════════════════════════════════════════════════════════════════════
# 6. TABLE 4 — QUINTILES
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70); print("  STEP 6: TABLE 4 — QUINTILES"); print("=" * 70)
qapi = compute_quintile_api(ep_fye)
print(f"\n  {'Q':<7} {'N':>4} {'API(0)':>8} {'CumAR':>7}")
print("  " + "-" * 30)
for q in range(1,6):
    r = qapi[(qapi['quintile']==q)&(qapi['event_month']==0)]
    if len(r):
        t = "(Bad)" if q==1 else "(Good)" if q==5 else ""
        print(f"  Q{q}{t:>4} {r['n_firms'].values[0]:>4} "
              f"{r['api'].values[0]:>8.4f} {(r['api'].values[0]-1)*100:>6.2f}%")
rho, pv = spearman_monotonicity(qapi)
print(f"\n  Spearman rho={rho:.3f} (p={pv:.4f})")
fig3_quintiles(qapi); plt.close()
qapi.to_csv(os.path.join(TABLE_DIR, 'table4_quintiles.csv'), index=False)

# ═══════════════════════════════════════════════════════════════════════
# 7. TABLE 5 — REGRESSIONS
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70); print("  STEP 7: TABLE 5 — REGRESSIONS"); print("=" * 70)
reg = cross_sectional_regression(ep_ann, data['linked'], data['controls'], data['momentum'])
print(f"\n  {'Dep':<11} {'Spec':<18} {'B(SUE)':>8} {'t':>6} {'p':>6} {'R2':>6} {'N':>5}")
print("  " + "-" * 60)
for _, r in reg.iterrows():
    s="***" if r['p_value']<.01 else("**" if r['p_value']<.05 else("*" if r['p_value']<.1 else ""))
    print(f"  {r['dep']:<11} {r['model']:<18} {r['beta_sue']:>8.5f} {r['t_stat']:>6.2f} "
          f"{r['p_value']:>6.4f} {r['r2_adj']:>6.3f} {r['n']:>5} {s}")
reg.to_csv(os.path.join(TABLE_DIR, 'table5_regressions.csv'), index=False)

# ═══════════════════════════════════════════════════════════════════════
# 8. TABLE 6 — PEAD
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70); print("  STEP 8: TABLE 6 — PEAD"); print("=" * 70)
pead_parts = []
for ac, nc, lab in [('ar_simple_ew','news_sue','Simple(EW)+SUE'),
                     ('ar_mm_vw','news_sue','MM(VW)+SUE'),
                     ('ar_ffc','news_sue','FFC+SUE')]:
    a = compute_api(ep_ann_ext, ac, nc)
    if len(a)>0: a['spec']=lab; pead_parts.append(a)
pead_api = pd.concat(pead_parts, ignore_index=True)
print(f"\n  {'Spec':<20} {'M0':>5} {'M+3':>5} {'M+6':>5} {'M+9':>5} {'M+12':>5}")
print("  " + "-" * 48)
for lab in ['Simple(EW)+SUE','MM(VW)+SUE','FFC+SUE']:
    sub = pead_api[pead_api['spec']==lab]
    v = [f"{compute_spread(sub,t)[0]:>4.1f}" if not np.isnan(compute_spread(sub,t)[0]) else "  — " for t in [0,3,6,9,12]]
    print(f"  {lab:<20} {'  '.join(v)}")
fig4_pead(pead_api); plt.close()
pead_api.to_csv(os.path.join(TABLE_DIR, 'table6_pead.csv'), index=False)

# ═══════════════════════════════════════════════════════════════════════
# 9. TABLE 7 — SIZE-CONDITIONAL PEAD
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70); print("  STEP 9: TABLE 7 — SIZE PEAD"); print("=" * 70)
sz = data['controls'][['permno','ln_size']].dropna().copy()
sz['ln_size'] = sz['ln_size'].astype(float)
sz['size_group'] = np.where(sz['ln_size'] >= sz['ln_size'].median(), 'Large', 'Small')
ep_sz = ep_ann_ext.merge(sz[['permno','size_group']], on='permno', how='inner')
pead_sz = []
for g in ['Small','Large']:
    a = compute_api(ep_sz[ep_sz['size_group']==g], 'ar_simple_ew', 'news_sue')
    if len(a)>0: a['size']=g; pead_sz.append(a)
pead_size = pd.concat(pead_sz, ignore_index=True)
for g in ['Small','Large']:
    sub = pead_size[pead_size['size']==g]
    s0,_,_ = compute_spread(sub,0); s12,_,_ = compute_spread(sub,12)
    if not np.isnan(s0) and not np.isnan(s12):
        print(f"  {g} cap: {s0:.1f}pp → {s12:.1f}pp  (drift={s12-s0:+.1f}pp)")
fig5_pead_by_size(pead_size); plt.close()
pead_size.to_csv(os.path.join(TABLE_DIR, 'table7_pead_by_size.csv'), index=False)

# ═══════════════════════════════════════════════════════════════════════
# 10. MULTI-YEAR REPLICATION (FY2016-2023)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(f"  STEP 10: MULTI-YEAR REPLICATION ({FISCAL_YEARS[0]}-{FISCAL_YEARS[-1]})")
print("=" * 70)

multi_rows = []
for fy in FISCAL_YEARS:
    print(f"\n  ── FY{fy} ", end="")
    try:
        d = pull_full_sample(fiscal_year=fy)
        ep_f = build_event_panel(d['linked'], d['msf'], d['mkt'],
                                  event_col='fiscal_ye', cache_name='ep_fye', fiscal_year=fy)
        ep_a = build_event_panel(d['linked'], d['msf'], d['mkt'],
                                  event_col='event_date', cache_name='ep_ann', fiscal_year=fy)
        _, t2 = compute_all_20_specs(ep_f, ep_a)
        for _, r in t2.iterrows():
            rd = r.to_dict(); rd['fiscal_year'] = fy; multi_rows.append(rd)
        bb = t2[(t2['model']=='Simple (EW)')&(t2['surprise']=='TSUE')&(t2['event']=='FYE')]
        ffc = t2[(t2['model']=='FFC')&(t2['surprise']=='SUE')&(t2['event']=='Ann')]
        if len(bb)>0 and len(ffc)>0:
            bbsp=bb['spread_pp'].values[0]; ffcsp=ffc['spread_pp'].values[0]
            att=(1-ffcsp/bbsp)*100 if bbsp!=0 else np.nan
            print(f"B&B={bbsp:.1f}pp  FFC={ffcsp:.1f}pp  Att={att:.0f}%")
        else: print("(partial)")
    except Exception as e:
        print(f"FAILED: {e}")
pd.DataFrame(multi_rows).to_csv(os.path.join(TABLE_DIR, 'multi_year_spreads.csv'), index=False)

# ═══════════════════════════════════════════════════════════════════════
# 11. STRATEGIES — FY2023, 12-month hold
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70); print("  STEP 11: STRATEGIES (FY2023, 12-month hold)"); print("=" * 70)
factor_ret = data['mkt'][['mktrf','smb','hml','umd']].dropna()
summary, strategies = run_all_strategies(ep_ann_ext, data['controls'],
                                          factor_returns=factor_ret, holding_months=12)
print()
print(summary.to_string(index=False))
fig6_strategy_comparison(strategies); plt.close()
summary.to_csv(os.path.join(TABLE_DIR, 'strategy_comparison.csv'), index=False)

# ── Decile D10-D1 (FY2023) ──────────────────────────────────────────
print("\n  ── Decile D10-D1 (FY2023) ──")
dec_ep = ep_ann_ext.dropna(subset=['sue_value','ret']).copy()
dec_ep = dec_ep[dec_ep['event_month'].between(0,12)]
fsue = dec_ep.groupby('permno')['sue_value'].first().reset_index()
fsue['decile'] = pd.qcut(fsue['sue_value'].rank(method='first'), 10, labels=range(1,11)).astype(int)
dec_ep = dec_ep.merge(fsue[['permno','decile']], on='permno', how='left')
dec_m = []
for em in range(0,13):
    md = dec_ep[dec_ep['event_month']==em]
    d10=md[md['decile']==10]['ret'].astype(float).mean()
    d1=md[md['decile']==1]['ret'].astype(float).mean()
    if not np.isnan(d10) and not np.isnan(d1):
        dec_m.append({'event_month':em,'ls_ret':d10-d1,'n_d10':(md['decile']==10).sum(),'n_d1':(md['decile']==1).sum()})
if dec_m:
    dec_df = pd.DataFrame(dec_m)
    dp = compute_performance(dec_df['ls_ret'].values, name='Decile D10-D1')
    dp.n_firms_avg = dec_df[['n_d10','n_d1']].sum(axis=1).mean()
    da = ffc_regression(dec_df['ls_ret'].values, factor_ret) if len(dec_df)>=6 else {}
    print(f"  Cum: {dp.cumulative_return:.2f}%  Sharpe: {dp.sharpe_ratio:.3f}  "
          f"N: {dp.n_months}mo  Firms: {dp.n_firms_avg:.0f}")
    if da:
        print(f"  Alpha: {da['alpha']*10000:.1f} bps/mo (t={da['alpha_t']:.2f})")
    dec_df.to_csv(os.path.join(TABLE_DIR, 'decile_strategy.csv'), index=False)

# ═══════════════════════════════════════════════════════════════════════
# 12. MULTI-YEAR POOLED STRATEGIES (calendar-time)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print(f"  STEP 12: POOLED STRATEGIES ({FISCAL_YEARS[0]}-{FISCAL_YEARS[-1]})")
print("=" * 70)

pooled_bin = []
pooled_dec = []

for fy in FISCAL_YEARS:
    try:
        d = pull_full_sample(fiscal_year=fy)
        ep_ext = build_event_panel(d['linked'], d['msf'], d['mkt'],
                                    event_col='event_date', w_end=12,
                                    cache_name='ep_ann_ext', fiscal_year=fy)
        hold = ep_ext[ep_ext['event_month'].between(0,12)].copy()
        hold['cal_month'] = pd.to_datetime(hold['date']).dt.to_period('M')

        # Binary L/S
        for cm, md in hold.groupby('cal_month'):
            gr = md[md['news_sue']=='Good']['ret'].astype(float).mean()
            br = md[md['news_sue']=='Bad']['ret'].astype(float).mean()
            if not np.isnan(gr) and not np.isnan(br):
                pooled_bin.append({'cal_month':cm, 'ls_ret':gr-br, 'fy':fy})

        # Decile D10-D1
        fs = hold.groupby('permno')['sue_value'].first().reset_index()
        fs['decile'] = pd.qcut(fs['sue_value'].rank(method='first'), 10, labels=range(1,11)).astype(int)
        hd = hold.merge(fs[['permno','decile']], on='permno', how='left')
        for cm, md in hd.groupby('cal_month'):
            d10r = md[md['decile']==10]['ret'].astype(float).mean()
            d1r  = md[md['decile']==1]['ret'].astype(float).mean()
            if not np.isnan(d10r) and not np.isnan(d1r):
                pooled_dec.append({'cal_month':cm, 'ls_ret':d10r-d1r, 'fy':fy})
    except Exception as e:
        print(f"  FY{fy}: {e}")

# FF factors for alpha
ff_m = data['mkt'][['date','mktrf','smb','hml','umd']].dropna().copy()
ff_m['date'] = pd.to_datetime(ff_m['date'])
ff_m['cal_month'] = ff_m['date'].dt.to_period('M').dt.to_timestamp()

for label, pool in [("Binary L/S", pooled_bin), ("Decile D10-D1", pooled_dec)]:
    pdf = pd.DataFrame(pool)
    if len(pdf) == 0: continue
    pdf = pdf.groupby('cal_month')['ls_ret'].mean().reset_index()
    pdf['cal_month'] = pdf['cal_month'].dt.to_timestamp()
    pdf = pdf.sort_values('cal_month')
    mg = pdf.merge(ff_m[['cal_month','mktrf','smb','hml','umd']], on='cal_month', how='inner')

    ann_sharpe = mg['ls_ret'].mean()/mg['ls_ret'].std()*np.sqrt(12) if mg['ls_ret'].std()>0 else 0
    print(f"\n  {label} ({len(mg)} months):")
    print(f"    Avg monthly: {mg['ls_ret'].mean()*100:.2f}%  |  Sharpe: {ann_sharpe:.2f}")

    if len(mg) >= 6:
        a = ffc_regression(mg['ls_ret'].values, mg[['mktrf','smb','hml','umd']])
        if a:
            print(f"    FFC Alpha: {a['alpha']*10000:.1f} bps/mo (t={a['alpha_t']:.2f}, p={a['alpha_p']:.3f})")
            print(f"    MKT={a['mkt_beta']:.3f}  SMB={a['smb']:.3f}  HML={a['hml']:.3f}  UMD={a['umd']:.3f}")

    sname = label.replace(' ','_').replace('-','').replace('/','').lower()
    pdf.to_csv(os.path.join(TABLE_DIR, f'pooled_{sname}.csv'), index=False)

# ═══════════════════════════════════════════════════════════════════════
# 13. TRANSACTION COSTS
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70); print("  STEP 13: TRANSACTION COSTS"); print("=" * 70)
tc = transaction_cost_analysis(ep_ann_ext, data['controls'], fiscal_year=PRIMARY_FY)
if tc:
    result, df = strategy_binary_pead(ep_ann_ext, holding_months=12)
    if result and 'ls_ret' in df.columns:
        df_net = compute_net_returns(df, tc['firm_spreads'])
        gross = compute_performance(df['ls_ret'].values, name="Gross")
        net = compute_performance(df_net['ls_ret_net'].values, name="Net")
        print(f"\n  {'Metric':<22} {'Gross':>8} {'Net':>8}")
        print("  " + "-" * 40)
        print(f"  {'Cum Return (%)':<22} {gross.cumulative_return:>8.2f} {net.cumulative_return:>8.2f}")
        print(f"  {'Sharpe':<22} {gross.sharpe_ratio:>8.3f} {net.sharpe_ratio:>8.3f}")
        act = tc['median_spread_bps']
        avg_m = np.mean(df['ls_ret'].values)*100
        be = avg_m*10000/4
        print(f"\n  Breakeven: {be:.0f} bps | Actual: {act:.1f} bps")
        print(f"  >>> {'SURVIVES' if be>act else 'DOES NOT survive'} (margin: {abs(be-act):.0f} bps)")
        fig7_net_of_cost(df['ls_ret'].values, df_net['ls_ret_net'].values, act)
        plt.close()

# ═══════════════════════════════════════════════════════════════════════
# DONE
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70); print("  COMPLETE"); print("=" * 70)
print(f"\n  Tables → {TABLE_DIR}/")
for f in sorted(os.listdir(TABLE_DIR)):
    print(f"    {f:<40} {os.path.getsize(os.path.join(TABLE_DIR,f))/1024:>6.1f} KB")
print(f"\n  Figures → {FIGURE_DIR}/")
for f in sorted(os.listdir(FIGURE_DIR)):
    print(f"    {f:<40} {os.path.getsize(os.path.join(FIGURE_DIR,f))/1024:>6.1f} KB")
print(f"\n  GitHub ready: {REPO}/")
print("=" * 70)
