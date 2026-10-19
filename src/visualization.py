from pathlib import Path
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
sns.set_theme(style='whitegrid', context='notebook')

def savefig(fig, outdir, name):
    outdir=Path(outdir); outdir.mkdir(parents=True, exist_ok=True); fig.savefig(outdir/name, dpi=160, bbox_inches='tight')

def plot_target(df, target, outdir):
    counts=df[target].fillna('Missing').astype(str).value_counts().sort_index(); fig,ax=plt.subplots(figsize=(8,4.5)); ax.bar(counts.index, counts.values); ax.set(title=f'Target balance: {target}',xlabel='Class',ylabel='Rows')
    total=counts.sum()
    for bar,val in zip(ax.patches,counts.values): ax.annotate(f'{val:,} ({val/total:.1%})',(bar.get_x()+bar.get_width()/2,bar.get_height()),ha='center',va='bottom',fontsize=9)
    fig.tight_layout(); savefig(fig,outdir,'01_target_balance.png'); return fig

def plot_distributions(df,target,outdir,limit=24):
    cols=[c for c in df.select_dtypes(include=np.number) if c!=target][:limit]; rows=max(1,math.ceil(len(cols)/3)); fig,axes=plt.subplots(rows,3,figsize=(15,rows*3.3),squeeze=False)
    for ax,col in zip(axes.flat,cols): ax.hist(df[col].dropna(), bins='auto'); ax.set_title(col); ax.set_xlabel(col); ax.set_ylabel('Count')
    for ax in list(axes.flat)[len(cols):]: ax.remove()
    fig.suptitle('Numeric feature distributions',y=1.002); fig.tight_layout(); savefig(fig,outdir,'02_distributions.png'); return fig

def plot_boxplots(df,target,outdir,limit=18):
    cols=[c for c in df.select_dtypes(include=np.number) if c!=target][:limit]
    if not cols: return None
    rows=max(1,math.ceil(len(cols)/3)); fig,axes=plt.subplots(rows,3,figsize=(15,rows*3.3),squeeze=False)
    for ax,col in zip(axes.flat,cols):
        groups=[df.loc[df[target]==label,col].dropna().values for label in df[target].dropna().unique()]
        labels=[str(label) for label in df[target].dropna().unique()]
        ax.boxplot(groups, labels=labels); ax.set_title(col); ax.tick_params(axis='x', rotation=25)
    for ax in list(axes.flat)[len(cols):]: ax.remove()
    fig.suptitle('Feature ranges by target class',y=1.002); fig.tight_layout(); savefig(fig,outdir,'03_boxplots_by_target.png'); return fig

def plot_correlations(df,target,outdir,max_cols=24):
    num=df.select_dtypes(include=np.number)
    if target not in num: return None
    if len(num.columns)>max_cols:
        ranked=num.corr()[target].drop(target).abs().nlargest(max_cols-1); num=num[list(ranked.index)+[target]]
    corr=num.corr(); fig,ax=plt.subplots(figsize=(max(9,len(corr.columns)*.55),max(7,len(corr.columns)*.48))); image=ax.imshow(corr.values, vmin=-1, vmax=1, aspect='auto'); ax.set_xticks(range(len(corr.columns)), labels=corr.columns, rotation=90); ax.set_yticks(range(len(corr.index)), labels=corr.index); fig.colorbar(image, ax=ax, label='Pearson r'); ax.set_title('Pearson correlation (numeric columns)'); fig.tight_layout(); savefig(fig,outdir,'04_correlation_heatmap.png'); return fig

def plot_missingness(df,outdir):
    pct=df.isna().mean().mul(100).sort_values(ascending=False); fig,ax=plt.subplots(figsize=(max(8,len(pct)*.38),4.5)); ax.bar(pct.index, pct.values); ax.set(title='Missing values by feature',xlabel='Feature',ylabel='Missing (%)'); ax.tick_params(axis='x',rotation=75); fig.tight_layout(); savefig(fig,outdir,'05_missingness.png'); return fig

def plot_target_association(df,target,outdir):
    if target not in df.select_dtypes(include=np.number): return None
    corr=df.select_dtypes(include=np.number).corr()[target].drop(target).sort_values(); fig,ax=plt.subplots(figsize=(9,max(4.5,len(corr)*.32))); ax.barh(corr.index, corr.values); ax.axvline(0,linestyle='--',linewidth=1); ax.set(title=f'Pearson correlation with {target}',xlabel='Correlation',ylabel='Feature'); fig.tight_layout(); savefig(fig,outdir,'06_target_correlations.png'); return fig
