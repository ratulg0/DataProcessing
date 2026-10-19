import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from src.data_loading import ROOT, find_target, load_cdc, load_uploaded
from src.preprocessing import candidate_clean, descriptive_statistics, quality_report
from src.visualization import plot_target,plot_distributions,plot_boxplots,plot_correlations,plot_missingness,plot_target_association,plot_column_bar

def analyse(name, bar_column=None):
    df=load_uploaded() if name=='uploaded' else load_cdc(); target=find_target(df)
    figures=ROOT/'outputs'/'figures'/name; reports=ROOT/'outputs'/'reports'/name; reports.mkdir(parents=True,exist_ok=True)
    print(f'\n{name.upper()}: {df.shape[0]:,} rows × {df.shape[1]} columns'); print('Target:',target,'| duplicate rows:',df.duplicated().sum()); print(df[target].value_counts(dropna=False))
    quality_report(df).to_csv(reports/'data_quality.csv',index=False); descriptive_statistics(df).to_csv(reports/'descriptive_statistics.csv',index=False)
    processed=candidate_clean(df); print('Missing values before:',int(df.isna().sum().sum()),'after candidate cleaning:',int(processed.isna().sum().sum()))
    plot_target(df,target,figures); plot_distributions(df,target,figures); plot_boxplots(df,target,figures); plot_correlations(df,target,figures); plot_missingness(df,figures); plot_target_association(df,target,figures); plot_column_bar(df, bar_column or target, figures); plt.close('all')
    print('Figures:',figures,'\nReports:',reports)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset',choices=['uploaded','cdc','all'],default='uploaded'); p.add_argument('--bar-column',default=None,help='Column to plot as a bar chart (defaults to the detected target column).'); a=p.parse_args()
    if a.dataset in ('uploaded','all'): analyse('uploaded', a.bar_column)
    if a.dataset in ('cdc','all'): analyse('cdc', a.bar_column)
if __name__=='__main__': main()
