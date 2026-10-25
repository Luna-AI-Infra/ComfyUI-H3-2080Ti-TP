"""Render three single-question figures from the published measurement subset."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / 'docs/paper/measurements.json').read_text(encoding='utf-8'))
OUT = ROOT / 'docs/figures'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11.5,'axes.titlesize':14,
                     'axes.labelsize':11.5,'xtick.labelsize':10.5,'ytick.labelsize':11,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'axes.edgecolor':'#aab8c4','axes.labelcolor':'#334155',
                     'text.color':'#172f43','figure.facecolor':'white','axes.axisbelow':True})


def save(fig, name):
    fig.savefig(OUT / name, dpi=220, facecolor='white')
    plt.close(fig)


def main():
    fig, ax = plt.subplots(figsize=(7.4,3.55))
    fig.subplots_adjust(left=.23,right=.86,top=.82,bottom=.24)
    stages=[('conditioning','#2b5876','Conditioning'),('sampler','#17817b','Sampler'),
            ('video_vae','#dba548','Video VAE'),('other','#cbd5df','Other')]
    for i,row in enumerate(DATA['e60']):
        left=0
        for key,color,label in stages:
            value=row.get(key,row['total']-row['conditioning']-row['sampler']-row['video_vae'])
            ax.barh(i,value,left=left,height=.55,color=color)
            if key=='sampler':ax.text(left+value/2,i,f'{value:.1f}',ha='center',va='center',color='white',fontsize=11)
            left+=value
        ax.text(row['total']+4,i,f"{row['total']:.3f} s",va='center',fontsize=10.5,fontweight='bold')
    ax.set_yticks(range(3),[r['label'] for r in DATA['e60']]);ax.invert_yaxis();ax.set_xlim(0,250)
    ax.set_xlabel('Complete request time (seconds)');ax.grid(axis='x',alpha=.17)
    ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0,pad=10)
    fig.suptitle('E60 | CPU pack reuse across length changes',x=.05,ha='left',y=.98,fontweight='bold')
    fig.legend(handles=[Patch(facecolor=c,label=l) for _,c,l in stages],loc='lower center',ncol=4,
               frameon=False,fontsize=10,bbox_to_anchor=(.52,.005),handlelength=1.1,columnspacing=1.1)
    save(fig,'e60-stage-times.png')

    fig,ax=plt.subplots(figsize=(7.4,2.95))
    fig.subplots_adjust(left=.22,right=.87,top=.73,bottom=.24)
    d=DATA['e44'];values=[d['baseline_ms'],d['tuned_ms']]
    ax.barh([0,1],values,color=['#adbcca','#17817b'],height=.50)
    for i,v in enumerate(values):ax.text(v+3,i,f'{v:.3f} ms',va='center',fontweight='bold')
    ax.set_yticks([0,1],['Vendor FC2','Tuned FC2']);ax.invert_yaxis();ax.set_xlim(0,250)
    ax.set_xlabel('Whole TP2 block latency (milliseconds)');ax.grid(axis='x',alpha=.17)
    ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0,pad=10)
    fig.suptitle('E44 | Same-precision tuning: whole-block effect',x=.05,ha='left',y=.98,fontweight='bold')
    fig.text(.05,.80,'1.013x speedup; original output precision retained',fontsize=11,color='#536779')
    save(fig,'e44-fc2-tuning.png')

    fig,ax=plt.subplots(figsize=(7.4,3.35))
    fig.subplots_adjust(left=.26,right=.94,top=.80,bottom=.25)
    for row in DATA['e52']:
        y=row['rank']*2+(0 if row['kind']=='compute' else 1)
        color='#2b5876' if row['kind']=='compute' else '#dba548'
        ax.broken_barh([(row['start_s'],row['duration_s'])],(y-.24,.48),facecolors=color)
        label=f"{row['duration_s']:.3f} s"
        x=row['start_s']+row['duration_s']+.05
        ax.text(x,y,label,va='center',fontsize=10.5)
    ax.set_yticks(range(4),['GPU0 compute span','GPU0 VAE upload','GPU1 compute span','GPU1 VAE upload'])
    ax.invert_yaxis();ax.set_xlim(-.15,3.65);ax.set_xlabel('Seconds from final-group compute start')
    ax.grid(axis='x',alpha=.17);ax.spines['left'].set_visible(False);ax.tick_params(axis='y',length=0,pad=9)
    fig.suptitle('E52 | Final-group span and decoder prefetch',x=.05,ha='left',y=.98,fontweight='bold')
    fig.text(.05,.04,'Separate capture; stream spans include waits, not kernel-active time.',fontsize=10,color='#536779')
    save(fig,'e52-transfer-overlap.png')


if __name__ == '__main__':
    main()
