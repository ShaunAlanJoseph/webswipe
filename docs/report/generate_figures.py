"""Rebuild original report figures from the archived project evidence."""
from pathlib import Path
import json
import shutil
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'images'
EVIDENCE = ROOT / 'evidence'
OUT.mkdir(exist_ok=True)
EVIDENCE.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                     'axes.spines.top':False,'axes.spines.right':False,
                     'savefig.bbox':'tight','pdf.fonttype':42})
BLUE, GREEN, GRAY = '#215b92', '#237b55', '#68747d'

def save(fig, name):
    fig.savefig(OUT / (name+'.pdf'))
    fig.savefig(OUT / (name+'.png'), dpi=170)
    plt.close(fig)

# Numerical evidence is frozen beside this source; no external project paths are needed.
for required in ['capture_30fps.json', 'selected_capture.json', 'selected_decoder_scores.json',
                 'simulation_results2.json', 'pinch_case.json']:
    if not (EVIDENCE/required).is_file():
        raise FileNotFoundError('Missing archived evidence: ' + required)

# Introductory schematic: an ideal keyboard gesture, never a purported recording.
fig, ax = plt.subplots(figsize=(7.0,3.0))
centers={}
for row,offset,y in [('qwertyuiop',0,2),('asdfghjkl',.5,1),('zxcvbnm',1,0)]:
    for i,c in enumerate(row):
        x=i+offset;centers[c]=(x,y)
        ax.add_patch(Rectangle((x-.43,y-.40),.86,.80,edgecolor='#a0a9b1',facecolor='#f1f4f6',lw=.8))
        ax.text(x,y,c.upper(),ha='center',va='center',fontsize=12)
path=np.array([centers[c] for c in 'helo'])
ax.plot(path[:,0],path[:,1],color=BLUE,linewidth=2.8,zorder=2,alpha=.85)
ax.scatter(*path[0],color=GREEN,s=65,zorder=3)
ax.scatter(*path[-1],color=BLUE,s=65,zorder=3)
ax.text(5.3,3.00,'Pinch on H  →  trace H–E–L–O  →  release',ha='center',fontsize=11)
ax.text(5.3,-.95,'Decoder suggestions: hello  |  help  |  held  |  hero',ha='center',fontsize=11,color=BLUE)
ax.set_xlim(-.6,9.8);ax.set_ylim(-1.2,3.35);ax.set_aspect('equal');ax.axis('off')
save(fig,'interaction')

# Selected two-detector architecture, with separate capture and inference.
fig,ax=plt.subplots(figsize=(7,8.0))
ax.set_xlim(0,10);ax.set_ylim(0,12.4);ax.axis('off')
def box(x,y,w,h,title,sub='',color=BLUE):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.07,rounding_size=0.10',
                facecolor='#f4f7fa',edgecolor=color,lw=1.15))
    ax.text(x+w/2,y+h*.72 if sub else y+h/2,title,ha='center',va='center',fontsize=10.2,fontweight='bold',color=color)
    if sub:ax.text(x+w/2,y+h*.30,sub,ha='center',va='center',fontsize=8.7)
def arrow(a,b):
    ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=1.15,color=GRAY))
box(2.65,11.35,4.7,.70,'RGB webcam','640 × 480; mirrored image')
box(2.65,10.18,4.7,.73,'Latest-frame reader','Capture continues during inference')
box(2.65,8.86,4.7,.86,'Frozen MediaPipe tracker','Landmarks and handedness')
arrow((5,11.25),(5,11.00));arrow((5,10.09),(5,9.80))
box(.20,7.12,4.20,1.02,'Position branch','Midpoint → One Euro smoothing')
box(5.02,6.92,4.68,1.42,'Two YOLOv10n checkpoints','Primary: lighting-adapted\nReference: earlier crop model')
arrow((4,8.77),(2.3,8.23));arrow((6,8.77),(7.36,8.43))
box(.20,5.13,4.20,1.28,'Right-hand swipe','Primary + reference open-hand veto\nOnset recovery; release delay')
box(5.02,5.13,4.68,1.28,'Left-hand controls','Reference scores + geometry/motion\nShort pinch: cycle; hold: delete',GREEN)
arrow((2.3,7.03),(2.3,6.50));arrow((7.36,6.83),(7.36,6.50))
arrow((5.50,6.83),(4.1,6.50))
box(.20,3.66,4.20,.91,'Trajectory features','50 Hz interpolation → 50 × 39')
arrow((2.3,5.04),(2.3,4.66))
box(.20,1.94,4.20,1.12,'BiLSTM + attention + CTC','10,000-word lexicon; top four')
box(5.02,1.94,4.68,1.12,'Candidates and text','Commit, cycle, or delete',GREEN)
arrow((2.3,3.57),(2.3,3.15));arrow((4.49,2.50),(4.93,2.50));arrow((7.36,5.04),(7.36,3.15))
ax.text(5,.90,'Kaggle: task-specific training and evaluation',ha='center',fontsize=10,color=GRAY)
ax.text(5,.47,'Laptop CPU: tracking, visual inference, and word decoding',ha='center',fontsize=10,color=GRAY)
save(fig,'architecture')

# Recalculate each subject's bars from the archived JSON, not copied labels.
subjects=[0,1,2,4,5,6,7,8,9]
folds=[json.loads((EVIDENCE/f'd1_robustness_s{i}.json').read_text()) for i in subjects]
condition=next(k for k in folds[0]['conditions'] if k.startswith('webcam:'))
series=[(label,[100*f['conditions'][condition][key][0] for f in folds],color)
        for label,key,color in [('SHARK2, no prior','SHARK2+smooth@50Hz',GRAY),
                               ('Clean BiLSTM','clean@50Hz',BLUE),('Augmented BiLSTM','aug@50Hz',GREEN)]]
fig,ax=plt.subplots(figsize=(7,3.8));x=np.arange(len(subjects));w=.24
for i,(label,ys,color) in enumerate(series):ax.bar(x+(i-1)*w,ys,w,label=label,color=color)
ax.set_xticks(x,subjects);ax.set_xlabel('Held-out subject');ax.set_ylabel('Top-1 word accuracy (%)');ax.set_ylim(0,105)
ax.legend(loc='upper center',bbox_to_anchor=(.5,1.19),ncol=3,frameon=False,fontsize=8.5)
ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True);fig.tight_layout();save(fig,'loso')

# Historical and selected configurations, all using the same 29 attempts.
capture=json.loads((EVIDENCE/'capture_30fps.json').read_text())
old=json.loads((EVIDENCE/'decoder_30fps.json').read_text())
personal=json.loads((EVIDENCE/'personalized_recheck_30fps.json').read_text())['combined']
selected=json.loads((EVIDENCE/'selected_decoder_scores.json').read_text())['scores']
selected_capture=json.loads((EVIDENCE/'selected_capture.json').read_text())
last=[selected['old_sitting_recheck'],selected['old_standing_recheck']]
counts_capture=[sum(capture[f'{c}_{kind}']['single_80pct_span'] for c in ['sitting','standing']) for kind in ['baseline','crop','crop']]
counts_capture.append(sum(selected_capture[c]['recheck']['single_80pct_span'] for c in ['old_sitting','old_standing']))
counts1=[old['combined_baseline_neural_e1_p0']['top1'],old['combined_crop_neural_e1_p0']['top1'],personal['top1'],sum(x['top1'] for x in last)]
counts4=[old['combined_baseline_neural_e1_p0']['top4'],old['combined_crop_neural_e1_p0']['top4'],personal['top4'],sum(x['top4'] for x in last)]
fig,ax=plt.subplots(figsize=(7.4,4.0));x=np.arange(4);w=.24
for i,(label,counts,color) in enumerate([('Complete capture',counts_capture,GRAY),('Correct top-1',counts1,BLUE),('Correct top-4',counts4,GREEN)]):
    bars=ax.bar(x+(i-1)*w,np.array(counts)*100/29,w,label=label,color=color)
    for bar,n in zip(bars,counts):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+1.5,f'{n}/29',ha='center',fontsize=7.5)
ax.set_xticks(x,['Earliest detector\nOriginal decoder','Crop detector\nOriginal decoder','Crop detector\nPersonalized*','Lighting + reference\nPersonalized*'],fontsize=8)
ax.set_ylim(0,100);ax.set_ylabel('All-attempt percentage (%)');ax.grid(axis='y',alpha=.16);ax.set_axisbelow(True)
ax.legend(loc='upper center',bbox_to_anchor=(.5,1.17),ncol=3,frameon=False,fontsize=9)
fig.text(.5,-.015,'* Recheck of recordings already inspected during development.',ha='center',fontsize=9,color=GRAY)
fig.tight_layout();save(fig,'webcam_comparison')

d=json.loads((EVIDENCE/'simulation_results2.json').read_text());fig,axes=plt.subplots(1,2,figsize=(7,3.5))
e=d['E5'];ax=axes[0]
for key,label,color in [('shape','Shape only',GRAY),('shark2','Shape + location',BLUE),('perkey_lex','Per-key + lexicon',GREEN)]:
    ax.plot(e['sigmas'],np.array(e[key])*100,'o-',ms=3,label=label,color=color)
ax.set_xlabel('Noise σ (key widths)');ax.set_ylabel('Top-1 accuracy (%)');ax.set_ylim(50,103);ax.grid(alpha=.18)
ax.legend(frameon=False,fontsize=8,loc='lower left');ax.set_title('Ideal-path noise experiment',fontsize=10)
e=d['E6'];ax=axes[1]
for key,label,color in [('bbox','Bounding box',GRAY),('rms','RMS',BLUE)]:
    v=np.asarray(e[key]['margin_variance']);ax.loglog(e['N'],v/v[0],'o-',ms=3,label=f"{label}: slope {e[key]['loglog_slope']:.2f}",color=color)
n=np.array(e['N']);ax.loglog(n,n[0]/n,'--',lw=1,color=GREEN,label='Reference: 1/N')
ax.set_xlabel('Resampled point count N');ax.set_ylabel('Relative margin variance');ax.grid(alpha=.18)
ax.legend(frameon=False,fontsize=8);ax.set_title('Normalization and averaging',fontsize=10)
fig.tight_layout();save(fig,'synthetic')
# A real missed pinch: archived crops above recorded evidence and counterfactual replay.
d=json.loads((EVIDENCE/'pinch_case.json').read_text());rows=d['rows']
fig=plt.figure(figsize=(7,5.5));gs=fig.add_gridspec(3,3,height_ratios=[1.25,1,.7],hspace=.52)
for i,(name,label) in enumerate([('case_open.jpg','Before closure'),('case_pinch.jpg','Visible pinch'),('case_release.jpg','After release')]):
    ax=fig.add_subplot(gs[0,i]);ax.imshow(plt.imread(OUT/name));ax.set_title(label,fontsize=10);ax.axis('off')
t=np.array([r['relative_s'] for r in rows]);ax=fig.add_subplot(gs[1,:])
ax.plot(t,[r['pinch_score'] for r in rows],color=BLUE,label='Recorded pinch score')
ax.plot(t,[r['raw_gap'] for r in rows],color=GRAY,label='Recorded normalized gap')
ax.set_xlim(0,1.4);ax.set_ylim(0,1);ax.set_ylabel('Score / gap ratio');ax.grid(alpha=.16)
ax.legend(loc='upper right',fontsize=8,frameon=False)
ax=fig.add_subplot(gs[2,:]);ax.step(t,[int(r['old_on']) for r in rows],where='post',color=GRAY,label='Previous pinch state')
ax.step(t,[int(r['new_on']) for r in rows],where='post',color=GREEN,label='Updated replay state')
for r in rows:
    if r['new_action']=='next':ax.axvline(r['relative_s'],color=GREEN,linestyle=':',label='Recovered cycle')
ax.set_xlim(0,1.4);ax.set_ylim(-.1,1.35);ax.set_yticks([0,1],['off','on']);ax.set_xlabel('Seconds after prompt (app decision time)');ax.grid(alpha=.16)
ax.legend(loc='upper right',fontsize=8,frameon=False);save(fig,'pinch_case')
print('Rebuilt six evidence-based figures; app renderer image is retained separately.')
