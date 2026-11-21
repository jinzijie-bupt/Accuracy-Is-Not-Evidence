"""Complete desktop Figure 4 against C section 2.3 and the current audit."""
from pathlib import Path
import sys,json
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'scripts'))
from diagram_common import *

DY=9   # v2 (2026-09-25): one added line of query text under the event labels
f=Fig(396,184+DY,'Query-crossed control with a fixed scored answer')
f.text('fixed-answer',"scored answer = A's interval (fixed)",198,8,size=8,anchor='center')
times=[.8,4,7.2,11,15,18,23.9,25,28]
cells=f.filmstrip('shared-real-video',times,4,17,388,26,holes=True,border=False)
events={'B':([0,1],C['partner'],'event B  0.0–4.6 s','“a person opens a bag.”'),
        'A':([4,5],C['evidence'],'event A  13.2–23.7 s','“another person is fixing a doorknob.”')}
for event,(indices,col,label,query) in events.items():
    for i in indices:
        x,y,w,h=cells[i];f.rect(f'event-{event}-frame-{i}',x,y,w,h,stroke=col,lw=1.2)
    x0=cells[indices[0]][0];x1=cells[indices[-1]][0]+cells[indices[-1]][2]
    f.line('event-'+event+'-bracket',[(x0,46),(x0,50),(x1,50),(x1,46)],color=col,lw=.75)
    f.text('event-'+event+'-label',label,(x0+x1)/2,61,size=7.5,color=col,anchor='center')
    f.text('event-'+event+'-query',query,max((x0+x1)/2,40),70,size=7,italic=True,color=col,anchor='center')

f.text('drop-label','likelihood drop',4,76+DY,size=8,color=C['gray'])
f.text('edit-A-label','edit A frames',126,76+DY,size=8,color=C['evidence'],anchor='center')
f.text('edit-B-label','edit B frames',207,76+DY,size=8,color=C['partner'],anchor='center')
gx,gy,cw,ch=86,83+DY,81,42
f.line('matrix-top',[(gx,gy),(gx+2*cw,gy)],color='#D0D0D0',lw=.5)
f.line('matrix-mid',[(gx,gy+ch),(gx+2*cw,gy+ch)],color='#D0D0D0',lw=.5)
f.line('matrix-bottom',[(gx,gy+2*ch),(gx+2*cw,gy+2*ch)],color='#D0D0D0',lw=.5)
f.line('matrix-left',[(gx,gy),(gx,gy+2*ch)],color='#D0D0D0',lw=.5)
f.line('matrix-divider',[(gx+cw,gy),(gx+cw,gy+2*ch)],color='#D0D0D0',lw=.5)
f.line('matrix-right',[(gx+2*cw,gy),(gx+2*cw,gy+2*ch)],color='#D0D0D0',lw=.5)
f.text('own-query','own query (A)',4,108+DY,size=8)
f.text('crossed-query','crossed query (B)',4,150+DY,size=8)
bars=[]
for r in range(2):
    for c in range(2):
        cx=gx+c*cw;cy=gy+r*ch;baseline=cy+ch-7
        high=r==c;bh=27 if high else 6;col=C['evidence'] if c==0 else C['partner']
        # v2: a zero line per cell instead of a separate grey 'clean' mark; the bar is centred on it.
        f.line(f'zero-line-{r}-{c}',[(cx+14,baseline),(cx+cw-14,baseline)],color='#9A9A9A',lw=.6)
        f.rect(f'edited-drop-{r}-{c}',cx+cw/2-6.5,baseline-bh,13,bh,fill=col,stroke=None)
        bars.append(dict(row_query='A' if r==0 else 'B',edit_event='A' if c==0 else 'B',relative_height=bh/27,clean_drop=0))

def glyph(key,x,y,mode):
    for rr in range(2):
        for cc in range(2):
            filled=(rr==cc) if mode=='diagonal' else cc==0
            f.rect(key+f'-{rr}-{cc}',x+cc*8,y+rr*8,8,8,fill=C['ink'] if filled else '#FFFFFF',stroke=C['edge'],lw=.5)

glyph('query-sensitive-pattern',263,89+DY,'diagonal')
f.text('query-reader','Reads the query',285,93+DY,size=8,bold=True)
f.text('reversal','contrast reverses',285,104+DY,size=7.5)
# v3 (, round5 A11): psi is d_E - d_O in the paper and Delta psi_own/cross are trained-minus-base
# changes; the raw partner contrast drawn here carries no symbol, so the panels name the rows instead.
f.math('reversal-signs',r'\mathrm{own}>0,\quad\mathrm{crossed}<0',285,117+DY,size=8)
f.line('hypothesis-divider',[(261,125+DY),(392,125+DY)],color='#D0D0D0',lw=.5)
glyph('position-pattern',263,135+DY,'column')
f.text('position-reader-1','Responds to edits at',285,137+DY,size=7.5,bold=True)
f.text('position-reader-2',"the answer's timestamps",285,148+DY,size=7.5)
f.math('position-equality',r'\mathrm{crossed}=\mathrm{own}',285,161+DY,size=8)
f.text('contrast-definition','partner contrast under a query:',4,179+DY,size=7.5)
f.math('contrast-equation',r'\mathrm{drop}(\mathrm{edit\ A})-\mathrm{drop}(\mathrm{edit\ B})',108,179+DY,size=8)

path=f.save(HERE,'fig4_cross_query')
doc=fitz.open(path);doc[0].get_pixmap(matrix=fitz.Matrix(3,3),alpha=False).save(HERE/'preview.png')
record={'source':'source/desktop_fig4/ editable.svg and desktop_fig4.png','requirements':'figures_C_appendix_system.md section 2.3',
        'retained_structure':'real filmstrip above 2x2 grid, two right-side pattern panels',
        'times_sec':times,'event_A_indices':[4,5],'event_B_indices':[0,1],
        'fixed_scored_interval_sec':[13.2,23.7],'fixed_edit_A_indices':[4,5],'fixed_edit_B_indices':[0,1],
        'bars_are_design_predictions_not_measurements':True,'bar_geometry':bars,
        'corrections':['All photographic cells replaced by authentic dataset frames.','A and B brackets mapped to the actual annotated events.','Clean reference is zero, not a nonzero grey drop.','Partner contrasts are the raw drop(edit A) - drop(edit B) of one model and carry no symbol (psi is d_E - d_O in the paper; Delta psi_own/Delta psi_cross are trained-minus-base changes); a sign flip is not asserted for Delta psi.'],
        'query_A':'another person is fixing a doorknob.','query_B':'a person opens a bag.',
        'pdf_sha256':sha(path),'svg_sha256':sha(HERE/'editable.svg')}
(HERE/'scientific_manifest.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
(HERE/'caption.tex').write_text(r'''\caption{\textbf{The query-crossed control holds the answer and edits fixed.} Both queries score event A's interval in the same video. Columns edit A or B; rows change only the query. Bars illustrate the predicted likelihood drops, not measurements; grey marks indicate the zero clean reference. A query-sensitive model reverses its partner contrast, whereas a response to the answer's timestamps is unchanged. Figure~\ref{fig:readouts}a reports the trained-minus-base changes in these contrasts.}
\label{fig:cross_design}
''',encoding='utf-8')
print('Figure 4 created:',path)
