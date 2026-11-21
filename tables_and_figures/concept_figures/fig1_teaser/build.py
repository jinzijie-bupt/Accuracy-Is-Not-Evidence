"""Figure 1 left: reference three-tier matched-placebo concept, in print points.

Uses only the shared diagram primitives and canonical photographs. The original
desktop reference and other figure versions are never modified.
"""
from pathlib import Path
import sys, json, hashlib
import numpy as np
import pymupdf as fitz
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT/'scripts'))
from diagram_common import Fig, C, frame, fontfile, PARSER, FontProperties

W, H = 174.24, 165.6
TIMES = [.8, 7.2, 11, 15, 18, 20.5, 23.9, 25, 28]
E, O = [3, 4, 5], [0, 7, 8]
QUERY = 'another person is fixing a doorknob.'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def textwidth(s, size=7.2, bold=False):
    return fitz.Font(fontfile=str(fontfile(bold))).text_length(s, fontsize=size)

def mathwidth(s, size=8):
    return float(PARSER.parse('$'+s+'$', dpi=72,
                 prop=FontProperties(family='STIXGeneral',size=size)).width)

def paired_label(fig, key, plain, expr, cx, baseline, *, math_first=False, color=C['ink']):
    """Place one ordinary STIX label beside a real STIX math expression."""
    pw, mw = textwidth(plain), mathwidth(expr)
    start = cx-(pw+mw+1.5)/2
    if math_first:
        fig.math(key+'-math',expr,start,baseline+.25,size=8,color=color)
        fig.text(key+'-plain',plain,start+mw+1.5,baseline,size=7.2,color=color)
    else:
        fig.text(key+'-plain',plain,start,baseline,size=7.2,color=color)
        fig.math(key+'-math',expr,start+pw+1.5,baseline+.25,size=8,color=color)

def reference_brace(fig,x1,x2,y1,y2):
    # Control points are the actual brace in desktop_fig1/editable.svg,
    # transformed to the selected cells. Smooth samples remain editable vector
    # geometry; no generated letter or frame content is retained.
    def B(p0,p1,p2,p3):
        t=np.linspace(0,1,15)[:,None]
        return ((1-t)**3*np.array(p0)+3*(1-t)**2*t*np.array(p1)+
                3*(1-t)*t*t*np.array(p2)+t**3*np.array(p3)).tolist()
    p=B([488.75,205.25],[490.6,193.45],[500.2,188],[511.5,188])
    p+=[[649.6,188]]
    p+=B([649.6,188],[662.7,188],[670.95,178.5],[671.6,165.9])[1:]
    p+=B([671.6,165.9],[672.35,178.8],[682.25,188],[696.1,188])[1:]
    p+=[[832.3,188]]
    p+=B([832.3,188],[844.95,188],[853.6,194.65],[855.6,205.8])[1:]
    pts=[(x1+(x-488.75)/(855.6-488.75)*(x2-x1),
          y1+(y-165.9)/(205.8-165.9)*(y2-y1)) for x,y in p]
    fig.line('annotated-interval-curly-brace',pts,color=C['evidence'],lw=.75)

def make():
    fig=Fig(W,H,'Matched-placebo audit: the same answer can reflect three mechanisms')
    # Tier 1: one query and one shared, authentic nine-frame video.
    fig.rect('query-box',2,1,W-4,15,fill=C['pale'],stroke=C['edge'],lw=.75,r=2)
    qw=fig.text('query-label','query',7,10.5,size=7.2,bold=True)
    fig.text('query-content',QUERY,7+qw+3,10.5,size=7.2)
    paired_label(fig,'annotated-label','annotated interval',r'y^{\!\star}',W/2,22.9,color=C['evidence'])
    x,w,main_y,main_h=3,W-6,30,17
    main=fig.filmstrip('clean-video',TIMES,x,main_y,w,main_h,evidence=E,placebo=O)
    reference_brace(fig,main[E[0]][0],main[E[-1]][0]+main[E[-1]][2],26.8,28.9)
    fig.text('placebo-count','placebo frames (same number)',W/2,54.5,size=7.2,
             color=C['placebo'],anchor='center')

    # Tier 2: identical ordered slots and crop aspect ratio in the two edits.
    # Fixed 2pt sprocket rims in the shared primitive are accounted for here.
    smallw=(W-12)/2
    smallcw=(smallw-8)/9
    smallh=4+smallcw/(main[0][2]/main[0][3])
    small_y=58
    leftx,rightx=3,W-3-smallw
    centres=[leftx+smallw/2,rightx+smallw/2]
    # Keep the shafts outside the count label's full text extent.
    fig.arrow('to-evidence-edit',[(36,47.5),(36,57)],lw=.75,head=2.7)
    fig.arrow('to-placebo-edit',[(W-36,47.5),(W-36,57)],lw=.75,head=2.7)
    edited_E=fig.filmstrip('evidence-edit',TIMES,leftx,small_y,smallw,smallh,evidence=E,masked=E)
    edited_O=fig.filmstrip('placebo-edit',TIMES,rightx,small_y,smallw,smallh,placebo=O,masked=O)
    for key,xx,cx,fill,color,expr,label in [
        ('evidence-label',leftx,centres[0],C['evidence_fill'],C['evidence'],r'x_E',': evidence masked'),
        ('placebo-label',rightx,centres[1],C['placebo_fill'],C['placebo'],r'x_O',': placebo masked')]:
        fig.rect(key+'-box',xx,70.5,smallw,11,fill=fill,stroke=C['edge'],lw=.75,r=2)
        paired_label(fig,key,label,expr,cx,78.0,math_first=True,color=color)

    # Tier 3: the answer is identical; support changes distinguish mechanisms.
    fig.rect('same-answer-box',40,84,94.24,12,fill=C['pale'],stroke=C['edge'],lw=.75,r=2)
    fig.math('same-answer',r'\mathrm{score\ the\ same\ answer}\ y^{\!\star}',W/2,92.0,size=8,anchor='center')
    cardw=(W-12)/3
    xs=[2,6+cardw,10+2*cardw]
    centres3=[xx+cardw/2 for xx in xs]
    for i,cx in enumerate(centres3):
        start=[(47,95.8),(W/2,96),(127.24,95.8)][i]
        fig.arrow('same-answer-to-card-'+str(i),[start,(cx,102.4)],lw=.75,head=2.7)
    for i,(xx,cx,title1,title2,heights) in enumerate(zip(xs,centres3,
            ['uses the','ignores the','reacts to'],['event','frames','any edit'],
            [(12,.5),(.5,.5),(12,12)])):
        key='model-'+str(i+1)
        fig.rect(key+'-card',xx,104,cardw,60.5,fill=C['pale'],stroke=C['edge'],lw=.75,r=2.4)
        fig.text(key+'-title-line1',title1,cx,113.0,size=7.2,bold=True,anchor='center')
        fig.text(key+'-title-line2',title2,cx,124.0,size=7.2,bold=True,anchor='center')
        baseline=143.6
        for j,(off,height,color) in enumerate(zip([10,35],heights,[C['evidence'],C['placebo']])):
            fig.rect(key+'-bar-'+str(j),xx+off,baseline-height,10,height,fill=color,stroke=None)
            fig.math(key+'-drop-'+str(j),r'd_E' if j==0 else r'd_O',xx+off+5,152.2,size=8,anchor='center')
        fig.line(key+'-baseline',[(xx+6,baseline),(xx+cardw-6,baseline)],color=C['ink'],lw=.75)
        if i==0:
            bx=xx+27;top=baseline-heights[0];bottom=baseline-heights[1]
            fig.line('psi-difference-bracket',[(bx-1.8,top),(bx+1.8,top),(bx,top),(bx,bottom),(bx-1.8,bottom),(bx+1.8,bottom)],color=C['ink'],lw=.75)
            fig.math('psi-bracket-label',r'\psi',xx+30.3,139.5,size=8)
            fig.math('psi-definition',r'\psi=d_E-d_O',cx,162.0,size=8,anchor='center')
        else:
            fig.math(key+'-psi-zero',r'\psi\approx0',cx,162.0,size=8,color=C['gray'],anchor='center')
    return fig,dict(clean=main,evidence_edit=edited_E,placebo_edit=edited_O)

def main():
    manifest=json.loads((ROOT/'frame_manifest.json').read_text(encoding='utf-8'))
    bytime={float(r['timestamp_sec']):r for r in manifest['frames']}
    for t in TIMES:assert sha(frame(t))==bytime[t]['sha256']
    fig,boxes=make()
    target=fig.save(HERE,'fig1_teaser')
    keep=dict(dataset=manifest['dataset'],split=manifest['split'],video_id=manifest['video_id'],
              video_sha256=manifest['video_sha256'],query=QUERY,annotated_interval=manifest['interval_A'],
              times=TIMES,evidence_slots=E,placebo_slots=O,frames=[bytime[t]for t in TIMES],
              frame_geometry=boxes,bar_heights_pt=[[12,.5],[.5,.5],[12,12]],
              card_titles=['uses the event','ignores the frames','reacts to any edit'],
              size_pt=[W,H],bar_values_are_illustrative=True)
    (HERE/'source_manifest.json').write_text(json.dumps(keep,indent=2),encoding='utf-8')
    print(target)

if __name__=='__main__':main()
