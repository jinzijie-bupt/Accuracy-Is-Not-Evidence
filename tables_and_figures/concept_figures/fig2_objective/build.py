"""0924 training-objective concept figure; coordinates are final printed points.

Uses the shared diagram engine and shared authentic frame manifest. Never edits
the manuscript, source reference, or shared frame pool.
"""
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import json
import hashlib
import itertools
import subprocess

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'scripts'))
from diagram_common import Fig, C, ROOT, PAPER, sha

M = json.loads((ROOT / 'frame_manifest.json').read_text(encoding='utf-8'))
TIMES = M['audit_objective_times']
E = M['audit_objective_E']
O = M['audit_objective_O']
assert TIMES == [.8, 15, 18, 25, 28] and E == [1, 2] and O == [0, 4]
assert M['event_A'] == 'another person is fixing a doorknob.'
assert M['interval_A'] == [13.2, 23.7]

fig = Fig(396, 168, 'Training an evidence-dependent objective with matched placebo edits')
fig.text('stage-input', 'Training pair', 4, 10, 8.5, bold=True)
fig.text('stage-scores', 'Log-likelihoods', 109, 10, 8.5, bold=True)
fig.text('stage-loss', 'Operator loss', 190, 10, 8.5, bold=True)
fig.math('operator-loss-symbol', r'\mathcal{L}_{\mathrm{op}}', 244, 10, 8)
fig.text('stage-total', 'Total loss', 315, 10, 8.5, bold=True)

# One literal dataset query and annotated answer are fixed for all six scores.
fig.text('query', 'query q: another person is fixing a doorknob.', 4, 24, 7.5)
fig.math('fixed-answer', r'y^{\!\star}=[13.2,\,23.7]\,\mathrm{s}', 4, 36, 8)
fig.text('fixed-score-inputs', 'fixed', 80, 36, 7, color=C['gray'])
fig.text('theta-status', 'trainable', 125, 36, 7, anchor='middle')
fig.math('theta-model', r'\theta\;\mathrm{(LoRA)}', 125, 48, 8, anchor='middle')
fig.text('ref-status', 'frozen', 162, 36, 7, anchor='middle')
fig.text('ref-model', 'reference', 162, 48, 7.5, anchor='middle')

rows = [('clean', 'clean', r'x', 54, 59, (), C['gray']),
        ('evidence', 'evidence edit', r'x_E', 91, 96, E, C['evidence']),
        ('placebo', 'placebo edit', r'x_O', 128, 133, O, C['placebo'])]
slot_records = []
for key, label, symbol, label_y, y, masked, role_color in rows:
    label_w = fig.text(key + '-label', label, 4, label_y, 8, color=role_color)
    fig.math(key + '-symbol', symbol, 4 + label_w + 4, label_y, 8, color=role_color)
    fig.rect(key + '-strip-outline', 3.6, y-.4, 90.8, 22.8,
             fill=None, stroke=C['edge'], lw=.5, r=1)
    boxes = fig.filmstrip(key + '-strip', TIMES, 4, y, 90, 22,
                         evidence=E if key == 'evidence' else (),
                         placebo=O if key == 'placebo' else (),
                         masked=masked, holes=False)
    slot_records.append(dict(row=key, boxes=boxes, masked=list(masked)))
    fig.arrow(key + '-to-score', [(97, y+11), (105, y+11)], head=2.5)
    for col, x, sub in [('theta',109,r'\theta'),('ref',146,r'\mathrm{ref}')]:
        fig.rect(key + '-cell-' + col, x, y, 33, 22, fill=C['pale'], r=1.5)
        fig.math(key + '-score-' + col,
                 r'\ell_{' + sub + '}(' + symbol + ')',
                 x+16.5, y+13.5, 8, anchor='middle')

fig.math('relative-likelihood',
         r'r_\theta(x)=\ell_\theta(x)-\ell_{\mathrm{ref}}(x)',
         4, 164, 8)
# v3 (, round5 A10): l_theta is a token SUM in Section 4, but the bound uses the audit's
# token-MEAN placebo drop, so d_O^theta is named in words instead of written as l_theta(x)-l_theta(x_O).
_dw=fig.math('placebo-drop-definition',r'd_O^\theta', 86, 164, 8)
fig.text('placebo-drop-meaning',': token-mean placebo drop', 86+_dw, 164, 8)
# Only one table-to-loss arrow: every loss reads its highlighted subset of the
# whole table, not merely the aligned row.
fig.arrow('table-to-losses', [(181, 107), (187.5, 107)], head=2.5)

losses = [
    ('evidence-margin', 'evidence margin', 42, [0,1], C['edge'],
     r'\operatorname{softplus}(-\beta[r_\theta(x)-r_\theta(x_E)])'),
    ('placebo-contrast', 'placebo contrast', 88, [1,2], C['ours'],
     r'\operatorname{softplus}(-\beta[r_\theta(x_O)-r_\theta(x_E)])'),
    ('placebo-bound', 'placebo bound', 127, [0,2], C['ours'],
     r'\operatorname{relu}(d_O^\theta-1.25d_O^{\mathrm{ref}}-0.10)'),
]
grid_records=[]
for key, label, y, active, border, expr in losses:
    fig.rect(key+'-box',190,y,116,38 if key=='evidence-margin' else 31,fill='#FFFFFF',stroke=border,
             lw=.75 if border==C['ours'] else .5,r=2)
    # v2 : the dependency mini-grids were dropped; the formulas name their inputs.
    fig.text(key+'-title',label,196,y+11,8,bold=True)
    fig.math(key+'-formula',expr,196,y+24,8)
    if key=='evidence-margin':
        fig.text('evidence-margin-readout','alone, it learns an edit detector',196,y+34,7,color=C['gray'])
    grid_records.append(dict(loss=key,active_rows=active))

# The three terms are added into L_op, then weighted across operators.
fig.line('loss-sum-upper',[(306,61),(310,61),(310,103.5)])
fig.line('loss-sum-lower',[(306,142.5),(310,142.5),(310,103.5)])
fig.arrow('loss-sum-output',[(306,103.5),(315,103.5)],head=2.5)
fig.rect('total-box',315,61,77,91,fill='#FFFFFF',stroke=C['edge'],r=2)
# v2: one two-line equation, the second line aligned under the first term; the clean-supervision label sits under
# the term it names, and the operator sum is written inline so its limit does not drop below the line.
lw_=fig.math('total-lhs',r'\mathcal{L}=',319,92,8)
fig.math('total-clean',r'\alpha\,\mathrm{NLL}_\theta(y^{\!\star}\mid x,q)',319+lw_,92,8)
fig.text('clean-supervision','clean supervision',319+lw_,102,6.8,color=C['gray'])
fig.math('total-operator-sum',r'+\,\Sigma_{\mathrm{op}}\,w_{\mathrm{op}}\mathcal{L}_{\mathrm{op}}',319+lw_-6.2,122,8)
fig.text('operator-weights','over edit operators',319+lw_,134,6.8,color=C['gray'])

pdf=fig.save(HERE,'fig2_objective')
qa=HERE/'qa';qa.mkdir(exist_ok=True)
import pymupdf as _fz
_fz.open(pdf)[0].get_pixmap(dpi=240).save(HERE/'preview.png')

def intersection(a,b):
    return min(a[2],b[2])-max(a[0],b[0]), min(a[3],b[3])-max(a[1],b[1])
text_records=[r for r in fig.records if r['kind'] in ('text','math')]
overlaps=[]
for a,b in itertools.combinations(text_records,2):
    dx,dy=intersection(a['bbox'],b['bbox'])
    if dx>.15 and dy>.15:overlaps.append([a['id'],b['id'],dx,dy])
outside=[r['id'] for r in fig.records if 'bbox' in r and
         (r['bbox'][0]<0 or r['bbox'][1]<0 or r['bbox'][2]>396 or r['bbox'][3]>168)]
images=[r for r in fig.records if r['kind']=='image']
image_map={r['id']:r for r in images}
same=[]
for key,_,_,_,_,masked,_ in rows[1:]:
    for i in range(5):
        if i not in masked:
            a=image_map[f'clean-strip-frame-{i}'];b=image_map[f'{key}-strip-frame-{i}']
            ok=a['payload_sha256']==b['payload_sha256'] and a['pixel_size']==b['pixel_size']
            same.append(dict(row=key,slot=i,identical=ok))
            assert ok
record=dict(width_pt=396,height_pt=168,main_font_range_pt=[7,8.5],math_body_pt=8,
            math_subscript_expected_pt=5.6,ordinary_text_objects=len(text_records),
            text_overlaps=overlaps,outside_canvas=outside,
            unchanged_frames=same,slots=slot_records,dependencies=grid_records,
            formulas_source=str(PAPER/'sections/objective.tex'),
            formulas_source_sha256=sha(PAPER/'sections/objective.tex'),
            correct_current_objective=dict(operator_loss='L_op',gamma=False,hinge='relu',
              total='alpha NLL_theta(y^star | x,q) + sum_op w_op L_op'),
            pdf_sha256=sha(pdf),editable_svg_sha256=sha(HERE/'editable.svg'))
(qa/'maker_checks.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
manifest=dict(dataset=M['dataset'],split=M['split'],video_id=M['video_id'],
              illustrative_only=True,not_a_training_example=True,
              query=M['event_A'],annotated_interval_sec=M['interval_A'],
              shared_manifest_sha256=sha(ROOT/'frame_manifest.json'),
              times_sec=TIMES,evidence_slots_0based=E,placebo_slots_0based=O,
              operator_shown='mask',frames=[f for f in M['frames'] if f['timestamp_sec'] in TIMES])
(HERE/'frame_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(json.dumps(dict(pdf=str(pdf),overlaps=overlaps,outside=outside),indent=2))
assert not overlaps and not outside
