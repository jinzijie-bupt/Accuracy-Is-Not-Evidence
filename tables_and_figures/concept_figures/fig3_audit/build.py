"""Fixed-answer audit diagram. Coordinates and font sizes are physical points."""
from pathlib import Path
import hashlib
import json
import shutil
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'scripts'))
from diagram_common import Fig, C, ROOT, PAPER, frame, sha, PARSER, fontfile, register, FONTS
from matplotlib.font_manager import FontProperties


def plain_width(s, size=8, bold=False):
    return FONTS[register(fontfile(bold))][1].text_length(s, fontsize=size)


def math_width(s, size=8):
    return float(PARSER.parse('$' + s + '$', dpi=72,
        prop=FontProperties(family='STIXGeneral', size=size)).width)


def inline(fig, key, runs, center, baseline, size=8, color=C['ink']):
    """Place text/math as complete baseline-aligned runs, without floating glyphs."""
    widths = [math_width(s, size) if k == 'm' else plain_width(s, size) for k, s in runs]
    x = center - sum(widths) / 2
    for i, ((kind, s), width) in enumerate(zip(runs, widths)):
        if kind == 'm':
            fig.math(f'{key}-{i}', s, x, baseline, size=size, color=color)
        else:
            fig.text(f'{key}-{i}', s, x, baseline, size=size, color=color)
        x += width


def build():
    fig = Fig(396, 160, 'Fixed-answer audit and trained-versus-base comparison')
    times = [.8, 15, 18, 25, 28]
    columns = [(5, 44, 'clean', r'x', C['ink'], [], []),
               (95, 134, 'evidence', r'x_E', C['evidence'], [1, 2], []),
               (185, 224, 'placebo', r'x_O', C['placebo'], [], [0, 4])]

    # One frozen answer/query conditions every scoring branch.
    fig.rect('fixed-answer-box', 5, 3, 258, 17, fill=C['pale'], r=2)
    inline(fig, 'fixed-conditioning', [('t', 'fixed answer '), ('m', r'y^{\!\star}'),
           ('t', '  ·  fixed query '), ('m', r'q'),
           ('t', '  ·  never regenerated')], 134, 14.6, size=8)
    fig.line('fixed-condition-spine', [(134, 20), (134, 24), (44, 24), (224, 24)], lw=.65)
    for x, cx, name, symbol, color, e, o in columns:
        fig.arrow(f'condition-to-{name}', [(cx, 24), (cx, 29)], lw=.65, head=2.3)
        inline(fig, f'{name}-heading', [('t', name+' '), ('m', symbol)], cx, 37.7,
               size=8, color=color)
        fig.filmstrip(f'{name}-video', times, x, 42, 78, 17,
                      evidence=e, placebo=o, masked=e+o)
        fig.arrow(f'{name}-to-model', [(cx, 59), (cx, 67)], lw=.65, head=2.5)
        fig.rect(f'{name}-model-box', x, 68, 78, 15, fill='#FFFFFF', r=2)
        fig.text(f'{name}-frozen-model', 'model (frozen)', cx, 78.6, size=8, anchor='center')
        fig.arrow(f'{name}-to-likelihood', [(cx, 83), (cx, 92)], lw=.65, head=2.5)

    # A shared band exposes all three likelihoods to both drop calculations.
    # This prevents the misleading source diagram's omission of the clean score.
    fig.rect('likelihood-band', 5, 92, 258, 16, fill=C['pale'], r=2)
    for split in (89, 179):
        fig.line(f'likelihood-separator-{split}', [(split, 95), (split, 105)], color='#CDD2D8', lw=.5)
    for _, cx, name, symbol, color, _, _ in columns:
        fig.math(f'{name}-likelihood', rf'\ell({symbol})', cx, 103.2, size=8, color=color, anchor='center')

    for name, x, cx, symbol, fill, color in [
        ('evidence', 5, 66.5, 'E', C['evidence_fill'], C['evidence']),
        ('placebo', 140, 201.5, 'O', C['placebo_fill'], C['placebo'])]:
        fig.arrow(f'scores-to-{name}-drop', [(cx, 108), (cx, 118)], color=color, lw=.65, head=2.5)
        fig.rect(f'{name}-drop-box', x, 119, 123, 16, fill=fill, stroke=color, r=2)
        fig.math(f'{name}-drop', rf'd_{symbol}=\ell(x)-\ell(x_{symbol})',
                 cx, 130.3, size=8, color=C['ink'], anchor='center')
        fig.line(f'{name}-drop-to-contrast', [(cx, 135), (cx, 140), (134, 140)], lw=.65)
    fig.arrow('drops-to-contrast', [(134, 140), (134, 145)], lw=.65, head=2.5)
    fig.rect('contrast-box', 67, 145, 134, 13, fill='#FFFFFF', r=2)
    fig.math('within-model-contrast', r'\psi=d_E-d_O', 134, 154.6, size=8.2, anchor='center')

    # The right region is repeated-scoring aggregation, with a separate raw d_O output.
    fig.line('comparison-divider', [(272.5, 3), (272.5, 158)], color='#B2B2B2', lw=.5, dash=[3, 3])
    fig.text('trained-vs-base', 'trained vs. base', 336.5, 10.5, size=8.5, bold=True, anchor='center')
    for name, x, col in [('trained', 280, C['ours']), ('base', 340, C['base'])]:
        fig.rect(f'comparison-{name}-box', x, 21, 53, 21, fill='#FFFFFF', r=2)
        fig.rect(f'comparison-{name}-stripe', x, 21, 2, 21, fill=col, stroke=None)
        fig.text(f'comparison-{name}-label', name, x+27, 30.7, size=8, anchor='center')
        fig.text(f'comparison-{name}-model', 'model', x+27, 39.2, size=7.5, anchor='center')
        fig.line(f'{name}-to-common-set', [(x+26.5, 42), (x+26.5, 47), (336.5, 47)], lw=.65)
    fig.arrow('common-set-entry', [(336.5, 47), (336.5, 50)], lw=.65, head=2)
    fig.rect('common-correct-set', 280, 50, 113, 23, fill=C['pale'], r=2)
    fig.math('common-correct-condition', r'H_\cap:\ \mathrm{IoU}\geq0.5', 336.5, 60.2, size=8, anchor='center')
    fig.text('common-correct-description', 'both correct on clean video', 336.5, 70.3, size=7.2, anchor='center')
    fig.arrow('common-set-to-gamma', [(336.5, 73), (336.5, 80)], lw=.65, head=2.5)
    fig.rect('gamma-box', 279, 80, 115, 23, fill='#FFFFFF', r=2)
    # Preserve the full one-line formula without tiny nested (3.92 pt) glyphs.
    # Display fraction and independently laid-out summation limits keep every
    # normal symbol at 8 pt and every single-level subscript at 5.6 pt.
    gamma_expr = r'\Gamma=\frac{1}{|H_\cap|}\sum_{i\in H_\cap}(\psi_i^{\mathrm{trained}}-\psi_i^{\mathrm{base}})'
    gamma_group = fig.group('gamma-definition')
    gamma_group.set('aria-label', gamma_expr)
    fig.math('gamma-normalizer', r'\Gamma=\dfrac{1}{|H_\cap|}', 282.5, 92, size=8)
    fig.math('gamma-sum', r'\sum', 326, 90, size=8, anchor='center')
    fig.math('gamma-sum-range', r'i\in H_\cap', 326, 100, size=8, anchor='center')
    fig.math('gamma-paired-difference', r'(\psi_i^{\mathrm{trained}}-\psi_i^{\mathrm{base}})', 339.5, 92, size=8)
    for key in ['gamma-normalizer', 'gamma-sum', 'gamma-sum-range', 'gamma-paired-difference']:
        el = next(e for e in fig.root if e.get('id') == key)
        fig.root.remove(el)
        gamma_group.append(el)
    fig.records.append(dict(id='gamma-definition', kind='math', text=gamma_expr,
                            size=8, bbox=[282.5, 80, 390.5, 102]))
    fig.arrow('trained-own-placebo-output', [(280, 31.5), (276.5, 31.5), (276.5, 118), (280, 118)],
              color=C['placebo'], lw=.65, head=2.2)
    fig.rect('trained-placebo-box', 280, 108, 113, 22, fill=C['placebo_fill'], stroke=C['placebo'], r=2)
    fig.text('trained-placebo-owner', "Trained model's", 336.5, 117.0, size=7.5, anchor='center')
    inline(fig, 'trained-placebo-readout', [('t', 'placebo drop '), ('m', r'd_O')],
           336.5, 126.2, size=8)

    # Operator legend uses honest photographic before/after samples.
    donor_shared = ROOT/'assets'/'P6JGZ_t005.73.png'
    donor_existing = PAPER/'assets'/'figure_frames'/'P6JGZ'/'fig2_swap_right.png'
    donor_hash = '8c66ea8a496820991205eab88c1eb195afe5c9aed2471de01fa71513a3e7ab69'
    if sha(donor_existing) != donor_hash:
        raise RuntimeError('Donor crop differs from verified source manifest')
    if not donor_shared.exists():
        shutil.copy2(donor_existing, donor_shared)
    if sha(donor_shared) != donor_hash:
        raise RuntimeError('Packaged donor crop differs from verified source')
    donor = donor_shared
    for name, left, right in [('mask', 282, 304), ('swap', 348, 370)]:
        fig.photo(f'{name}-before', frame(15), left, 137, 13, 13)
        fig.rect(f'{name}-before-border', left, 137, 13, 13, stroke=C['edge'], lw=.4)
        fig.arrow(f'{name}-operator', [(left+15, 143.5), (right-1.5, 143.5)], lw=.5, head=2)
        if name == 'mask':
            fig.masked_cell('mask-after', right, 137, 13, 13)
            fig.rect('mask-after-border', right, 137, 13, 13, stroke=C['edge'], lw=.4)
        else:
            fig.photo('swap-after', donor, right, 137, 13, 13)
            fig.rect('swap-after-border', right, 137, 13, 13, stroke=C['edge'], lw=.4)
        fig.text(f'{name}-operator-label', name, (left+right+13)/2, 158, size=7.5, anchor='center')

    pdf = fig.save(HERE, 'fig3_audit')
    shared = json.loads((ROOT/'frame_manifest.json').read_text(encoding='utf-8'))
    manifest = dict(figure='fig3_audit', dataset=shared['dataset'], split=shared['split'],
        video_id=shared['video_id'], video_sha256=shared['video_sha256'],
        query=shared['event_A'], answer_interval_sec=shared['interval_A'],
        sample_times_sec=times, evidence_slots_0based=[1,2], placebo_slots_0based=[0,4],
        equal_edit_count=2, conditioning='One fixed query and answer string; no regeneration.',
        sampling='Chronological illustration from a held-out test clip, not an evaluation batch.',
        source_frames=[f for t in times for f in shared['frames'] if f['timestamp_sec']==t],
        images=[r for r in fig.records if r['kind']=='image'],
        swap_donor=dict(video_id='P6JGZ', timestamp_sec=172/30, frame_index_0based=172,
            source_video_sha256='64ec5956be5104f45b57a0d327f8e39b52d658c24f8ad029d9cf6c724c96c1d7',
            original_crop_file=str(donor_existing), original_crop_sha256=donor_hash,
            displayed_asset=str(donor), displayed_asset_sha256=sha(donor)),
        pdf_sha256=sha(pdf), svg_sha256=sha(HERE/'editable.svg'))
    (HERE/'frame_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(str(pdf))


if __name__ == '__main__':
    build()
