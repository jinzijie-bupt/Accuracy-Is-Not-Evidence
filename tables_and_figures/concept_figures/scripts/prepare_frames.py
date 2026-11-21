from pathlib import Path
import sys,json,hashlib
import cv2
from PIL import Image
from diagram_common import ROOT,PAPER,sha

video=PAPER/'assets/source_videos/K7NRW.mp4'
old=json.loads((PAPER/'assets/figure_frames/K7NRW/manifest.json').read_text(encoding='utf-8'))
assert sha(video)==old['source_video_sha256']
times=[.8,4,7.2,11,15,16.5,18,20.5,23.9,25,27,28]
cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS);frames=[]
for t in times:
    i=round(t*fps);cap.set(cv2.CAP_PROP_POS_FRAMES,i);ok,bgr=cap.read();assert ok
    im=Image.fromarray(cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB))
    path=ROOT/'assets'/f'K7NRW_t{t:06.2f}.jpg'
    im.save(path,quality=95,subsampling=0)
    frames.append(dict(timestamp_sec=t,frame_index=i,size=im.size,file=path.name,sha256=sha(path)))
cap.release()
record=dict(dataset='Charades-STA',split='test',video_id='K7NRW',video_sha256=sha(video),fps=fps,
            event_A=old['query'],interval_A=old['annotated_interval_sec'],event_B=old['future_crossed_query'],
            annotation_source=old['annotation_source'],annotation_sha256=old['annotation_source_sha256'],frames=frames,
            sampling='Illustrative chronological samples, not an experimental evaluation batch.',
            teaser_times=[.8,7.2,11,15,18,20.5,23.9,25,28],teaser_E=[3,4,5],teaser_O=[0,7,8],
            audit_objective_times=[.8,15,18,25,28],audit_objective_E=[1,2],audit_objective_O=[0,4])
(ROOT/'frame_manifest.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print('Prepared',len(frames),'authentic frames.')
