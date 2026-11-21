"""Shared physical-point primitives: live-text SVG and embedded-font PDF.

All coordinates are points, origin top left. Text y is its baseline.
"""
from pathlib import Path
import base64
import io
import json
import math
import shutil
import hashlib
import xml.etree.ElementTree as ET
import matplotlib as mpl
from matplotlib.mathtext import MathTextParser
from matplotlib.font_manager import FontProperties
import pymupdf as fitz
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT.parent
NS='http://www.w3.org/2000/svg'
XL='http://www.w3.org/1999/xlink'
ET.register_namespace('',NS);ET.register_namespace('xlink',XL)
FONT_DIR=Path(mpl.get_data_path())/'fonts/ttf'
C=dict(evidence='#D55E00',placebo='#0072B2',partner='#882255',ours='#009E73',
       ink='#1A1A1A',gray='#7A7A7A',edge='#9A9A9A',base='#BDBDBD',line='#4D4D4D',
       pale='#F4F5F7',evidence_fill='#FBE9DF',placebo_fill='#E3F0F7',partner_fill='#EFE1E8')
mpl.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix'})
PARSER=MathTextParser('path')
FONTS={}

def register(path):
    path=Path(path);key='fig_'+path.stem
    if key not in FONTS:
        FONTS[key]=(path,fitz.Font(fontfile=str(path)))
    return key

def fontfile(bold=False,italic=False):
    return FONT_DIR/('STIXGeneral'+('BolIta' if bold and italic else 'Bol' if bold else 'Italic' if italic else '')+'.ttf')

def qtag(t):return '{'+NS+'}'+t
def rgb(c):return tuple(int(c.lstrip('#')[i:i+2],16)/255 for i in (0,2,4)) if c else None
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def frame(t):
    return ROOT/'assets'/f'K7NRW_t{float(t):06.2f}.jpg'

class Fig:
    def __init__(self,width,height,title=''):
        self.w,self.h=width,height
        self.root=ET.Element(qtag('svg'),{'version':'1.1','width':f'{width}pt','height':f'{height}pt','viewBox':f'0 0 {width} {height}'})
        ET.SubElement(self.root,qtag('title')).text=title
        self.doc=fitz.open();self.pdf=self.doc.new_page(width=width,height=height)
        self.doc.set_metadata({'title':title,'author':'Zijie Jin et al.'})
        self.records=[];self.count=0
        self.rect('background',0,0,width,height,fill='#FFFFFF',stroke=None)
    def group(self,key):return ET.SubElement(self.root,qtag('g'),{'id':key})
    def el(self,kind,key,attrs,parent=None):
        return ET.SubElement(parent if parent is not None else self.root,qtag(kind),dict(id=key,**{k:str(v) for k,v in attrs.items()}))
    def pdftext(self,text,fn,size,x,y,color):
        self.pdf.insert_font(fontname=fn,fontfile=str(FONTS[fn][0]))
        self.pdf.insert_text((x,y),text,fontname=fn,fontsize=size,color=rgb(color))
    def rect(self,key,x,y,w,h,fill=None,stroke=C['edge'],lw=.5,r=0,dash=None,parent=None):
        a={'x':x,'y':y,'width':w,'height':h,'fill':fill or 'none','stroke':stroke or 'none','stroke-width':lw}
        if r:a.update(rx=r,ry=r)
        if dash:a['stroke-dasharray']=' '.join(map(str,dash))
        self.el('rect',key,a,parent)
        self.pdf.draw_rect(fitz.Rect(x,y,x+w,y+h),color=rgb(stroke),fill=rgb(fill),width=lw,dashes=('['+' '.join(map(str,dash))+'] 0') if dash else None,radius=min(.5,r/min(w,h)) if r else None)
        self.records.append(dict(id=key,kind='rect',bbox=[x,y,x+w,y+h],fill=fill,stroke=stroke))
    def line(self,key,pts,color=C['line'],lw=.75,dash=None,parent=None):
        self.el('polyline',key,{'points':' '.join(f'{x},{y}' for x,y in pts),'fill':'none','stroke':color,'stroke-width':lw,**({'stroke-dasharray':' '.join(map(str,dash))} if dash else {})},parent)
        self.pdf.draw_polyline(pts,color=rgb(color),width=lw,dashes=('['+' '.join(map(str,dash))+'] 0') if dash else None)
        self.records.append(dict(id=key,kind='line',points=pts,lw=lw))
    def polygon(self,key,pts,fill=C['line'],stroke=None,lw=.5,parent=None):
        self.el('polygon',key,{'points':' '.join(f'{x},{y}' for x,y in pts),'fill':fill or 'none','stroke':stroke or 'none','stroke-width':lw},parent)
        self.pdf.draw_polyline(pts,color=rgb(stroke),fill=rgb(fill),width=lw,closePath=True)
    def arrow(self,key,pts,color=C['line'],lw=.75,head=3):
        g=self.group(key);self.line(key+'-shaft',pts,color,lw,parent=g)
        a,b=pts[-2:];dx=b[0]-a[0];dy=b[1]-a[1];l=math.hypot(dx,dy);ux,uy=dx/l,dy/l
        self.polygon(key+'-head',[b,(b[0]-head*ux+head*.48*uy,b[1]-head*uy-head*.48*ux),(b[0]-head*ux-head*.48*uy,b[1]-head*uy+head*.48*ux)],fill=color,parent=g)
    def text(self,key,label,x,y,size=8,bold=False,italic=False,color=C['ink'],anchor='start',parent=None):
        ff=fontfile(bold,italic);fn=register(ff);fo=FONTS[fn][1];w=fo.text_length(label,fontsize=size)
        if anchor in ['center','middle']:x-=w/2
        elif anchor in ['end','right']:x-=w
        attrs={'x':x,'y':y,'font-family':'STIXGeneral, Times New Roman, serif','font-size':size,'fill':color,'font-weight':'bold' if bold else 'normal','font-style':'italic' if italic else 'normal'}
        self.el('text',key,attrs,parent).text=label
        self.pdftext(label,fn,size,x,y,color)
        asc,desc=fo.ascender*size,fo.descender*size
        self.records.append(dict(id=key,kind='text',text=label,size=size,bbox=[x,y-asc,x+w,y-desc]))
        return w
    def math(self,key,expr,x,y,size=8,color=C['ink'],anchor='start'):
        """MathText lays out glyphs; both exports retain real embedded/live glyphs."""
        expr=expr.strip('$');p=PARSER.parse('$'+expr+'$',dpi=72,prop=FontProperties(family='STIXGeneral',size=size))
        w,h,depth=float(p.width),float(p.height),float(p.depth)
        if anchor in ['center','middle']:x-=w/2
        elif anchor in ['end','right']:x-=w
        g=self.group(key)
        tx=self.el('text',key+'-text',{'fill':color,'aria-label':expr},g)
        for j,glyph in enumerate(p.glyphs):
            ft,fs,num=glyph[:3];gx,gy=glyph[-2:]
            fn=register(ft.fname);style=ft.style_name.lower();fam=ft.family_name
            attrs={'x':x+gx,'y':y-gy,'font-family':fam+', serif','font-size':fs,'font-weight':'bold' if 'bold' in style else 'normal','font-style':'italic' if 'italic' in style or 'oblique' in style else 'normal'}
            self.el('tspan',key+f'-glyph-{j}',attrs,tx).text=chr(num)
            self.pdftext(chr(num),fn,fs,x+gx,y-gy,color)
        for j,(rx,ry,rw,rh) in enumerate(p.rects):self.rect(key+f'-rule-{j}',x+rx,y-ry-rh,rw,rh,fill=color,stroke=None,parent=g)
        self.records.append(dict(id=key,kind='math',text=expr,size=size,bbox=[x,y-(h-depth),x+w,y+depth]))
        return w
    def photo(self,key,path,x,y,w,h,parent=None):
        """Embed an authentic frame, centre-cropped to the cell without distortion."""
        im=Image.open(path).convert('RGB');iw,ih=im.size;ar=w/h
        if iw/ih>ar:
            cw=ih*ar;box=((iw-cw)/2,0,(iw+cw)/2,ih)
        else:
            ch=iw/ar;box=(0,(ih-ch)/2,iw,(ih+ch)/2)
        im=im.crop(tuple(round(v) for v in box))
        target=(max(1,round(w/72*500)),max(1,round(h/72*500)))
        if im.width>target[0] or im.height>target[1]:im=im.resize(target,Image.Resampling.LANCZOS)
        b=io.BytesIO();im.save(b,format='JPEG',quality=93,subsampling=0);data=b.getvalue()
        self.el('image',key,{'x':x,'y':y,'width':w,'height':h,'preserveAspectRatio':'none','{'+XL+'}href':'data:image/jpeg;base64,'+base64.b64encode(data).decode()},parent)
        self.pdf.insert_image(fitz.Rect(x,y,x+w,y+h),stream=data)
        self.records.append(dict(id=key,kind='image',path=str(path),bbox=[x,y,x+w,y+h],pixel_size=im.size,payload_sha256=hashlib.sha256(data).hexdigest()))
    def masked_cell(self,key,x,y,w,h,parent=None):
        """A masked frame: black, crossed in grey so it cannot be read as a dark video frame (v2, 2026-09-25)."""
        self.rect(key,x,y,w,h,fill='#000000',stroke=None,parent=parent)
        i=min(w,h)*.12
        self.line(key+'-x1',[(x+i,y+i),(x+w-i,y+h-i)],color='#9A9A9A',lw=.45,parent=parent)
        self.line(key+'-x2',[(x+i,y+h-i),(x+w-i,y+i)],color='#9A9A9A',lw=.45,parent=parent)
    def filmstrip(self,key,times,x,y,w,h,evidence=(),placebo=(),masked=(),holes=True,border=True):
        g=self.group(key);n=len(times);gap=.8;rim=2 if holes else .7
        self.rect(key+'-body',x,y,w,h,fill=C['ink'] if holes else None,stroke=None,parent=g)
        cw=(w-2*gap-(n-1)*gap)/n;ch=h-2*rim
        boxes=[]
        for i,t in enumerate(times):
            xx=x+gap+i*(cw+gap);yy=y+rim
            if i in masked:self.masked_cell(key+f'-mask-{i}',xx,yy,cw,ch,parent=g)
            else:self.photo(key+f'-frame-{i}',frame(t),xx,yy,cw,ch,parent=g)
            if border and i in evidence:self.rect(key+f'-E-{i}',xx,yy,cw,ch,stroke=C['evidence'],lw=1.2,parent=g)
            if border and i in placebo:self.rect(key+f'-O-{i}',xx,yy,cw,ch,stroke=C['placebo'],lw=1.2,dash=[3,2],parent=g)
            boxes.append([xx,yy,cw,ch])
        if holes:
            step=3.4;ss=.9
            for j in range(int((w-2)/step)):
                xx=x+1+j*step
                self.rect(key+f'-hole-u-{j}',xx,y+.5,ss,ss,fill='#FFFFFF',stroke=None,parent=g)
                self.rect(key+f'-hole-l-{j}',xx,y+h-1.4,ss,ss,fill='#FFFFFF',stroke=None,parent=g)
        return boxes
    def save(self,folder,stem='figure'):
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        ET.indent(self.root);ET.ElementTree(self.root).write(folder/'editable.svg',encoding='utf-8',xml_declaration=True)
        self.doc.subset_fonts();self.doc.save(folder/(stem+'.pdf'),garbage=4,deflate=True)
        (folder/'geometry.json').write_text(json.dumps({'width_pt':self.w,'height_pt':self.h,'objects':self.records},indent=2),encoding='utf-8')
        fd=ROOT/'fonts';fd.mkdir(exist_ok=True)
        for path,_ in FONTS.values():shutil.copy2(path,fd/path.name)
        return folder/(stem+'.pdf')
