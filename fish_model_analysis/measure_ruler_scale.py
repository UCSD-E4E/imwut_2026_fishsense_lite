"""Re-measure the Wildco board's clicked span against its OWN printed scale.

Produces the 341 mm in `calibration.MEASURED_REFERENCES_M` and the table in
FINDINGS section 7b. Reads the JPEG siblings of the raws on the NAS, so it needs
the share mounted; the path below is a macOS mount point, `~/mnt/fishsense_data`
on the Linux box.

    python measure_ruler_scale.py 0.90      # the nine near-range frames

Method, and why it is not circular: each frame is straightened along the line
between the two clicked points, the printed inch ticks either side of it are
located by height, a 15-tooth comb is fitted to find the inch pitch, and the
clicked span is read off in INCHES. Being a ratio of pixels to pixels inside one
frame it cancels range, focal length and the laser calibration outright, so no
part of the instrument under test enters the answer.

Only the near frames are usable: at 1.8 m the ticks are too fine to separate.
"""

import sys, time, numpy as np, pandas as pd
from PIL import Image
import os
BASE=os.environ.get("REEF_ROOT",
    "/Volumes/nas/fishsense_data/REEF/data/2024.06.20.REEF/08_2023/"
    "082923_Pool Calibration")

def strip_of(jpg, head, tail):
    im=Image.open(jpg).convert("L"); a=np.asarray(im,float); H,W=a.shape
    s=W/4014.0; h=np.array(head)*s; t=np.array(tail)*s
    u=t-h; L=float(np.hypot(*u)); u/=L; n=np.array([-u[1],u[0]])
    xs=np.arange(-60,L+60,0.5); ys=np.arange(-95,35,0.5)
    P=h[None,None,:]+xs[None,:,None]*u+ys[:,None,None]*n
    X=np.clip(P[...,0],0,W-1).astype(int); Y=np.clip(P[...,1],0,H-1).astype(int)
    return a[Y,X], 120.0, 120.0+L/0.5

def ticks_of(strip):
    bg=np.percentile(strip,90); dark=np.clip(bg-strip,0,None)
    band=dark[130:190,:].sum(0); prof=np.clip(band-np.percentile(band,25),0,None)
    cand=[i for i in range(2,len(prof)-2)
          if prof[i]>=prof[i-1] and prof[i]>=prof[i+1] and prof[i]>0.25*prof.max()]
    grp=[]
    for i in cand:
        if grp and i-grp[-1][-1]<=3: grp[-1].append(i)
        else: grp.append([i])
    cols=np.array([int(np.mean(g)) for g in grp])
    hs=[]
    for c in cols:
        cm=dark[100:200,max(0,c-1):c+2].mean(1); on=np.where(cm>0.35*cm.max())[0]
        hs.append(on.max()-on.min() if len(on) else 0)
    return cols, np.array(hs,float), len(prof)

def fit(jpg, head, tail):
    strip,hcol,tcol=strip_of(jpg,head,tail)
    cols,hs,C=ticks_of(strip)
    if len(cols)<20: return None
    w=np.clip(hs-np.percentile(hs,55),0,None)
    W=np.zeros(C)
    for c,v in zip(cols,w): W[max(0,c-1):c+2]=np.maximum(W[max(0,c-1):c+2],v)
    span=cols.max()-cols.min(); K=np.arange(15); base=np.arange(C)[:,None]
    best=None
    for p in np.arange(span/14.6, span/12.4, 0.08):
        idx=np.round(base+K[None,:]*p).astype(int)
        S=np.where(idx<C, W[np.clip(idx,0,C-1)],0.0).sum(1)*(idx[:,-1]<C)
        j=int(S.argmax())
        if best is None or S[j]>best[0]: best=(S[j],p,float(j))
    _,p,x0=best
    k=[];c=[]
    for i in range(15):
        d=np.abs(cols-(x0+i*p)); j=int(d.argmin())
        if d[j]<p*0.10 and hs[j]>np.percentile(hs,70): k.append(i); c.append(cols[j])
    if len(k)<8: return None
    k=np.array(k,float); c=np.array(c,float)
    q=np.polyfit(k,c,2); rms=float((c-np.polyval(q,k)).std())
    def to_in(col):
        r=[x.real for x in np.roots([q[0],q[1],q[2]-col]) if abs(x.imag)<1e-9]
        return min(r,key=lambda v:abs(v-7.0))
    return to_in(tcol)-to_in(hcol), rms, len(k), to_in(hcol), to_in(tcol)

r=pd.read_csv("fish_model_analysis/data/ruler.csv")
r=r[pd.to_numeric(r.dive_id,errors="coerce").notna()].copy()
ll=pd.read_csv("laser_labeling_analysis/laser_labels_cleaned.csv")
m=ll.drop_duplicates("image_id").set_index("image_id").image_path
r["path"]=pd.to_numeric(r.image_id).map(m); r["depth"]=pd.to_numeric(r.depth_m)
sel=r[r.path.notna() & (r.depth < float(sys.argv[1]))].sort_values("depth")
print(f"{len(sel)} frames under {sys.argv[1]} m", flush=True)
res=[]
for _,row in sel.iterrows():
    t0=time.time()
    jpg=f"{BASE}/{row.path.split('/')[-2]}/{row.path.split('/')[-1][:-4]}.JPG"
    ht=[float(v) for v in row.ht.split(";")]
    out=fit(jpg,(ht[0],ht[1]),(ht[2],ht[3]))
    nm=row.path.split('/')[-1]
    if out is None:
        print(f"{nm:<14} {row.depth:>5.2f}  comb failed  ({time.time()-t0:.1f}s)", flush=True); continue
    sp,rms,n,hi,ti=out; res.append(sp)
    print(f"{nm:<14} {row.depth:>5.2f} m  head {hi:6.3f}  tail {ti:6.3f}  "
          f"span {sp:6.3f} in = {sp*25.4:6.1f} mm  (rms {rms:.2f}px, {n} ticks, {time.time()-t0:.1f}s)",
          flush=True)
a=np.array(res)
if len(a):
    print(f"\nn={len(a)}  median {np.median(a)*25.4:.1f} mm  mean {a.mean()*25.4:.1f}  "
          f"sd {a.std(ddof=1)*25.4:.1f} mm", flush=True)
    print(f"file 342.9 mm -> {100*(342.9-np.median(a)*25.4)/(np.median(a)*25.4):+.2f} % long; "
          f"repo comment 341.8 +- 0.3 mm", flush=True)
