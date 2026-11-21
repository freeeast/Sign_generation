"""Index official NSA ZIP members and preprocess a bounded, source-disjoint sample."""
import argparse
from collections import Counter, defaultdict
import csv
import io
import json
import mmap
from pathlib import Path
import re
import struct
import zlib

import numpy as np
from scipy.spatial.transform import Rotation
from src.nsa_motion import ArrayUnpickler,FIELDS

LOCAL=struct.Struct("<IHHHHHIIIHH")
CENTRAL=struct.Struct("<4s6H3I5H2I")
NAME=re.compile(rb"poses/(?P<clip>.{11}_[^/]+-rgb_front)/[^/]+_(?P<frame>\d+)_3D\.pkl$")
ACTIVE=np.r_[[3,6,9,12,13,14,15,16,17,18,19,20,21],25:55]
KEYS=("global_orient","body_pose","jaw_pose","leye_pose","reye_pose","left_hand_pose","right_hand_pose")
WIDTHS=(3,63,3,3,3,45,45)
HELD={"3sZjSNcqXpI","33jxeIIbBnM","ETOZLBScxWY","DKOzSnHwSFI","bcNtUYh3GNs","33YRZ5UZTA4"}
SPLITS=("train","val","test")


def start_at(raw,split):
    i=raw.find((split+"_poses.zip").encode())
    if i<LOCAL.size or raw[i-LOCAL.size:i-LOCAL.size+4]!=b"PK\x03\x04":
        raise ValueError(f"{split} outer member not found")
    _,_,_,method,*_other,n,x=LOCAL.unpack_from(raw,i-LOCAL.size)
    if method:
        raise ValueError("Expected stored inner ZIP")
    return i+n+x


def locals_by_clip(raw,offset):
    clips=defaultdict(list)
    while offset+LOCAL.size<=len(raw) and raw[offset:offset+4]==b"PK\x03\x04":
        _,_,flags,method,_,_,crc,packed,size,n,x=LOCAL.unpack_from(raw,offset)
        name=raw[offset+LOCAL.size:offset+LOCAL.size+n]
        start=offset+LOCAL.size+n+x
        offset=start+packed
        if offset>len(raw):
            break
        if flags&8:
            raise ValueError(f"Unexpected streaming descriptor {name!r}")
        m=NAME.fullmatch(name)
        if m:
            clip=m["clip"].decode()
            clips[clip].append((int(m["frame"]),start,packed,size,crc,method))
    return clips


def index_central(raw,start,end):
    clips={}
    other=Counter()
    while start+CENTRAL.size<=end and raw[start:start+4]==b"PK\x01\x02":
        h=CENTRAL.unpack_from(raw,start)
        n,x,c=h[10:13]
        pos=start+CENTRAL.size
        name=raw[pos:pos+n]
        start=pos+n+x+c
        m=NAME.fullmatch(name)
        if m:
            clip=m["clip"].decode()
            t=int(m["frame"])
            if clip not in clips:
                clips[clip]=[0,0,0,0]
            v=clips[clip]
            v[0]+=1;v[1]+=t;v[2]+=t*t;v[3]=max(v[3],t)
        else:
            kind="macos" if name.startswith(b"__MACOSX") else "directories" if name.endswith(b"/") else "other"
            other[kind]+=1
    if start!=end:
        raise ValueError(f"Archive central index stopped at {start}, expected {end}")
    result={}
    bad=[]
    for clip,(count,total,squares,maximum) in clips.items():
        n=count-1
        if maximum==n and total==n*(n+1)//2 and squares==n*(n+1)*(2*n+1)//6:
            result[clip]=count
        else:
            bad.append((clip,count,maximum))
    return result,bad,dict(other)


def central_bounds(raw,inner_start,limit):
    end=raw.rfind(b"PK\x06\x06",inner_start,limit)
    if end<inner_start:
        raise ValueError("Missing ZIP64 directory marker")
    sig,size,v1,v2,d1,d2,entries,entries2,central_size,central_offset=struct.unpack_from("<4sQHHIIQQQQ",raw,end)
    if entries!=entries2 or min(d1,d2)!=0:
        raise ValueError("Multi-volume ZIP unexpected")
    if inner_start+central_offset+central_size!=end:
        raise ValueError("Invalid central offsets")
    return inner_start+central_offset,end,entries


def describe(ids):
    count=np.array(list(ids.values()))
    qs=np.percentile(count,[0,25,50,75,95,100])
    return dict(clips=len(ids),frames=int(count.sum()),sources=len({c[:11] for c in ids}),
                mean_frames=float(count.mean()),frame_quantiles=qs.tolist(),
                nominal_24fps_duration_quantiles_sec=(qs/24).tolist())


def canonical(fit):
    t=len(fit["body_pose"])
    return np.concatenate([fit.get(k,np.zeros((t,w),np.float32)) for k,w in zip(KEYS,WIDTHS)],axis=1).reshape(t,55,3)


def restore(base,poses):
    out=dict(base)
    t=len(poses)
    flat=poses.reshape(t,-1)
    i=0
    for k,w in zip(KEYS,WIDTHS):
        out[k]=flat[:,i:i+w].astype(np.float32)
        i+=w
    return out


def get_fit(raw,records):
    rows=[]
    for _,start,packed,size,crc,method in sorted(records):
        if size>65536 or method not in (0,8):
            raise ValueError("Unexpected frame payload")
        payload=raw[start:start+packed]
        data=zlib.decompress(payload,-15) if method else payload
        if len(data)!=size or (zlib.crc32(data)&0xffffffff)!=crc:
            raise ValueError("Bad fit member CRC")
        r=ArrayUnpickler(io.BytesIO(data)).load()
        row={}
        for src,(name,width) in FIELDS.items():
            data=np.asarray(r[src])
            if data.shape not in ((width,),(1,width)) or not np.isfinite(data).all() or data.dtype.kind!="f":
                raise ValueError((src,data.shape,data.dtype))
            row[name]=data.reshape(width).astype(np.float32)
        rows.append(row)
    return {k:np.stack([x[k] for x in rows]) for k in rows[0]}


def pick(clips,limit,seed,exclude):
    bins=defaultdict(list)
    for c in sorted(clips):
        if c[:11] not in exclude:
            bins[c[:11]].append(c)
    rng=np.random.default_rng(seed)
    keys=sorted(bins)
    rng.shuffle(keys)
    for v in bins.values():
        rng.shuffle(v)
    # Round-robin recordings before taking a second clip from a recording.
    result=[]
    while keys and len(result)<limit:
        next_keys=[]
        for k in keys:
            result.append(bins[k].pop())
            if len(result)==limit:
                break
            if bins[k]:
                next_keys.append(k)
        keys=next_keys
    return result


def quality(fit):
    pose=canonical(fit)
    rots=Rotation.from_rotvec(pose.reshape(-1,3)).as_matrix().reshape(*pose.shape[:-1],3,3)
    axis=Rotation.from_matrix(rots.reshape(-1,3,3)).as_rotvec()
    restored=Rotation.from_rotvec(axis).as_matrix().reshape(rots.shape)
    steps=Rotation.from_matrix((rots[:-1].swapaxes(-1,-2)@rots[1:]).reshape(-1,3,3)).magnitude().reshape(len(rots)-1,55)
    groups={"root":[0],"body":ACTIVE[:13],"left":np.arange(25,40),"right":np.arange(40,55),"legs":[1,2,4,5,7,8,10,11]}
    q={f"{k}_step_p99_deg":float(np.rad2deg(np.percentile(steps[:,ids],99))) for k,ids in groups.items()}
    q.update(beta_max_span=float(np.ptp(fit["betas"],axis=0).max()),
             camera_z_span=float(np.ptp(fit["cam_trans"][:,2])),
             cam_step_jerk_p95=float(np.percentile(np.linalg.norm(np.diff(fit["cam_trans"],n=3,axis=0),axis=1),95)),
             step_gt90_count=int((steps>np.pi/2).sum()),
             roundtrip_max_matrix=float(np.max(np.abs(restored-rots))),
             local_rotation_gt150_fraction=float(np.mean(np.linalg.norm(pose[:,25:],axis=-1)>np.deg2rad(150))))
    return q


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prefix",type=Path,required=True)
    p.add_argument("--train-central",type=Path)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--limits",type=int,nargs=3,default=(600,75,100))
    p.add_argument("--audit-only",action="store_true")
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    with a.prefix.open("rb") as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as raw:
        starts={s:start_at(raw,s) for s in SPLITS}
        archive={}
        full={}
        train_other={}
        if a.train_central and a.train_central.exists():
            with a.train_central.open("rb") as z,mmap.mmap(z.fileno(),0,access=mmap.ACCESS_READ) as m:
                end=m.rfind(b"PK\x06\x06")
                match=struct.unpack_from("<4sQHHIIQQQQ",m,end)
                if match[8]!=end: raise ValueError("Training central range mismatch")
                full["train"],bad,extra=index_central(m,0,end)
                train_other={"central_entries":match[6],"bad_indices":bad,"ignored_entries":extra}
        for split in SPLITS:
            if split=="train":
                archive[split]={"archive_coverage":"complete central index" if split in full else "prefix only",**train_other}
                continue
            # The nested ZIP64 footer precedes the outer data descriptor and following header.
            start,end,entries=central_bounds(raw, starts[split],starts["test" if split=="val" else "train"]-24)
            full[split],bad,extra=index_central(raw,start,end)
            archive[split]={"archive_coverage":"complete central index","central_entries":entries,
                            "bad_indices":bad,"ignored_entries":extra}
        collections={}
        for split in SPLITS:
            found=locals_by_clip(raw,starts[split])
            collections[split]={k:v for k,v in found.items() if sorted(r[0] for r in v)==list(range(len(v)))
                                and (split not in full or full[split].get(k)==len(v)) and len(v)>=40}
            archive[split]["sample_pool"]=describe({k:len(v) for k,v in collections[split].items()})
            if split not in full:
                full[split]={k:len(v) for k,v in collections[split].items()}
        groups={s:{c[:11] for c in full[s]} for s in SPLITS}
        for s in SPLITS:
            archive[s].update(describe(full[s]))
        overlap={s+"_"+u:sorted(groups[s]&groups[u]) for s,u in (("train","val"),("train","test"),("val","test"))}
        with (a.output/"archive_inventory.csv").open("w",newline="") as handle:
            w=csv.writer(handle);w.writerow(["split","clip","source","frames"])
            for s in SPLITS:w.writerows((s,k,k[:11],n) for k,n in sorted(full[s].items()))
        (a.output/"inventory.json").write_text(json.dumps(dict(official_archive=archive,source_overlap=overlap),indent=2)+"\n")
        print(json.dumps({k:(v["clips"],v["frames"],v["sources"],v["sample_pool"]["clips"]) for k,v in archive.items()}),flush=True)
        if a.audit_only:return
        exclusion={s:HELD | set().union(*(groups[k]&groups[s] for k in SPLITS if k!=s)) for s in SPLITS}
        rows=[];reject=[]
        for j,split in enumerate(SPLITS):
            # When only an indexed prefix is available, also eliminate the last possibly incomplete clip.
            candidates=collections[split]
            if split=="train" and archive[split]["archive_coverage"]=="prefix only":
                candidates={k:v for k,v in candidates.items() if max(row[1] for row in v)<len(raw)-65536}
            for k in pick(candidates,a.limits[j],532+j,exclusion[split]):
                try:
                    d=get_fit(raw,candidates[k])
                    q=quality(d)
                except (ValueError,KeyError,zlib.error) as err:
                    reject.append(dict(clip=k,split=split,error=str(err)))
                    continue
                dest=a.output/"fits"/split
                dest.mkdir(parents=True,exist_ok=True)
                path=dest/(k+".npz")
                np.savez_compressed(path,**d)
                rows.append(dict(split=split,clip=k,source=k[:11],frames=len(d["body_pose"]),
                                 path=str(path.relative_to(a.output)),**q))
            print(split,len([r for r in rows if r["split"]==split]),flush=True)
        with (a.output/"sample.csv").open("w",newline="") as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        (a.output/"rejections.json").write_text(json.dumps(reject,indent=2)+"\n")


if __name__=="__main__":
    main()
