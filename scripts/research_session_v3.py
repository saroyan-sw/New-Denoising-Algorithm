"""Bounded, resumable CPU research protocol. See research_plan.md before use."""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import random
import socket
import sys
import threading
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from denoise_research.session_guard import atomic_write_json, read_power_status
from denoise_research.operators import bilateral_filter, gaussian_blur, median_filter

OUT = ROOT / 'outputs' / 'research_v3'
OLD = ROOT / 'outputs' / 'transformer_fusion_session'
SIGMA = 25 / 255
# Equivalent active-work start after the user explicitly resumed on 2026-10-22.
# It carries forward 2,208.606 seconds consumed in the original 2026-10-14 session,
# leaving 12,191.394 seconds rather than resetting the four-hour allowance.
START = 1792610865.57409
SEEDS = [20261014, 20261015, 20261016]
VERSION = 4
BASE = ['identity', 'gaussian_07', 'gaussian_11', 'bilateral', 'median', 'nlm_07', 'nlm_09', 'nlm_11', 'cbm3d']
BANK = ['identity', 'bilateral', 'nlm_07', 'cbm3d']
CONFIG = dict(version=VERSION, sigma=SIGMA, seeds=SEEDS, bank=BANK, lambdas=[0., .01, .03],
              steps=40, lr=.08, training_sources=16, benchmark_sources=20,
              objective='MC-SURE / sigma^2 - lambda * (MUSIQ - train_mean)/train_std',
              mc_probes=2, mc_epsilon=.001, boundary='operator-specific native; same noisy array',
              output_clip=True, no_input_clip=True, budget_start=START,
              original_budget_start=1791923291.0, carried_active_seconds=2208.6060474,
              resumed_at=1792613074.1801372, max_hours=4, max_runs=10)

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, data):
    atomic_write_json(path, data)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''):
            h.update(b)
    return h.hexdigest()

def validate_hash_manifest(record, paths):
    """Validate an exact key-to-file hash manifest; also used by gate tests."""
    for key,path in paths.items():
        path=Path(path)
        if key not in record or not path.exists() or record[key]!=sha(path):
            raise RuntimeError('Lock hash mismatch: '+key)

def validate_holdout_binding(manifest, lock_sha, data_path, complete=None):
    data_sha=sha(data_path)
    if manifest.get('lock_sha256')!=lock_sha or manifest.get('data_sha256')!=data_sha:
        raise RuntimeError('Holdout data is stale or not bound to the selected lock')
    if complete is not None and (complete.get('lock_sha256')!=lock_sha or complete.get('data_sha256')!=data_sha):
        raise RuntimeError('Stale holdout completion marker')
    return data_sha

def rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB'), dtype=np.float32) / 255

def png(path, x):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.round(np.clip(x, 0, 1) * 255).astype('uint8')).save(path)

def save_np(path, **data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp.npz')
    (np.savez if 'partial' in path.name else np.savez_compressed)(temp, **data)
    replace_retry(temp, path)

def replace_retry(source, target):
    """Preserve atomic replacement while tolerating transient Windows file scans."""
    for attempt in range(20):
        try:
            os.replace(source, target)
            return
        except PermissionError:
            if attempt == 19:
                raise
            time.sleep(.05 * (attempt + 1))

class Watchdog:
    """Host-side thread monitors physical Windows power independent of chat."""
    def __init__(self, root=OUT, interval=30, reader=read_power_status):
        self.root, self.interval, self.reader = Path(root), interval, reader
        self.done, self.stop = threading.Event(), threading.Event()
        # boundary() and the host polling thread can otherwise race while replacing
        # the same heartbeat on Windows. Serialize the full read/decision/write.
        self.poll_lock = threading.Lock()
        self.reason = None
        self.poll()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def poll(self):
        with self.poll_lock:
            p = self.reader()
            reason = None
            if p.battery_percent is not None and p.battery_percent <= 15:
                reason = 'battery_at_or_below_15'
            elif p.source != 'simulation' and (p.on_ac is None or (p.battery_present and p.battery_percent is None)):
                reason = 'physical_battery_telemetry_unavailable'
            elif time.time() - START >= 4 * 3600:
                reason = 'four_hour_budget_exhausted'
            if reason:
                self.reason = reason
                self.stop.set()
            write(self.root / 'watchdog.json', dict(time=time.time(), pid=os.getpid(),
                  battery=p.battery_percent, ac=p.on_ac, source=p.source, stop=self.stop.is_set(), reason=self.reason))

    def loop(self):
        while not self.done.wait(self.interval):
            try:
                self.poll()
            except BaseException as error:
                self.reason = repr(error)
                self.stop.set()

    def boundary(self, stage):
        self.poll()
        if self.stop.is_set():
            write(self.root / 'stopped.json', dict(stage=stage, reason=self.reason, time=time.time()))
            raise RuntimeError(f'Watchdog stopped at {stage}: {self.reason}')

    def close(self):
        self.done.set()
        self.thread.join(timeout=2)

def no_lock():
    if (OUT / 'locked_model.json').exists():
        raise RuntimeError('Model selection is locked; further development forbidden')

def tensor(x):
    import torch
    return torch.from_numpy(np.ascontiguousarray(x)).permute(0, 3, 1, 2).float()

def setup_torch():
    import torch
    torch.set_num_threads(4)
    torch.manual_seed(20261014)
    np.random.seed(20261014)
    random.seed(20261014)
    torch.use_deterministic_algorithms(True)
    os.environ['TORCH_HOME'] = str(OLD / 'model_cache')
    return torch

def atomic_torch(path, data):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as f:
        torch.save(data, f)
        f.flush()
        os.fsync(f.fileno())
    replace_retry(temp, path)

def check_data(split):
    manifest = read(ROOT / 'benchmark20.json')
    samples = manifest[split]
    for item in samples:
        if sha(ROOT / item['path']) != item['sha256']:
            raise RuntimeError('Source changed: ' + item['path'])
    frozen = OUT / f'{split}_data.npz'
    hashes = read(OUT / 'data_hashes.json')
    if frozen.name not in hashes or sha(frozen) != hashes[frozen.name]:
        raise RuntimeError('Frozen noisy/clean data changed: ' + str(frozen))
    return samples

def content(path):
    x = rgb(path)
    g = x.mean(2)
    dx, dy = np.diff(g, axis=1), np.diff(g, axis=0)
    edge = float(np.abs(dx).mean() + np.abs(dy).mean())
    fine = float(np.abs(g[1:-1,1:-1] * 4 - g[:-2,1:-1] - g[2:,1:-1] - g[1:-1,:-2] - g[1:-1,2:]).mean())
    repeat = max(float(np.corrcoef(g[:, :-s].ravel(), g[:, s:].ravel())[0,1]) for s in [4,8,16])
    small = np.asarray(Image.fromarray((x*255).astype('uint8')).convert('L').resize((9,8)))
    dhash = ''.join('1' if a else '0' for a in (small[:,1:] > small[:,:-1]).ravel())
    return dict(edge=edge, fine=fine, repeat=repeat, challenge=float(x.std()), dhash=dhash)

def prepare(wd):
    no_lock()
    if (ROOT / 'benchmark20.json').exists():
        raise RuntimeError('benchmark20 is immutable; already prepared')
    source = ROOT / 'experiments/coefficient_fitting/results/nima_technical_100/images'
    unique = {}
    for p in sorted(source.glob('*_clean.png')):
        sid = p.stem.split('_r')[0].split('_', 1)[1]
        if sid not in unique:
            unique[sid] = dict(id=sid, path=p.relative_to(ROOT).as_posix(), sha256=sha(p), **content(p))
    pool = list(unique.values())
    rng = np.random.default_rng(20261014)
    selected = []
    rules = [('smooth', 'edge', False), ('edges', 'edge', True), ('fine_texture', 'fine', True),
             ('repeated_structure', 'repeat', True), ('challenging', 'challenge', True)]
    for category, field, reverse in rules:
        ranked = sorted(pool, key=lambda r:r[field], reverse=reverse)[:10]
        picks = [ranked[i] for i in sorted(rng.choice(len(ranked), 4, replace=False))]
        for p in picks:
            selected.append(dict(p, category=category))
            pool.remove(p)
    # Conservative near-duplicate exclusion for training; exact source IDs also excluded.
    def distance(a,b):
        return sum(x != y for x,y in zip(a['dhash'], b['dhash']))
    eligible = [p for p in pool if min(distance(p,q) for q in selected) > 6]
    train = [eligible[i] for i in sorted(rng.choice(len(eligible), 16, replace=False))]
    manifest = dict(version=3, selection_seed=20261014, selection='Four seeded choices from each sequential top-10 content-proxy stratum; no denoising scores used',
        categories='Statistical content proxies, verified by clean grid; category membership is not semantic ground truth',
        prior_exposure='All archived sources historically exposed. Preliminary v1/v2 excluded from valid selection. Final holdout is separate Urban100.',
        crop='Existing native 128x128 archive crop, no resize; original crop metadata in archived metrics.csv',
        corruption=dict(sigma=SIGMA, clip_input=False, seeds=SEEDS), benchmark=selected, training=train,
        split_audit=dict(source_overlap=0, sha256_overlap=0, minimum_train_benchmark_dhash_distance=min(distance(a,b) for a in train for b in selected)))
    write(ROOT / 'benchmark20.json', manifest)
    write(OUT / 'config.json', CONFIG)
    for split, rows in [('benchmark', selected), ('training', train)]:
        clean = np.stack([rgb(ROOT/p['path']) for p in rows])
        noisy = []
        draw_seeds = SEEDS if split == 'benchmark' else [20261004]
        for seed in draw_seeds:
            noisy.append(np.stack([x + np.random.default_rng(seed+i*1009).normal(0,SIGMA,x.shape).astype('float32') for i,x in enumerate(clean)]))
        save_np(OUT/f'{split}_data.npz', clean=clean, noisy=np.stack(noisy))
    clean = np.load(OUT/'benchmark_data.npz')['clean']
    grid(clean, [p['id']+' '+p['category'] for p in selected], OUT/'clean_selection.png')
    write(OUT/'data_hashes.json', {p.name:sha(p) for p in OUT.glob('*_data.npz')})
    wd.boundary('prepared')

def grid(images, labels, path, cols=5):
    h,w = images[0].shape[:2]
    sheet = Image.new('RGB',(cols*w, ((len(images)+cols-1)//cols)*(h+28)), 'white')
    draw = ImageDraw.Draw(sheet)
    for i,(x,label) in enumerate(zip(images,labels)):
        ox,oy=(i%cols)*w,(i//cols)*(h+28)
        sheet.paste(Image.fromarray(np.round(np.clip(x,0,1)*255).astype('uint8')),(ox,oy+28))
        draw.text((ox+2,oy+2),label[:25],fill='black')
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    sheet.save(path)

def operator(name,x):
    from skimage.restoration import denoise_nl_means
    if name == 'identity': return x.copy()
    if name.startswith('gaussian'): return gaussian_blur(x, .7 if name.endswith('07') else 1.1)
    if name == 'bilateral': return bilateral_filter(x,radius=3,sigma_spatial=2,sigma_range=.12)
    if name == 'median': return median_filter(x,3)
    if name.startswith('nlm'):
        return denoise_nl_means(x,h=float(name[-2:])/10*SIGMA,sigma=SIGMA,patch_size=5,patch_distance=6,fast_mode=True,channel_axis=-1,preserve_range=True).astype('float32')
    if name == 'cbm3d':
        from bm3d import bm3d_rgb
        return bm3d_rgb(x,sigma_psd=SIGMA).astype('float32')
    raise ValueError(name)

def classical(split, wd):
    if split != 'holdout': no_lock()
    data=np.load(OUT/f'{split}_data.npz')
    noisy=data['noisy']
    for name in BASE:
        path=OUT/'cache'/split/f'{name}.npz'
        if path.exists(): continue
        values=np.zeros_like(noisy); times=np.zeros(noisy.shape[:2])
        progress=path.with_name(name+'_partial.npz')
        done=0
        if progress.exists():
            p=np.load(progress); values=p['values']; times=p['times']; done=int(p['done'])
        for index in range(done, noisy.shape[0]*noisy.shape[1]):
            wd.boundary(f'{split}_{name}_{index}')
            seed,i=divmod(index,noisy.shape[1]); start=time.perf_counter()
            values[seed,i]=operator(name,noisy[seed,i]); times[seed,i]=time.perf_counter()-start
            save_np(progress,values=values,times=times,done=index+1)
        save_np(path,values=values,times=times)
        print(split,name,'seconds',round(times.sum(),2),flush=True)

def vendor_files():
    commit='fc1732f4a4514e42ce15e5b3a1e18c828af47a1e'
    return {
      'dncnn_color_blind.pth':'https://github.com/cszn/KAIR/releases/download/v1.0/dncnn_color_blind.pth',
      'drunet_color.pth':'https://github.com/cszn/KAIR/releases/download/v1.0/drunet_color.pth',
      'models/basicblock.py':f'https://raw.githubusercontent.com/cszn/KAIR/{commit}/models/basicblock.py',
      'models/network_unet.py':f'https://raw.githubusercontent.com/cszn/KAIR/{commit}/models/network_unet.py',
      'models/network_dncnn.py':f'https://raw.githubusercontent.com/cszn/KAIR/{commit}/models/network_dncnn.py',
      'LICENSE':f'https://raw.githubusercontent.com/cszn/KAIR/{commit}/LICENSE'}

def verify_vendor():
    provenance_path=OUT/'vendor/provenance.json'
    if not provenance_path.exists(): raise RuntimeError('Missing vendor provenance; run acquire')
    provenance=read(provenance_path); expected=vendor_files()
    if set(provenance)!=set(expected): raise RuntimeError('Vendor provenance file set mismatch')
    for name,url in expected.items():
        path=OUT/'vendor'/name; record=provenance[name]
        if not path.exists() or record.get('url')!=url or record.get('sha256')!=sha(path) or record.get('bytes')!=path.stat().st_size:
            raise RuntimeError('Vendor provenance mismatch: '+name)
    return sha(provenance_path)

def acquire(wd):
    no_lock()
    files=vendor_files()
    provenance={}
    for name,url in files.items():
        wd.boundary('download_'+name)
        path=OUT/'vendor'/name; path.parent.mkdir(parents=True,exist_ok=True)
        if not path.exists():
            temp=path.with_suffix(path.suffix+'.download')
            download(url,temp,wd); os.replace(temp,path)
        provenance[name]=dict(url=url,sha256=sha(path),bytes=path.stat().st_size)
    write(OUT/'vendor/provenance.json',provenance)
    verify_vendor()

def download(url, path, wd):
    """Bounded network reads with progress, retries, and watchdog boundaries."""
    for attempt in range(4):
        wd.boundary('download_attempt')
        offset=path.stat().st_size if path.exists() else 0
        request=urllib.request.Request(url,headers={'Range':f'bytes={offset}-'} if offset else {})
        try:
            with urllib.request.urlopen(request,timeout=20) as response:
                append=offset>0 and response.status==206
                if not append:offset=0
                expected=int(response.headers.get('Content-Length','0'))+offset
                with path.open('ab' if append else 'wb') as stream:
                    count=offset;last=time.monotonic()
                    while True:
                        block=response.read(65536)
                        if not block:break
                        stream.write(block);count+=len(block)
                        if time.monotonic()-last>10:
                            stream.flush();print(path.name,count,'bytes',flush=True);wd.boundary('download_progress');last=time.monotonic()
                if expected and path.stat().st_size!=expected:raise OSError('Incomplete download')
            return
        except (OSError,TimeoutError) as error:
            if attempt==3:raise
            print('Download retry',attempt+1,repr(error),flush=True)

def network(name):
    verify_vendor()
    torch=setup_torch()
    sys.path.insert(0,str(OUT/'vendor'))
    if name=='dncnn':
        from models.network_dncnn import DnCNN
        model=DnCNN(in_nc=3,out_nc=3,nc=64,nb=20,act_mode='R')
        checkpoint='dncnn_color_blind.pth'
    else:
        from models.network_unet import UNetRes
        model=UNetRes(in_nc=4,out_nc=3,nc=[64,128,256,512],nb=4,act_mode='R',downsample_mode='strideconv',upsample_mode='convtranspose',bias=False)
        checkpoint='drunet_color.pth'
    model.load_state_dict(torch.load(OUT/'vendor'/checkpoint,map_location='cpu',weights_only=True),strict=True)
    model.eval().requires_grad_(False)
    return model

def neural(split,wd):
    if split!='holdout': no_lock()
    torch=setup_torch(); noisy=np.load(OUT/f'{split}_data.npz')['noisy']
    for name in ['dncnn','drunet']:
        path=OUT/'cache'/split/f'{name}.npz'
        if path.exists(): continue
        wd.boundary('load_'+name); model=network(name)
        values=np.zeros_like(noisy); times=np.zeros(noisy.shape[:2]); partial=path.with_name(name+'_partial.npz'); done=0
        if partial.exists():
            p=np.load(partial); values=p['values'];times=p['times'];done=int(p['done'])
        for index in range(done,noisy.shape[0]*noisy.shape[1]):
            wd.boundary(f'{split}_{name}_{index}'); seed,i=divmod(index,noisy.shape[1])
            x=tensor(noisy[seed,i][None]); start=time.perf_counter()
            with torch.no_grad():
                if name=='drunet': x=torch.cat([x,torch.full_like(x[:,:1],SIGMA)],1)
                y=model(x)
            values[seed,i]=y[0].permute(1,2,0).numpy();times[seed,i]=time.perf_counter()-start
            save_np(partial,values=values,times=times,done=index+1)
        save_np(path,values=values,times=times)
        print(split,name,'seconds',round(times.sum(),2),flush=True)
        del model

def stack(split,bank=BANK):
    return np.stack([np.load(OUT/'cache'/split/f'{name}.npz')['values'] for name in bank],axis=2)

def simplex(v):
    u=np.sort(v)[::-1]; c=np.cumsum(u)-1; i=np.arange(1,len(v)+1); good=u-c/i>0
    return np.maximum(v-c[good][-1]/i[good][-1],0)

def optimize_quadratic(gram,linear):
    from scipy.optimize import minimize
    k=len(linear)
    r=minimize(lambda w:float(w@gram@w+linear@w),np.ones(k)/k,
        jac=lambda w:2*gram@w+linear,method='SLSQP',bounds=[(0,1)]*k,
        constraints=[dict(type='eq',fun=lambda w:w.sum()-1,jac=lambda w:np.ones(k))],
        options=dict(ftol=1e-13,maxiter=2000))
    if not r.success: raise RuntimeError(r.message)
    return r.x

def sure_terms(wd):
    no_lock()
    path=OUT/f'sure_terms_v{VERSION}.npz'
    if path.exists(): return
    y=np.load(OUT/'training_data.npz')['noisy'][0]; z=stack('training')[0]
    div=np.zeros((len(y),len(BANK)),dtype='float64')
    partial=OUT/f'sure_partial_v{VERSION}.npz'; done=0
    if partial.exists():
        with np.load(partial) as p:
            div=p['div'].copy();done=int(p['done'])
    for i in range(done,len(y)):
        for j in range(CONFIG['mc_probes']):
            probe=np.random.default_rng(30261014+i*47+j).choice([-1.,1.],size=y[i].shape).astype('float32')
            for k,name in enumerate(BANK):
                wd.boundary(f'divergence_{i}_{j}_{name}')
                perturbed=operator(name,y[i]+.001*probe)
                div[i,k]+=np.mean(probe*(perturbed-z[i,k]))/.001/CONFIG['mc_probes']
        save_np(partial,div=div,done=i+1)
    residual=z-y[:,None]
    gram=np.einsum('ikhwc,ilhwc->kl',residual.astype('float64'),residual.astype('float64'))/y.size
    linear=2*SIGMA**2*div.mean(0)
    w=optimize_quadratic(gram,linear)
    save_np(path,gram=gram,linear=linear,weights=w,div=div)
    write(OUT/f'sure_solution_v{VERSION}.json',dict(weights=w.tolist(),bank=BANK,risk=float(w@gram@w+linear@w-SIGMA**2)))

def oracle(wd):
    no_lock()
    from scipy.optimize import minimize
    x=np.load(OUT/'benchmark_data.npz')['clean']; z=stack('benchmark')[0].astype('float64')
    def fit(zz,xx):
        k=zz.shape[1]
        def objective(w):
            p=np.einsum('nkhwc,k->nhwc',zz,w); mask=(p>0)&(p<1); e=np.clip(p,0,1)-xx
            return float(np.mean(e**2)),2*np.einsum('nhwc,nkhwc->k',e*mask,zz)/e.size
        starts=[np.eye(k)[i] for i in range(k)]+[np.ones(k)/k]
        results=[minimize(objective,w,jac=True,method='SLSQP',bounds=[(0,1)]*k,
             constraints=[dict(type='eq',fun=lambda w:w.sum()-1,jac=lambda w:np.ones(k))],options=dict(ftol=1e-13,maxiter=1000)) for w in starts]
        valid=[r for r in results if r.success]
        if not valid: raise RuntimeError('oracle failed convergence')
        best=min(valid,key=lambda r:r.fun);return best.x,float(best.fun)
    w,mse=fit(z,x)
    image_weights=[];image_mse=[]
    for i in range(len(x)):
        wd.boundary('oracle_'+str(i));ww,mm=fit(z[i:i+1],x[i:i+1]);image_weights.append(ww.tolist());image_mse.append(mm)
    errors=(np.clip(z,0,1)-x[:,None]).transpose(1,0,2,3,4).reshape(len(BANK),-1)
    write(OUT/f'oracle_v{VERSION}.json',dict(constraints='global/per-image simplex, clipped-output MSE, seed0 development only',
        global_weights=w.tolist(),global_mse=mse,per_image_weights=image_weights,per_image_mse=image_mse,
        error_correlations=np.corrcoef(errors).tolist(),bank=BANK))
    save_np(OUT/f'cache/benchmark/oracle_global_v{VERSION}.npz',values=np.einsum('sikhwc,k->sihwc',stack('benchmark'),w),times=np.zeros((3,20)))

def model_metric(name):
    setup_torch()
    import pyiqa
    return pyiqa.create_metric(name,device='cpu',as_loss=True).eval().requires_grad_(False)

def run_id(lam): return (f'v{VERSION}_lambda_'+str(lam)).replace('.','p')

def optimization_step(logits, optimizer, gram, linear, z, scorer, cal, lam, step):
    import torch
    optimizer.zero_grad();weights=torch.softmax(logits,0)
    indices=list(range((step%4)*4,(step%4)*4+4))
    pred=(z[indices]*weights[None,:,None,None,None]).sum(1).clamp(0,1)
    pred.retain_grad();quality=scorer(pred).mean();risk=(weights@gram@weights+linear@weights-SIGMA**2)/SIGMA**2
    loss=risk-lam*(quality-cal['mean'])/cal['std'];loss.backward()
    gradient_norm=float(pred.grad.norm())
    optimizer.step()
    return dict(step=step,loss=float(loss.detach()),sure=float(risk.detach()),musiq=float(quality.detach()),weights=weights.detach().tolist()),gradient_norm

def train(lam,wd):
    no_lock();torch=setup_torch()
    if lam not in CONFIG['lambdas']: raise ValueError('Undeclared lambda')
    rid=run_id(lam); directory=OUT/'runs'/rid;directory.mkdir(parents=True,exist_ok=True)
    if (directory/'model.json').exists(): return
    ledgerpath=OUT/'run_budget.json'
    ledger=read(ledgerpath) if ledgerpath.exists() else dict(preliminary_runs=2,started=[])
    if rid not in ledger['started']:
        if ledger['preliminary_runs']+len(ledger['started'])>=10: raise RuntimeError('Ten-run limit')
        ledger['started'].append(rid);write(ledgerpath,ledger)
    wd.boundary('training_load')
    terms=np.load(OUT/f'sure_terms_v{VERSION}.npz');w0=terms['weights']
    logits=torch.tensor(np.log(np.maximum(w0,.001)),dtype=torch.float32,requires_grad=True)
    optimizer=torch.optim.Adam([logits],lr=CONFIG['lr'])
    gram=torch.tensor(terms['gram'],dtype=torch.float32);linear=torch.tensor(terms['linear'],dtype=torch.float32)
    z=torch.from_numpy(stack('training')[0]).permute(0,1,4,2,3)
    scorer=model_metric('musiq')
    quality_cal=OUT/f'quality_calibration_v{VERSION}.json'
    if quality_cal.exists(): cal=read(quality_cal)
    else:
        scores=[]
        with torch.no_grad():
            for k in range(len(BANK)):
                wd.boundary('iqa_calibration');scores.extend(scorer(z[:4,k].clamp(0,1)).flatten().tolist())
        cal=dict(mean=float(np.mean(scores)),std=max(float(np.std(scores)),1));write(quality_cal,cal)
    audit_weights=torch.softmax(logits,0)
    audit_pred=(z[:4]*audit_weights[None,:,None,None,None]).sum(1).clamp(0,1)
    audit_quality=scorer(audit_pred).mean()
    musiq_logit_gradient_norm=float(torch.autograd.grad(audit_quality,logits)[0].norm())
    checkpoint=directory/'checkpoint.pt';history=[];start=0
    if checkpoint.exists():
        saved=torch.load(checkpoint,weights_only=False);logits.data.copy_(saved['logits']);optimizer.load_state_dict(saved['optimizer']);history=saved['history'];start=saved['next_step']
        if saved.get('config')!=CONFIG or not 0<=start<=CONFIG['steps'] or len(history)!=start:
            raise RuntimeError('Incompatible or corrupt training checkpoint')
        torch.set_rng_state(saved['torch_rng']);random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng'])
        prior_training_seconds=float(saved.get('elapsed_training_seconds',0.))
    else: prior_training_seconds=0.
    start_time=time.perf_counter();gradient_norm=0.
    if lam==0:
        weights=w0
    else:
        for step in range(start,CONFIG['steps']):
            wd.boundary('train_before_'+str(step))
            entry,gn=optimization_step(logits,optimizer,gram,linear,z,scorer,cal,lam,step)
            gradient_norm=max(gradient_norm,gn);history.append(entry)
            atomic_torch(checkpoint,dict(logits=logits.detach(),optimizer=optimizer.state_dict(),scheduler=None,scaler=None,
                 torch_rng=torch.get_rng_state(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),config=CONFIG,
                 next_step=step+1,history=history,stop_requested=wd.stop.is_set(),
                 elapsed_training_seconds=prior_training_seconds+time.perf_counter()-start_time))
            if step%10==0: print(rid,'step',step,'loss',round(entry['loss'],5),flush=True)
            wd.boundary('train_after_'+str(step))
        weights=torch.softmax(logits.detach(),0).numpy()
    model=dict(run_id=rid,weights=list(map(float,weights)),bank=BANK,lambda_iqa=lam,config=CONFIG,
        musiq_frozen=all(not p.requires_grad for p in scorer.parameters()),input_gradient_norm=gradient_norm,
        musiq_logit_gradient_norm=musiq_logit_gradient_norm,
        musiq_preprocess=scorer.net.data_preprocess_opts,
        training_seconds=prior_training_seconds+time.perf_counter()-start_time,
        training_seconds_note='Aggregate active optimizer time across checkpoint resumes; excludes scoring and output inference',
        checkpoint_sha256=sha(checkpoint) if checkpoint.exists() else None,
        training_manifest_sha256=sha(ROOT/'benchmark20.json'),history=history,
        pyiqa_version=importlib.metadata.version('pyiqa'),musiq_score_direction='higher_is_better',
        musiq_input='Fusion output clipped to RGB [0,1]; PyIQA eval normalization and multiscale preprocessing',
        musiq_checkpoint_url='https://huggingface.co/chaofengc/IQA-PyTorch-Weights/resolve/main/musiq_koniq_ckpt-e95806b9.pth',
        musiq_checkpoint_sha256=sha(OLD/'model_cache/hub/pyiqa/musiq_koniq_ckpt-e95806b9.pth'))
    write(directory/'model.json',model)
    atomic_torch(directory/'deploy.pt',dict(weights=torch.tensor(weights),bank=BANK,config=CONFIG))
    predict_model(model,'benchmark',wd)

def predict_model(model,split,wd):
    wd.boundary('predict_'+model['run_id']);z=stack(split,model['bank'])
    start=time.perf_counter();v=np.einsum('sikhwc,k->sihwc',z,np.array(model['weights'],dtype='float32'));fusion_time=(time.perf_counter()-start)/v.shape[0]/v.shape[1]
    times=sum(np.load(OUT/'cache'/split/f'{name}.npz')['times'] for name in model['bank'])+fusion_time
    save_np(OUT/'cache'/split/(model['run_id']+'.npz'),values=v,times=times)

def equal(split):
    z=stack(split);times=sum(np.load(OUT/'cache'/split/f'{name}.npz')['times'] for name in BANK)
    save_np(OUT/f'cache/{split}/equal_fusion_v{VERSION}.npz',values=z.mean(2),times=times)

def retrospective_pilots():
    """Apply historical pilot weights to benchmark20; mark overlap/ineligibility."""
    bank=BASE[:5];z=stack('benchmark',bank)
    timings=sum(np.load(OUT/'cache/benchmark'/f'{name}.npz')['times'] for name in bank)
    old_manifest=read(OLD/'dev_manifest.json')
    fit_ids={p['source_id'] for p in old_manifest['samples'] if p['role']=='fit'}
    overlap=sorted(fit_ids & {p['id'] for p in read(ROOT/'benchmark20.json')['benchmark']})
    for path in sorted((OLD/'training').glob('*/model.json')):
        model=read(path);rid='ineligible_pilot_'+model['run_id']
        values=np.einsum('sikhwc,k->sihwc',z,np.asarray(model['weights'],dtype='float32'))
        save_np(OUT/'cache/benchmark'/f'{rid}.npz',values=values,times=timings)
    write(OUT/'pilot_ineligibility.json',dict(reason='Pre-protocol runs trained on part of benchmark20; no model selection or generalization claims',overlapping_source_ids=overlap))

def evaluate(split,wd):
    import psutil
    from skimage.metrics import structural_similarity
    torch=setup_torch();data=np.load(OUT/f'{split}_data.npz');clean=data['clean']
    samples=read(ROOT/'benchmark20.json')['benchmark'] if split=='benchmark' else read(OUT/'holdout_manifest.json')['samples']
    wd.boundary('DISTS_load');dists=model_metric('dists')
    files=sorted((OUT/'cache'/split).glob('*.npz'))
    for path in files:
        if path.stem.endswith('_partial'):continue
        destination=OUT/'metrics'/split/(path.stem+'.json')
        if destination.exists():continue
        wd.boundary('evaluate_'+path.stem);p=np.load(path);values=np.clip(p['values'],0,1);times=p['times'];rows=[]
        for seed in range(len(values)):
            for i in range(len(clean)):
                wd.boundary(f'metrics_{path.stem}_{seed}_{i}');x=clean[i];y=values[seed,i]
                with torch.no_grad(): score=float(dists(tensor(y[None]),tensor(x[None])))
                mse=float(np.mean((x.astype('float64')-y)**2))
                memory=psutil.Process().memory_info()
                rows.append(dict(id=samples[i]['id'],seed=SEEDS[seed],method=path.stem,psnr=-10*np.log10(mse),
                    ssim=float(structural_similarity(x,y,data_range=1,channel_axis=-1,gaussian_weights=True,sigma=1.5,use_sample_covariance=False)),
                    dists=score,runtime_s=float(times[seed,i]),process_peak_rss_bytes=getattr(memory,'peak_wset',memory.rss)))
        write(destination,dict(rows=rows))
        print(split,path.stem,'PSNR',round(np.mean([r['psnr'] for r in rows]),4),flush=True)
    summarize(split)

def summarize(split):
    rows=[]
    for p in sorted((OUT/'metrics'/split).glob('*.json')): rows.extend(read(p)['rows'])
    if not rows:return
    with (OUT/f'{split}_results.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    methods=sorted(set(r['method'] for r in rows));summary={}
    for m in methods:
        rr=[r for r in rows if r['method']==m]
        summary[m]={k:float(np.mean([r[k] for r in rr])) for k in ['psnr','ssim','dists','runtime_s']}
        ids=sorted(set(r['id'] for r in rr))
        for metric in ['psnr','ssim','dists']:
            summary[m]['seed_mean_'+metric]={str(s):float(np.mean([r[metric] for r in rr if r['seed']==s])) for s in SEEDS}
            summary[m]['sample_sd_'+metric]=float(np.std([r[metric] for r in rr],ddof=1))
            source_means=[np.mean([r[metric] for r in rr if r['id']==ident]) for ident in ids]
            summary[m]['source_mean_sd_'+metric]=float(np.std(source_means,ddof=1))
        summary[m]['max_process_peak_rss_bytes']=max(r['process_peak_rss_bytes'] for r in rr)
    write(OUT/f'{split}_summary.json',summary)
    delta={}
    for m in methods:
        for baseline in ['cbm3d','dncnn','drunet']:
            if baseline not in methods: continue
            ids=sorted(set(r['id'] for r in rows))
            for metric in ['psnr','ssim','dists']:
                d=[]
                for ident in ids:
                    a=np.mean([r[metric] for r in rows if r['id']==ident and r['method']==m])
                    b=np.mean([r[metric] for r in rows if r['id']==ident and r['method']==baseline])
                    d.append((b-a) if metric=='dists' else (a-b))
                rng=np.random.default_rng(20261014);means=np.asarray(d)[rng.integers(0,len(d),(10000,len(d)))].mean(1)
                key=m+' minus '+baseline if metric=='psnr' else m+' '+metric.upper()+' improvement over '+baseline
                delta[key]=dict(metric=metric,positive_means_candidate_better=True,mean=float(np.mean(d)),
                    ci95=np.quantile(means,[.025,.975]).tolist(),wins=int(np.sum(np.array(d)>0)),
                    unit='source image averaged over three paired noise draws')
    write(OUT/f'{split}_paired_deltas.json',delta)

def views(method,split='benchmark'):
    data=np.load(OUT/f'{split}_data.npz');clean=data['clean'];noise=data['noisy'][0]
    names=['clean','noisy',method,'cbm3d','drunet']
    arrays=[clean,noise]+[np.load(OUT/'cache'/split/f'{name}.npz')['values'][0] for name in names[2:]]
    dest=OUT/'views'/split/method
    for page in range(4):
        images=[];labels=[]
        for i in range(page*5,page*5+5):
            for name,values in zip(names,arrays):images.append(values[i]);labels.append(f'{i+1:02} {name}')
        grid(images,labels,dest/f'page_{page+1}.png',5)
    losses=np.mean((np.clip(arrays[2],0,1)-clean)**2,axis=(1,2,3))-np.mean((np.clip(arrays[3],0,1)-clean)**2,axis=(1,2,3))
    worst=int(np.argmax(losses))
    imgs=[a[worst] for a in arrays]+[np.abs(np.clip(arrays[2][worst],0,1)-clean[worst])*5]
    grid(imgs,names+['absolute error x5'],dest/f'worst_{worst+1:02}.png',6)

def lock(wd):
    no_lock(); summary=read(OUT/'benchmark_summary.json');vendor_sha=verify_vendor()
    runs=[read(p) for p in (OUT/'runs').glob('*/model.json')]
    current=[m for m in runs if m['config']['version']==VERSION]
    if {m['lambda_iqa'] for m in current}!={0.,.01,.03}:raise RuntimeError('Current protocol ablations incomplete')
    positive=[m for m in runs if m['lambda_iqa']>0]
    if len(positive)<2:raise RuntimeError('Predeclared positive IQA ablations incomplete')
    if any(name not in summary for name in BASE+['dncnn','drunet',f'equal_fusion_v{VERSION}']+[r['run_id'] for r in runs]):raise RuntimeError('Comparisons incomplete')
    chosen=max(positive,key=lambda m:summary[m['run_id']]['psnr'])
    best_overall=max(runs,key=lambda m:summary[m['run_id']]['psnr'])
    chosen_dir=OUT/'runs'/chosen['run_id']
    write(OUT/'locked_model.json',dict(model=chosen,best_overall_fusion=best_overall['run_id'],selection='Highest development PSNR among IQA-trained positive-lambda candidates; SURE-only control reported separately',
        config_sha256=sha(OUT/'config.json'),benchmark_sha256=sha(ROOT/'benchmark20.json'),script_sha256=sha(Path(__file__)),
        vendor_provenance_sha256=vendor_sha,model_json_sha256=sha(chosen_dir/'model.json'),deploy_sha256=sha(chosen_dir/'deploy.pt'),
        musiq_checkpoint_sha256=sha(OLD/'model_cache/hub/pyiqa/musiq_koniq_ckpt-e95806b9.pth'),
        dists_checkpoint_sha256=sha(OLD/'model_cache/hub/pyiqa/DISTS_weights-f5e65c96.pth'),
        vgg_checkpoint_sha256=sha(OLD/'model_cache/hub/checkpoints/vgg16-397923af.pth'),
        pyiqa_version=importlib.metadata.version('pyiqa'),time=time.time()))
    wd.boundary('model_locked')

def holdout(wd):
    lockfile=OUT/'locked_model.json'
    if not lockfile.exists(): raise RuntimeError('Holdout gate closed')
    selected=read(lockfile)
    chosen_dir=OUT/'runs'/selected['model']['run_id']
    verify_vendor()
    validate_hash_manifest(selected,{
        'config_sha256':OUT/'config.json','benchmark_sha256':ROOT/'benchmark20.json','script_sha256':Path(__file__),
        'vendor_provenance_sha256':OUT/'vendor/provenance.json','model_json_sha256':chosen_dir/'model.json',
        'deploy_sha256':chosen_dir/'deploy.pt','musiq_checkpoint_sha256':OLD/'model_cache/hub/pyiqa/musiq_koniq_ckpt-e95806b9.pth',
        'dists_checkpoint_sha256':OLD/'model_cache/hub/pyiqa/DISTS_weights-f5e65c96.pth',
        'vgg_checkpoint_sha256':OLD/'model_cache/hub/checkpoints/vgg16-397923af.pth'})
    if selected.get('pyiqa_version')!=importlib.metadata.version('pyiqa'):raise RuntimeError('PyIQA version differs from lock')
    lock_sha=sha(lockfile)
    complete_path=OUT/'holdout_complete.json';data_path=OUT/'holdout_data.npz';manifest_path=OUT/'holdout_manifest.json'
    if complete_path.exists() and (not data_path.exists() or not manifest_path.exists()):
        raise RuntimeError('Holdout completion marker lacks bound data')
    if data_path.exists():
        if not manifest_path.exists():raise RuntimeError('Holdout data lacks manifest')
        validate_holdout_binding(read(manifest_path),lock_sha,data_path,read(complete_path) if complete_path.exists() else None)
        if complete_path.exists():return
    if not data_path.exists():
        samples=[];images=[];dev=read(ROOT/'benchmark20.json');known=dev['training']+dev['benchmark']
        for index in range(81,101):
            wd.boundary('holdout_download_'+str(index));url=f'https://raw.githubusercontent.com/wuzhongwulidong/URBAN100/master/img{index:03}.png'
            path=OUT/'holdout_raw'/f'{index:03}.png';path.parent.mkdir(parents=True,exist_ok=True)
            if not path.exists():
                temp=path.with_suffix('.download');download(url,temp,wd);os.replace(temp,path)
            x=rgb(path);h,w=x.shape[:2];top,left=(h-128)//2,(w-128)//2;crop=x[top:top+128,left:left+128];images.append(crop)
            crop_path=path.with_name(f'{index:03}_crop.png');png(crop_path,crop);features=content(crop_path)
            distance=min(sum(a!=b for a,b in zip(features['dhash'],p['dhash'])) for p in known)
            if distance<=6 or sha(crop_path) in {p['sha256'] for p in known}:raise RuntimeError('Possible holdout near-duplicate needs audit')
            samples.append(dict(id=f'Urban100_{index:03}',source_url=url,raw_sha256=sha(path),crop_sha256=sha(crop_path),top=top,left=left,size=128,min_dhash_distance=distance))
        clean=np.stack(images);noisy=np.stack([np.stack([x+np.random.default_rng(seed+1000000+i*1009).normal(0,SIGMA,x.shape).astype('float32') for i,x in enumerate(clean)]) for seed in SEEDS])
        save_np(data_path,clean=clean,noisy=noisy)
        write(manifest_path,dict(samples=samples,lock_sha256=lock_sha,
              data_sha256=sha(data_path),noise_seed_offset=1000000))
    validate_holdout_binding(read(manifest_path),lock_sha,data_path)
    classical('holdout',wd);neural('holdout',wd);equal('holdout');predict_model(selected['model'],'holdout',wd)
    evaluate('holdout',wd);views(selected['model']['run_id'],'holdout')
    write(complete_path,dict(time=time.time(),lock_sha256=lock_sha,data_sha256=sha(data_path)))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['prepare','acquire','classical','neural','sure','oracle','train','evaluate','views','lock','holdout','pilot-review']);parser.add_argument('--split',default='benchmark');parser.add_argument('--lambda-iqa',type=float);parser.add_argument('--method');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);wd=Watchdog()
    try:
        wd.boundary('command_entry')
        if args.command != 'prepare' and (OUT/'config.json').exists() and read(OUT/'config.json') != CONFIG:
            raise RuntimeError('Runtime configuration differs from frozen config.json')
        if args.command in {'classical','neural','evaluate','views'} and args.split in {'benchmark','training'}:
            check_data(args.split)
        elif args.command in {'sure'}:
            check_data('training')
        elif args.command in {'oracle','train','lock','pilot-review'}:
            check_data('training'); check_data('benchmark')
        if args.command=='prepare':prepare(wd)
        elif args.command=='acquire':acquire(wd)
        elif args.command=='classical':classical(args.split,wd)
        elif args.command=='neural':neural(args.split,wd)
        elif args.command=='sure':sure_terms(wd)
        elif args.command=='oracle':oracle(wd)
        elif args.command=='train':train(args.lambda_iqa,wd)
        elif args.command=='evaluate':equal(args.split);evaluate(args.split,wd)
        elif args.command=='views':views(args.method,args.split)
        elif args.command=='lock':lock(wd)
        elif args.command=='holdout':holdout(wd)
        elif args.command=='pilot-review':retrospective_pilots()
    finally:wd.close()

if __name__=='__main__':main()
