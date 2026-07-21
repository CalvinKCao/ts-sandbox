"""Full split-safe oracle-coarse refinement experiment and held-out discriminator."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from experiments.ordinal_patch_refinement_killtest import smoke
from experiments.ordinal_patch_refinement_killtest.nonoverlap_protocol import build_protocol
from models.diffusion_tsf.dit import FactorizedDiT
from models.diffusion_tsf.ordinal_window_norm import ordinal_encode, ordinal_decode
from models.diffusion_tsf.preprocessing import TimeSeriesTo2D
from models.diffusion_tsf.train_multivariate_pipeline import load_dataset
from utils.eval_mmpd_gaussian_anchor import load_tsf_pack_pool
from utils.eval_discriminator_texture_staged_vs_mmpd import InvertedSliceDiscriminator, binary_auroc

def main():
 p=argparse.ArgumentParser(); p.add_argument('--dataset',default='ETTh1'); p.add_argument('--resolution',type=int,choices=[256,512],default=256); p.add_argument('--epochs',type=int,default=20); p.add_argument('--seed',type=int,default=42); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
 H=16; R=a.resolution; P=R//16; device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); smoke.set_seed(a.seed)
 V=7 if a.dataset=='ETTh1' else 8; protocol=build_protocol(a.dataset,V); pool_by={}
 for split in ('train','val','test'): pool_by[split]=load_tsf_pack_pool(a.dataset,list(range(V)),lookback=96,horizon=H,train_stride=1,test_stride=4,pack_splits=[split])[0]
 _,_,_,stats=load_dataset(a.dataset,list(range(V)),lookback=96,horizon=H,stride=1,test_stride=4,use_ordinal_window_norm=True); ladder=stats['ordinal_ladder']; rank=ladder.rank_max_per_variate().float().to(device)
 def mats(split,limit=None):
  xs=[]; cs=[]; ys=[]; raw=[]; chosen=protocol['splits'][split]['indices'];
  for wi in chosen[:limit]:
   past,fut=pool_by[split][wi]; past,fut=past[None].to(device),fut[None,...,:H].to(device); po,fo,lb,shift=ordinal_encode(past,fut,ladder=ladder,apply_ood_shift=True,causal_only=True)
   target=smoke._cdf_from_values(fo,rank,R).repeat_interleave(P,-1); coarse=smoke._cdf_from_values(fo,rank,16); up=F.interpolate(coarse,size=(R,R),mode='nearest'); hist=smoke._cdf_from_values(po[...,-16:],rank,R).repeat_interleave(P,-1)
   for v in range(V):
    bins=TimeSeriesTo2D.bin_indices_from_cdf(coarse[:,v:v+1])[0,0].long()
    for t in range(H):
     r,c=int(bins[t])*P,t*P; x,valid=smoke._extract_block(up[0,v:v+1],r,c,P); y,_=smoke._extract_block(target[0,v:v+1],r,c,P); h,_=smoke._extract_block(hist[0,v:v+1],r,0,P); rows=torch.arange(P,device=device).view(1,P,1); b=(rows==0).float().expand(1,P,P); tp=torch.full_like(x,t/15); vp=torch.linspace(r/R,(r+P-1)/R,P,device=device).view(1,P,1).expand_as(x); xs.append(torch.cat([x,b,valid,tp,vp]));cs.append(h);ys.append(y);raw.append((wi,v,t,r,c,past[0,v].cpu(),fut[0,v].cpu(),up[0,v].cpu(),target[0,v].cpu()))
  return torch.stack(xs),torch.stack(cs),torch.stack(ys),raw
 tx,tc,ty,_=mats('train'); vx,vc,vy,_=mats('val'); ex,ec,ey,meta=mats('test')
 # Grid from latest vertical-dual recipe; microbatch keeps A100 memory bounded.
 grid=[(lr,b) for lr in (5e-5,2.41e-4,1.5e-3) for b in (512,1024,2048)]; best=None
 for lr,eff in grid:
  m=FactorizedDiT(5,1,1,P,(8,8),384,8,6,context_dim=1).to(device); opt=torch.optim.AdamW(m.parameters(),lr=lr); micro=min(64,len(tx)); acc=max(1,eff//micro)
  for e in range(min(4,a.epochs)):
   order=torch.randperm(len(tx),device=device); opt.zero_grad()
   for j,i in enumerate(order.split(micro)):
    loss=F.binary_cross_entropy_with_logits(m(tx[i],torch.zeros(len(i),device=device),tc[i]),ty[i]); (loss/acc).backward()
    if (j+1)%acc==0: opt.step();opt.zero_grad()
  with torch.no_grad(): val=float(F.binary_cross_entropy_with_logits(m(vx,torch.zeros(len(vx),device=device),vc),vy))
  if best is None or val<best[0]: best=(val,lr,eff,m)
 _,lr,eff,m=best
 opt=torch.optim.AdamW(m.parameters(),lr=lr); loader=DataLoader(TensorDataset(tx,tc,ty),batch_size=64,shuffle=True)
 for _ in range(a.epochs):
  for x,c,y in loader: opt.zero_grad();loss=F.binary_cross_entropy_with_logits(m(x.to(device),torch.zeros(len(x),device=device),c.to(device)),y.to(device));loss.backward();opt.step()
 with torch.no_grad(): probs=torch.sigmoid(m(ex,torch.zeros(len(ex),device=device),ec)); patches,_=smoke._project_monotone(probs)
 # Persist per-patch test tensors; discriminator consumes their decoded patch traces.
 z=patches.sum(2).squeeze(1).cpu().numpy(); gt=ey.sum(2).squeeze(1).cpu().numpy(); naive=ex[:,0].sum(1).cpu().numpy()
 a.output.mkdir(parents=True,exist_ok=True); np.savez_compressed(a.output/'heldout_patches.npz',refined=z,gt=gt,naive=naive,inputs=ex.cpu().numpy(),targets=ey.cpu().numpy()); json.dump({'dataset':a.dataset,'resolution':R,'patch':P,'train_patches':len(tx),'val_patches':len(vx),'test_patches':len(ex),'best_lr':lr,'effective_batch':eff},open(a.output/'manifest.json','w'),indent=2)
 print(json.dumps({'output':str(a.output),'train':len(tx),'val':len(vx),'test':len(ex),'lr':lr,'batch':eff}))
if __name__=='__main__': main()
