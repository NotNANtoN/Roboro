"""50k screening V2 — supports alpha_lr, target_entropy_scale, reward_scale."""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

from roboro.core.seed import set_seed
from roboro.critics.q import ContinuousQCritic, TwinQCritic
from roboro.critics.target import TargetNetwork
from roboro.actors.squashed_gaussian import SquashedGaussianActor
from prepare import TASKS, evaluate

STEPS = 50_000; SEED = 42; HD = int(os.environ.get("HD", "128"))
LR = float(os.environ.get("LR", "1e-3"))
ALPHA_LR = float(os.environ.get("ALR", os.environ.get("LR", "1e-3")))
TAU = float(os.environ.get("TAU", "0.005"))
ALPHA = float(os.environ.get("ALPHA", "0.1"))
TE_SCALE = float(os.environ.get("TES", "1.0"))  # target_entropy = -action_dim * TES
RS = float(os.environ.get("RS", "1.0"))  # reward scale
BS = int(os.environ.get("BS", "256"))
AD = int(os.environ.get("AD", "2"))

class Buf:
    def __init__(s,c,od,ad,seed=None):
        s.o=np.zeros((c,od),np.float32);s.a=np.zeros((c,ad),np.float32)
        s.r=np.zeros(c,np.float32);s.n=np.zeros((c,od),np.float32)
        s.d=np.zeros(c,np.bool_);s.p=s.s=0;s.c=c;s.rng=np.random.default_rng(seed)
    def add(s,o,a,r,n,d):
        i=s.p;s.o[i]=o;s.a[i]=a;s.r[i]=r;s.n[i]=n;s.d[i]=d
        s.p=(s.p+1)%s.c;s.s=min(s.s+1,s.c)
    def sample(s,n):
        i=s.rng.integers(0,s.s,size=n)
        return(torch.from_numpy(s.o[i]),torch.from_numpy(s.a[i]),
               torch.from_numpy(s.r[i]),torch.from_numpy(s.n[i]),torch.from_numpy(s.d[i]))
    def __len__(s): return s.s

def run(tn):
    task=TASKS[tn]; set_seed(SEED); env=gym.make(task.env_id)
    od=env.observation_space.shape[0]; ad=env.action_space.shape[0]
    al,ah=float(np.min(env.action_space.low)),float(np.max(env.action_space.high))
    actor=SquashedGaussianActor(obs_dim=od,action_dim=ad,action_low=al,action_high=ah,
                                 hidden_dim=HD,n_layers=2,activation="relu",use_layer_norm=True)
    q1=ContinuousQCritic(feature_dim=od,action_dim=ad,hidden_dim=HD,n_layers=2,activation="relu",use_layer_norm=True)
    q2=ContinuousQCritic(feature_dim=od,action_dim=ad,hidden_dim=HD,n_layers=2,activation="relu",use_layer_norm=True)
    critic=TwinQCritic(q1,q2); ct=TargetNetwork(critic,mode="polyak",tau=TAU)
    buf=Buf(100000,od,ad,seed=SEED)
    aopt=torch.optim.Adam(actor.parameters(),lr=LR)
    copt=torch.optim.Adam(critic.parameters(),lr=LR)
    la=nn.Parameter(torch.tensor(float(ALPHA)).log())
    alopt=torch.optim.Adam([la],lr=ALPHA_LR)
    te=-float(ad)*TE_SCALE
    obs,_=env.reset(seed=SEED)
    ot=torch.as_tensor(obs,dtype=torch.float32).unsqueeze(0); gs=0
    for step in range(1,STEPS+1):
        a_,_=actor.act(ot);anp=a_.squeeze(0).cpu().numpy()
        no,r,term,trunc,_=env.step(anp)
        buf.add(obs,anp,float(r)*RS,no,term)
        obs=no;ot=torch.as_tensor(obs,dtype=torch.float32).unsqueeze(0)
        if term or trunc: obs,_=env.reset();ot=torch.as_tensor(obs,dtype=torch.float32).unsqueeze(0)
        if step>=500 and len(buf)>=BS:
            bo,ba,br,bn,bd=buf.sample(BS);alpha=la.exp();gs+=1
            with torch.no_grad():
                na,nlp=actor(bn);tq=ct(bn,na);st=br+0.99*(~bd).float()*(tq-alpha*nlp)
            q1v,q2v=critic.both(bo,ba);cl=F.mse_loss(q1v,st)+F.mse_loss(q2v,st)
            copt.zero_grad();cl.backward();copt.step()
            if gs%AD==0:
                ap,lp=actor(bo);qp=critic(bo,ap);al_=(alpha.detach()*lp-qp).mean()
                aopt.zero_grad();al_.backward();aopt.step()
                all_=-(la.exp()*(lp.detach()+te)).mean();alopt.zero_grad();all_.backward();alopt.step()
            ct.update()
    def pfn(o):
        with torch.no_grad(): return actor.act(torch.as_tensor(o,dtype=torch.float32).unsqueeze(0),deterministic=True)[0].squeeze(0).numpy()
    return evaluate(task.env_id,pfn,10,task.eval_seed)

if __name__=="__main__":
    t0=time.time();hr=run("hopper");wr=run("walker");el=time.time()-t0
    hn=max(0,hr/3500);wn=max(0,wr/5000);sc=(hn+wn)/2
    tag=os.environ.get("TAG","default")
    print(f"[{tag}] score={sc:.4f} hopper={hr:.1f}({hn:.3f}) walker={wr:.1f}({wn:.3f}) time={el:.1f}s")
