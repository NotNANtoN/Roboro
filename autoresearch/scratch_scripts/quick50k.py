"""50k-step screening — better predictor of 100k behavior than 20k."""
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
from prepare import TASKS, evaluate, normalize_return

STEPS = 50_000
SEED = 42

HD = int(os.environ.get("HD", "128"))
NL = int(os.environ.get("NL", "2"))
ACT = os.environ.get("ACT", "relu")
LN = os.environ.get("LN", "1") == "1"
LR = float(os.environ.get("LR", "1e-3"))
GAMMA = float(os.environ.get("GAMMA", "0.99"))
BS = int(os.environ.get("BS", "256"))
WU = int(os.environ.get("WU", "500"))
AD = int(os.environ.get("AD", "2"))
TAU = float(os.environ.get("TAU", "0.005"))
ALPHA = float(os.environ.get("ALPHA", "0.1"))
GC = float(os.environ.get("GC", "0"))


class Buf:
    def __init__(self, cap, od, ad, seed=None):
        self.o=np.zeros((cap,od),np.float32); self.a=np.zeros((cap,ad),np.float32)
        self.r=np.zeros(cap,np.float32); self.n=np.zeros((cap,od),np.float32)
        self.d=np.zeros(cap,np.bool_); self.p=self.s=0; self.c=cap
        self.rng=np.random.default_rng(seed)
    def add(self,o,a,r,n,d):
        i=self.p; self.o[i]=o;self.a[i]=a;self.r[i]=r;self.n[i]=n;self.d[i]=d
        self.p=(self.p+1)%self.c; self.s=min(self.s+1,self.c)
    def sample(self,n):
        i=self.rng.integers(0,self.s,size=n)
        return(torch.from_numpy(self.o[i]),torch.from_numpy(self.a[i]),
               torch.from_numpy(self.r[i]),torch.from_numpy(self.n[i]),
               torch.from_numpy(self.d[i]))
    def __len__(self): return self.s


def run(tn):
    task = TASKS[tn]; set_seed(SEED)
    env = gym.make(task.env_id)
    od=env.observation_space.shape[0]; ad=env.action_space.shape[0]
    al,ah = float(np.min(env.action_space.low)),float(np.max(env.action_space.high))

    actor = SquashedGaussianActor(obs_dim=od,action_dim=ad,action_low=al,action_high=ah,
                                   hidden_dim=HD,n_layers=NL,activation=ACT,use_layer_norm=LN)
    q1=ContinuousQCritic(feature_dim=od,action_dim=ad,hidden_dim=HD,n_layers=NL,activation=ACT,use_layer_norm=LN)
    q2=ContinuousQCritic(feature_dim=od,action_dim=ad,hidden_dim=HD,n_layers=NL,activation=ACT,use_layer_norm=LN)
    critic=TwinQCritic(q1,q2); ct=TargetNetwork(critic,mode="polyak",tau=TAU)
    buf=Buf(100_000,od,ad,seed=SEED)
    aopt=torch.optim.Adam(actor.parameters(),lr=LR)
    copt=torch.optim.Adam(critic.parameters(),lr=LR)
    la=nn.Parameter(torch.tensor(float(ALPHA)).log())
    alopt=torch.optim.Adam([la],lr=LR); te=-float(ad)

    obs,_=env.reset(seed=SEED)
    ot=torch.as_tensor(obs,dtype=torch.float32).unsqueeze(0)
    gs=0

    for step in range(1,STEPS+1):
        a_,_=actor.act(ot); anp=a_.squeeze(0).cpu().numpy()
        no,r,term,trunc,_=env.step(anp)
        buf.add(obs,anp,float(r),no,term)
        obs=no; ot=torch.as_tensor(obs,dtype=torch.float32).unsqueeze(0)
        if term or trunc:
            obs,_=env.reset(); ot=torch.as_tensor(obs,dtype=torch.float32).unsqueeze(0)

        if step>=WU and len(buf)>=BS:
            bo,ba,br,bn,bd=buf.sample(BS); alpha=la.exp(); gs+=1
            with torch.no_grad():
                na,nlp=actor(bn); tq=ct(bn,na)
                st=br+GAMMA*(~bd).float()*(tq-alpha*nlp)
            q1v,q2v=critic.both(bo,ba)
            cl=F.mse_loss(q1v,st)+F.mse_loss(q2v,st)
            copt.zero_grad(); cl.backward()
            if GC>0: nn.utils.clip_grad_norm_(critic.parameters(),GC)
            copt.step()
            if gs%AD==0:
                ap,lp=actor(bo); qp=critic(bo,ap)
                al_=(alpha.detach()*lp-qp).mean()
                aopt.zero_grad(); al_.backward(); aopt.step()
                all_=-(la.exp()*(lp.detach()+te)).mean()
                alopt.zero_grad(); all_.backward(); alopt.step()
            ct.update()

    def pfn(o):
        with torch.no_grad():
            return actor.act(torch.as_tensor(o,dtype=torch.float32).unsqueeze(0),deterministic=True)[0].squeeze(0).numpy()
    er=evaluate(task.env_id,pfn,10,task.eval_seed); env.close()
    return er


if __name__=="__main__":
    t0=time.time()
    hr=run("hopper"); wr=run("walker")
    el=time.time()-t0
    hn=max(0,hr/3500); wn=max(0,wr/5000); sc=(hn+wn)/2
    tag=os.environ.get("TAG","default")
    print(f"[{tag}] score={sc:.4f} hopper={hr:.1f}({hn:.3f}) walker={wr:.1f}({wn:.3f}) time={el:.1f}s")
