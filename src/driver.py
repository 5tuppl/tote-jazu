import sys, json, time, random, copy, torch
import tote_grpo as T
SEED=0
def setup():
    random.seed(SEED); torch.manual_seed(SEED)
    pairs=T.make_corpus()
    Vs=T.Vocab([p[0] for p in pairs]); Vt=T.Vocab([p[1] for p in pairs])
    tr,te=T.split(pairs); return pairs,Vs,Vt,tr,te
phase=sys.argv[1]
pairs,Vs,Vt,tr,te=setup()

if phase in ("sft_light","sft_full"):
    steps=900 if phase=="sft_light" else 2500
    random.seed(SEED); torch.manual_seed(SEED)
    m=T.Seq2Seq(len(Vs),len(Vt))
    t=time.time(); m=T.sft(m,tr,Vs,Vt,steps=steps,lr=1e-3)
    em,ce=T.evaluate(m,te,Vs,Vt)
    fn="sft_light.pt" if phase=="sft_light" else "sft_full.pt"
    torch.save(m.state_dict(),fn)
    json.dump({phase:{"exact":em,"cer":ce,"steps":steps}},open(f"m_{phase}.json","w"))
    print(f"{phase} {time.time()-t:.0f}s  test exact={em:.3f} cer={ce:.3f}")

elif phase=="grpo":
    random.seed(SEED); torch.manual_seed(SEED)
    m=T.Seq2Seq(len(Vs),len(Vt)); m.load_state_dict(torch.load("sft_light.pt"))
    ref=T.Seq2Seq(len(Vs),len(Vt)); ref.load_state_dict(torch.load("sft_light.pt"))
    for p in ref.parameters(): p.requires_grad=False
    ref.eval()
    em0,ce0=T.evaluate(m,te,Vs,Vt)
    t=time.time()
    m,curve=T.grpo(m,tr,te[:30],Vs,Vt,steps=1500,ref=ref,beta=0.04,log_every=50)
    em1,ce1=T.evaluate(m,te,Vs,Vt)
    torch.save(m.state_dict(),"grpo.pt")
    json.dump({"pre":{"exact":em0,"cer":ce0},"post":{"exact":em1,"cer":ce1},
               "curve":curve},open("m_grpo.json","w"))
    print(f"GRPO {time.time()-t:.0f}s  pre exact={em0:.3f} cer={ce0:.3f} -> post exact={em1:.3f} cer={ce1:.3f}")

elif phase=="purerl":
    random.seed(SEED); torch.manual_seed(SEED)
    m=T.Seq2Seq(len(Vs),len(Vt))
    t=time.time()
    m,curve=T.grpo(m,tr,te[:30],Vs,Vt,steps=1200,ref=None,log_every=100)
    em,ce=T.evaluate(m,te,Vs,Vt)
    json.dump({"pure_rl":{"exact":em,"cer":ce},"curve":curve},open("m_purerl.json","w"))
    print(f"pureRL {time.time()-t:.0f}s  test exact={em:.3f} cer={ce:.3f}")
