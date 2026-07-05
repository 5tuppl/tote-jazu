import sys, json, os, time, random, torch
import tote_grpo as T
SEED=0
random.seed(SEED); torch.manual_seed(SEED)
pairs=T.make_corpus()
Vs=T.Vocab([p[0] for p in pairs]); Vt=T.Vocab([p[1] for p in pairs])
tr,te=T.split(pairs)

CHUNK=int(sys.argv[1]) if len(sys.argv)>1 else 300
ST="grpo_state.pt"; CV="grpo_curve.json"

m=T.Seq2Seq(len(Vs),len(Vt))
ref=T.Seq2Seq(len(Vs),len(Vt)); ref.load_state_dict(torch.load("sft_light.pt"))
for p in ref.parameters(): p.requires_grad=False
ref.eval()

if os.path.exists(ST):
    s=torch.load(ST); m.load_state_dict(s["model"]); done=s["done"]
    curve=json.load(open(CV))
    # restore rng so sampling continues sensibly
    torch.set_rng_state(s["rng"]); random.setstate(eval(s["pyrng"]))
else:
    m.load_state_dict(torch.load("sft_light.pt")); done=0
    em0,ce0=T.evaluate(m,te,Vs,Vt)
    curve=[{"step":0,"exact":em0,"cer":ce0,"reward":None}]
    print(f"init pre-GRPO exact={em0:.3f} cer={ce0:.3f}")

TOTAL=1500
run=min(CHUNK, TOTAL-done)
t=time.time()
# run `run` steps; log every 50 within grpo by offsetting step numbers
m,c=T.grpo(m,tr,te[:30],Vs,Vt,steps=run,ref=ref,beta=0.04,log_every=50)
for row in c:
    row["step"]+=done
    curve.append(row)
done+=run
torch.save({"model":m.state_dict(),"done":done,
            "rng":torch.get_rng_state(),"pyrng":repr(random.getstate())},ST)
json.dump(curve,open(CV,"w"))
em,ce=T.evaluate(m,te,Vs,Vt)
print(f"chunk {run} steps {time.time()-t:.0f}s  done={done}/{TOTAL}  test exact={em:.3f} cer={ce:.3f}")
if done>=TOTAL:
    torch.save(m.state_dict(),"grpo.pt")
    json.dump({"post":{"exact":em,"cer":ce}},open("m_grpo_final.json","w"))
    print("GRPO COMPLETE")
