"""Verify host watchdog signaling and real Adam/MUSIQ next-update recovery."""
import tempfile
import time
from pathlib import Path
import research_session_v3 as r
from denoise_research.session_guard import PowerStatus

def main():
    torch=r.setup_torch()
    with tempfile.TemporaryDirectory() as tmp:
        calls=[]
        def reader():
            calls.append(time.monotonic())
            return PowerStatus(80 if len(calls)==1 else 14,True,True,'simulation')
        wd=r.Watchdog(tmp,.05,reader)
        assert wd.stop.wait(2), 'Independent watchdog failed to signal'
        assert len(calls)>=2 and max(b-a for a,b in zip(calls,calls[1:]))<1
        wd.close()
        assert r.read(Path(tmp)/'watchdog.json')['stop']
    scorer=r.model_metric('musiq')
    torch.manual_seed(20261014)
    z=torch.rand(16,4,3,32,32)
    gram=torch.eye(4)*.002;linear=torch.ones(4)*.001;cal={'mean':20.,'std':5.}
    logits=torch.zeros(4,requires_grad=True);opt=torch.optim.Adam([logits],lr=.08)
    r.optimization_step(logits,opt,gram,linear,z,scorer,cal,.01,0)
    checkpoint=r.OUT/'safety_recovery.pt'
    r.atomic_torch(checkpoint,dict(logits=logits.detach(),optimizer=opt.state_dict(),torch_rng=torch.get_rng_state(),next_step=1))
    r.optimization_step(logits,opt,gram,linear,z,scorer,cal,.01,1)
    expected=logits.detach().clone()
    saved=torch.load(checkpoint,weights_only=True)
    resumed=saved['logits'].clone().requires_grad_();opt2=torch.optim.Adam([resumed],lr=.08)
    opt2.load_state_dict(saved['optimizer']);torch.set_rng_state(saved['torch_rng'])
    r.optimization_step(resumed,opt2,gram,linear,z,scorer,cal,.01,saved['next_step'])
    error=float((resumed.detach()-expected).abs().max())
    assert error==0, error
    assert all(not p.requires_grad and p.grad is None for p in scorer.parameters())
    r.write(r.OUT/'safety_verification.json',dict(independent_low_battery_signal=True,poll_test_interval=.05,production_poll_interval=30,
        physical_power=str(r.read_power_status()),actual_musiq_adam_resume_max_error=error,scorer_frozen=True,checkpoint_sha256=r.sha(checkpoint)))
    print('Watchdog and actual MUSIQ/Adam recovery verified',flush=True)

if __name__=='__main__':main()
