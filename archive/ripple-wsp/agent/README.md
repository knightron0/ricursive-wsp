# Agent loop (Ripple editing via a persistent GPU kernel); run mapping inferred from file and log timestamps, not recorded
- `agent.py` + `vm.py` (`code/infra/vm.py`): toy1, c1_spanish, c1_spanish_r2, c1_spanish_r3, c1_pirate_r1.
- `agent2.py` + `vm2.py`: same loop pointed at a second GPU machine; c2_fact_r1.
- `agent4.py` + `vm4.py`: c3_caps_r1, c3_caps_r1b (and its misfire); waits out VM-link failures.
- `vm3.py`: VM driver for a third machine (identical to vm2.py except the sandbox ID); no agent3.py survives.
