"""Debug: trace unit actions + shed for the first N steps of a probe self-play."""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "probes"))

from _engine import import_engine  # noqa: E402
from probe_core import make_agent, PRESETS  # noqa: E402

spec = sys.argv[1] if len(sys.argv) > 1 else "GooseRush"
n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 30
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 1000

ke = import_engine()
env = ke.make("kaggriculture", configuration={"episodeSteps": 720, "seed": seed}, debug=True)
params = json.loads(spec) if spec.startswith("{") else PRESETS[spec]
agents = [make_agent(params), make_agent(dict(params))]
env.reset()

for _ in range(n_steps):
    obs0 = env.state[0].observation
    step = obs0.step
    acts = []
    for i in range(2):
        acts.append(agents[i](env.state[i].observation))
    farm = obs0.farms[0]
    priv = obs0.private
    shed = {k: v for k, v in priv["shed"].items() if v}
    invs = [dict(x) for x in priv["inventories"]]
    print(f"s{step:>3} farmer={farm['farmer']} hands={farm['hands']}")
    print(f"     shed={shed} invs={invs} money={farm['money']}")
    print(f"     acts={json.dumps(acts[0])}")
    env.step(acts)
    if env.done:
        break
