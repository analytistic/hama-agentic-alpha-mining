# HAMA runtime

`src/hama` contains the runtime for Harness-Augmented Agentic Alpha Mining.

One call to `hama.rollout(...)` is one complete paper-defined trajectory. At
each state it performs skill loading, query-parameterized memory loading,
candidate analysis, and one atomic factor-pool edit. The environment returns
the next state and marginal IC reward, after which the same context repeats the
three actions until the configured horizon. `hama.Agent` enforces an upper
bound of 50 model-completion steps for the whole trajectory.

```python
from hama import Agent, Harness, Model, rollout

model = Model(model="your-model", base_url="http://localhost:8000/v1")
agent = Agent(model=model, system_prompt="Mine one factor.", harness=harness)
run = rollout(task, environment, agent)
```

For authenticated endpoints, place the secret in the project-level `.env`:

```dotenv
OPENAI_API_KEY=your-key
OPENAI_BASE_URL=https://api.openai.com/v1
```

`Model` loads this file during initialization. Existing process environment
variables take precedence over values in `.env`.

## Qlib reward

`QlibFactorEvaluator` evaluates every expression on one immutable data split.
The fixed combination procedure cross-sectionally z-scores each factor and
averages the resulting signals. For a submitted edit, the environment computes

```text
reward = IC(combined candidate pool) - IC(combined current pool)
```

An added or replacement factor is first checked against the comparison pool
using the maximum absolute mean daily correlation. A rejected edit leaves the
state unchanged and receives zero reward.

## Training

Training starts from the configured factor pool. Each round freezes the current
harness and collects one trajectory; its final factor pool initializes the next
round. Advantages use return-to-go minus a per-step historical EMA baseline.
Leave-one-component-out prefill scores attribute each transition to the invoked
skill and memory entries. A conflict-aware semantic-gradient engine proposes
atomic revisions, and a separate edit optimizer updates the selected harness
parameters. Evaluation resets to a configured initial pool and never updates
the harness.

Copy and adjust `examples/config.json`, especially the Qlib provider
path and model name, then run:

```bash
uv sync
uv run hama train examples/config.json
uv run hama evaluate examples/config.json \
  --harness examples/hama-output/harness-final \
  --split test
uv run hama report hama-training-report.html examples/hama-output
```

Training checkpoints, the final harness, isolated rollout workspaces, and
evaluation summaries are written below the configured `output_dir`. Evaluation
uses a frozen harness and never invokes the semantic-gradient or edit engines.
Each training run also creates `harness-repo/`: its initial commit stores the
starting harness, and every effective optimizer update becomes one subsequent
commit with round and parameter metadata. Recent commit history is included in
the next optimizer prompt.

Harness parameters use a directory layout. Skill descriptions are exposed from
`skills/manifest.json`, while each strategy is loaded from the corresponding
`skills/<skill-id>/SKILL.md`; memory entries live in `memory.json`.

The report command creates one self-contained HTML file showing every
transition, factor-pool edit, reward, return, EMA baseline, advantage,
leave-one-component-out attribution, semantic proposal, applied text diff, and
the final test metrics. It needs no server and can combine consecutive run
directories. TensorBoard logging remains optional and is disabled by default.

The training CLI uses `HarnessEvolutionOptimizer`: one round may update an
existing parameter, add or replace a complete skill or memory entry, or remove
a harmful duplicate. Thus the cardinalities of both the skill library and the
memory bank can evolve. The example harness starts with several routed skills;
outcome-grounded memories are accumulated from rollout evidence.
