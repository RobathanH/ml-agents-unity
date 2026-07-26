# Unity RL on Lambda Cloud — multi-project run management

One PC, one Lambda account, several independent Unity RL training projects,
each driven by its own agent, each launching and monitoring its own cloud
instances at its own times. **There is no central orchestrator.** Coordination
is entirely by naming convention and disjoint ownership, so an agent that
follows the rules below cannot damage another project's run even if it never
learns that project exists.

Read this before launching anything. Details that live better in code are
pointed at, not duplicated.

---

## 1. The project tag is the ownership key

`Project` = the run-id with its trailing `_NNN` stripped:
`CrawlerSumoEGNN_014` -> `CrawlerSumoEGNN`. Everything derives from it:

| Resource | Name | Owner |
|---|---|---|
| Lambda instance | `unity-rl-<Project>` | one agent |
| Staging run dirs | `results/cloud_staging/<Project>_NNN/` | one agent |
| Prune quota | the `<Project>` group, capped independently | one agent |
| Management journal | `results/cloud_staging/management_<Project>.md` | one agent |
| Git branch | one per project (e.g. `crawler-sumo`) | one agent |

**The rule: never act on a resource that does not carry your project tag.**
Concretely — never terminate, ssh into, pull from, or sync from an instance
whose name is not `unity-rl-<YourProject>`. `cloud_status.ps1 -Project <p>`
tags every instance `[MINE]` or `[OTHER PROJECT - do not touch]` and refuses
to ssh into the latter. The Lambda API has no per-project scoping and the
account is shared, so this convention is the only thing standing between your
cleanup step and someone else's 20-hour run.

## 2. One git worktree per project — do this first

`D:\UnityRL\ml-agents` is a **single git checkout**. If two agents work there
on different branches, `git checkout` by one silently deletes the other's
source and Unity assets mid-run. This is the single most destructive failure
mode in this setup and it has no runtime warning.

```powershell
git -C D:\UnityRL\ml-agents worktree add D:\UnityRL\ml-agents-<project> <branch>
```

Each worktree then owns its own `Project/` (so its own Unity `Library/` and
its own editor lock), `envs/`, `config/`, and `results/cloud_staging/`. Local
scripts resolve paths from their own location, so each worktree stages into
its own root automatically. Cost: a few GB and one slow first Unity import per
worktree. Pay it.

If you genuinely must share one tree, then every agent stays on one branch and
nobody switches branches while another agent is active. Prefer the worktree.

## 3. Launching

```powershell
.\scripts\lambda\cloud_train.ps1 -RunId <Project>_NNN -TimeLimitHours 24 `
    -Branch <branch> -Config config/ppo/<X>.yaml `
    -EnvBin envs/<X>_Multi_linux/<X>.x86_64 -BuildTgz envs/<X>_Multi_linux.tgz
```

`-Config`/`-EnvBin` are relative to the repo on the instance and default to
CrawlerSumo, so existing sumo invocations are unchanged. The script launches,
provisions, uploads the build, starts training in tmux, and arms the watchdog.

Its **preflight** refuses to launch when `unity-rl-<Project>` already exists
(one instance per project), and prints other projects' live instances with
their combined $/hr without blocking. `-AllowConcurrent` overrides, deliberately.

Instance-side details are in the scripts and don't need repeating here:
`setup_instance.sh` (Ubuntu 22.04 / py3.10, `torch==2.8.0`, `setuptools<81`,
xvfb — Unity Linux players SIGSEGV with no display server even under
`-nographics`), `launch_training.sh` (`CONFIG=` / `ENV_BIN=` env overrides),
`watchdog.sh` (SIGINT at limit-grace so the final checkpoint exports, then
self-terminates via the API — only ever the instance id it was given).

## 4. Money

Launching or extending costs real money and needs the user's go-ahead. Judge
it on the **account total, not your own instance**: the user pays the sum
across all projects, and `cloud_status.ps1` prints the combined `$/hr` at the
top. Quote that figure when you ask. gpu_1x_a10 is ~$0.75/hr; a 24h run is
~$18-31 depending on type.

The watchdog is the cost fail-safe — it terminates on schedule even if your
session dies. Never launch without one armed.

## 5. Data safety

Checkpoints only exist on the instance until synced. The instance is destroyed
on schedule. Therefore:

- `sync_cloud_staging.ps1` mirrors artifacts into `results/cloud_staging/`.
  `-Project <p>` restricts the pull to your own instances; omit it (as the
  durable scheduled task does) to cover every active instance.
- **Pruning always groups run dirs by project and applies `-MaxGB` to each
  group independently**, so one project's growth can never evict another
  project's only copy of a checkpoint. `_misc` (gifs, eval, loose logs) is
  file-pruned but never deleted wholesale.
- Concurrent syncs are serialized per staging root by a `sync.pid` file, not
  by command-line matching — every worktree's root is called `cloud_staging`,
  so a command-line match would kill another project's live sync.
- Staging is **deletable space**. Promote anything worth keeping (e.g. to
  `Project/Assets/<Project>/Models/`) before the pruner reaches it.

The durable Windows scheduled task `UnityRL_CloudStagingSync` is the fail-safe
that survives session death. Registering scheduled tasks is a user action —
ask, don't attempt it. Run 006 lost ~13M steps of checkpoints to a
session-bound sync loop dying with its session; that is why this exists.

## 6. Local serialization

Headless Unity (`-batchmode -executeMethod`) takes an exclusive project lock.
A second editor on the same `Project/` **crashes** rather than waiting. With
one worktree per project this is already solved; within a project, serialize
your own builds by polling first:

```powershell
while (Get-Process Unity -ErrorAction SilentlyContinue) { Start-Sleep 5 }
```

Local GPU eval and GIF capture are not exclusive but do contend — expect
slower tournaments when another project is also evaluating.

## 7. The management journal

Each project keeps `results/cloud_staging/management_<Project>.md` as its
durable state: the user's standing directive, guardrails, the pre-registered
judgment battery and decision rules, a run log, and a dated judgment history.

It exists so a fresh session can pick the project up cold, and so decisions
are pre-registered rather than rationalized after seeing results. Append a
dated entry at every judgment; never rewrite history. It is gitignored (under
`/results`), so it is local-only — anything another machine needs belongs in
the repo.

## 8. Conventions worth inheriting

- **PowerShell 5.1 files must be pure ASCII.** A BOM-less UTF-8 em-dash
  decodes as cp1252 `0x94`, a smart closing quote, which terminates a string
  literal early and fails the parse of the *entire* script. House style is
  `--`. Never use `Set-Content -Encoding utf8` on a Unity asset either: it
  writes a BOM that corrupts the YAML header, and `.gitattributes` marks those
  files binary so the diff will not show you the damage.
- **Rebuild every player after an observation-shape change.** Builds serialize
  the prefab; ML-Agents logs no mismatch and will quietly feed a policy the
  wrong observation width. Trainer, eval and viewer builds all need it.
- **Eval must match training physics.** Standalone builds use env-param
  defaults; pass `--env-param name=value` (or `-EnvParams`) or the results are
  meaningless.
- **Commit config changes with rationale before launching.** The instance
  clones the branch from GitHub, so unpushed work is invisible to it.

## 9. New-project checklist

1. `git worktree add` a directory for the project's branch.
2. Pick a `Project` tag; name run-ids `<Project>_NNN`.
3. Add the behavior's tags to `BEHAVIOR_TAGS` in `scripts/read_run_stats.py`.
4. Create `results/cloud_staging/management_<Project>.md` with the directive,
   guardrails, judgment battery and decision rules.
5. Build the Linux trainer tgz; launch with `-RunId`, `-Branch`, `-Config`,
   `-EnvBin`, `-BuildTgz`.
6. Verify the preflight showed your instance as the only one for your tag, and
   report the account-wide burn rate to the user.
