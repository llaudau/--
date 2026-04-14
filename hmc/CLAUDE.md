# CLAUDE.md — project notes for Claude Code

## Python environment

All analysis scripts require numpy, pandas, matplotlib, scipy.
The venv is at:

```
/home/khw/Documents/Git_repository/qcd/data_analyze/.venv
```

Always run analysis scripts with:

```bash
/home/khw/Documents/Git_repository/qcd/data_analyze/.venv/bin/python analyze_*.py
```

or activate first:

```bash
source /home/khw/Documents/Git_repository/qcd/data_analyze/.venv/bin/activate
```

Never ask the user to install packages — the venv already has everything.

## Tianhe supercomputer

- Architecture: **ARM64 (aarch64)** — binaries compiled locally (x86) will NOT run on Tianhe
- Always rebuild on Tianhe after syncing source: `ssh tianhe "cd ~/wkh/hmc && make clean && make"`
- SLURM partition: `thcp1`
- 64 cores/node, 8 NUMA domains → 8 OMP threads per rank, 8 ranks per node
- `--cpu-bind=numa` is NOT supported on this cluster
- MPI compiler: `/usr/local/ompi/bin/mpicxx`

## Syncing

```bash
# local → Tianhe (source only)
rsync -avz --exclude-from=.rsyncignore . tianhe:~/wkh/hmc/

# Tianhe → local (results)
rsync -avz tianhe:~/wkh/hmc/results/ /home/khw/Documents/Git_repository/hmc/results/
```
