# 32×32 carbon benchmark

One 32,768-atom structure from `mp0b3_no_cool`, with three timing repeats.
[Input metadata](../../inputs/carbon_32768/metadata.json) records the source and settings.

PoreTorch runs on `phy-tcm-pc80` with the RTX 5090. PoreBlazer runs on
`fast9` (`phy-tcm-pc79`). Both use one CPU thread. PoreTorch uses warm starts.

Attach to the sessions:

```bash
tmux attach -t poretorch-32768
ssh -t fast9 'tmux attach -t poreblazer-32768'
```

Each program writes `status.json`, `metadata.json`, `raw_timings.json` and
`timings.csv` in its own output directory. The launchers write logs and exit
codes here. Status changes to `complete` or `failed` when the job ends.

The `poretorch-32768-report` tmux session checks for completed results every
30 seconds and updates the main README and benchmark plot. Pending values
are shown as `-`. It stops after both benchmark jobs have ended.
