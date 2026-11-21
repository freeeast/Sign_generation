# Sign motion research

The active mission is to train a shared model on real sign motion pseudo-labels and generate new articulated SMPL-X motion from noise. Read [the objective](docs/RESEARCH_OBJECTIVE.md) and [Codex handoff](docs/CODEX_HANDOFF.md) before changing the experiment.

- Use `experiments/generative_motion/` for new source, configs and aggregate results. Write datasets, checkpoints, generated motion and video to `work/`. The Stage 5–10 and SharedMotion directories are historical, with entries documented in [project history](docs/PROJECT_HISTORY.md).
- Keep generation as the primary acceptance criterion. Per-video reconstruction, nearest-neighbor retrieval and history-conditioned prediction may serve as comparisons. The unconditional sampler takes a trained checkpoint, seed and length without a reference video, pose prefix, sentence or test motion.
- Preserve SMPL-X local joint order and hand-mean conventions. Reuse the bilateral 43-joint rotation helpers. Keep recording sources disjoint across official splits. Validate on validation and reserve test for a locked comparison.
- Keep licensed data and private models off GitHub. Recover permitted assets into `work/` using the documented source and hash. Never commit credentials, source videos, NSA archives, model templates or experiment binaries. Record checkpoint hash, data provenance, seed, commit and config in the run log.
- Run `python scripts/check_handoff.py` to verify historical entries and use `--smoke` after restoring the archived GRU. See the [generator README](experiments/generative_motion/README.md) for the tested train, sample, evaluation and FK commands.
- Report actual run status and keep planned milestones explicit. Valid rotation matrices, distribution statistics and linguistic correctness require separate evidence. Keep reports and raw results for failed experiments.
