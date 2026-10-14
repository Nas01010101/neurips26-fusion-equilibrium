STARTER_SHA := 5b467988d2e5f28e20be490a926d0a71649de28d

.PHONY: setup vendor test data-train score cache train mast submission weights
setup:
	uv sync --group dev

vendor:
	test -d vendor/starter || GIT_LFS_SKIP_SMUDGE=1 git clone -q https://github.com/Sophelio/fusion-equilibrium-challenge-starter vendor/starter
	git -C vendor/starter checkout -q $(STARTER_SHA)

test:
	uv run pytest -q

# DIII-D train split to the SSD (internal disk has ~30 GB free)
data-train:
	uvx --from huggingface_hub hf download Sophelio/fusion-equilibrium-challenge \
	  --repo-type dataset --include "data/diii_d_train/*" \
	  --local-dir /Volumes/PortableSSD/datasets/fusion-equilibrium

# Official scorer on held-out train shots: make score ARGS="--n-shots 20 --skip 7000"
score:
	cd vendor/starter && uv run python local_score.py $(ARGS)

# --- reproduction (docs/REPRODUCE.md) ---
RUNS ?= /Volumes/PortableSSD/scratch/fusion-eq-runs
PY := PYTHONPATH=src uv run python -m
TRAIN := --model pca_mlp --stride 6 --n-val 200
KW := "width":2048,"epochs":80,"lr":1e-3,"clip":1.0

# DIII-D train shots -> per-shot feature npz
cache:
	$(PY) fusion_eq.data /Volumes/PortableSSD/datasets/fusion-equilibrium/data/diii_d_train $(RUNS)/cache

# the three ensemble members (seeds 0, 1, 2), pickled to $(RUNS)/models
train:
	$(PY) fusion_eq.evaluate $(TRAIN) --kw '{$(KW)}' --tag mlp64-w2048-lr1e3-clip-v200 --save
	$(PY) fusion_eq.evaluate $(TRAIN) --kw '{$(KW),"seed":1}' --tag mlp64-w2048-lr1e3-clip-s1-v200 --save
	$(PY) fusion_eq.evaluate $(TRAIN) --kw '{$(KW),"seed":2}' --tag mlp64-w2048-lr1e3-clip-s2-v200 --save

weights:
	cd $(RUNS)/models && shasum -a 256 -c $(CURDIR)/docs/weights.sha256

# MAST physics predictions (CPU, resumable) -> $(RUNS)/mast/preds
mast:
	$(PY) fusion_eq.mast_batch --workers 8

# build + validate; add ARGS="--repo <user>/fusion-eq-predictions" to push and write the pointer zip
submission:
	$(PY) fusion_eq.make_submission --max-shots 0 --out $(RUNS)/submission $(ARGS)
