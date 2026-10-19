STARTER_SHA := 4b6af065cf5ce0db88f2cac4546a9ac5c6147f26

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
TRAIN := --model pca_ridge --stride 6 --n-val 200

# DIII-D train shots -> per-shot feature npz
cache:
	$(PY) fusion_eq.data /Volumes/PortableSSD/datasets/fusion-equilibrium/data/diii_d_train $(RUNS)/cache

# the DIII-D ridge, pickled to $(RUNS)/models
train:
	$(PY) fusion_eq.evaluate $(TRAIN) --kw '{"ip_interact": true}' --tag ridge32-ipx-clip-v200 --save

weights:
	cd weights && shasum -a 256 -c $(CURDIR)/docs/weights.sha256
	mkdir -p $(RUNS)/models && cp weights/*.pkl $(RUNS)/models/

# MAST physics predictions (CPU, resumable) -> $(RUNS)/mast/preds
mast:
	$(PY) fusion_eq.mast_batch --workers 8
	$(PY) fusion_eq.mast_post

# build + validate; add ARGS="--repo <user>/fusion-eq-predictions" to push and write the pointer zip
submission:
	$(PY) fusion_eq.make_submission --max-shots 0 --out $(RUNS)/submission $(ARGS)
