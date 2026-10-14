STARTER_SHA := 5b467988d2e5f28e20be490a926d0a71649de28d

.PHONY: setup vendor test data-train score
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
