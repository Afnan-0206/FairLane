.PHONY: all test exp-noisy-neighbor exp-heterogeneous clean

GO := "C:/Program Files/Go/bin/go.exe"
PYTHON := .venv/Scripts/python.exe

all: test

test:
	$(GO) test -v ./policy
	$(GO) test -v ./sim

exp-noisy-neighbor:
	$(GO) run ./cmd/exp_noisy_neighbor
	$(PYTHON) ./analysis/plot_noisy_neighbor.py
	$(PYTHON) ./analysis/plot_counters_over_time.py

exp-heterogeneous:
	$(GO) run ./cmd/exp_heterogeneous
	$(PYTHON) ./analysis/plot_heterogeneous.py

clean:
	rm -rf experiments/results/*.json experiments/results/*.png
