IMAGE ?= auditgym

.PHONY: check test docker-build docker-check demo export eval

check:
	python -m auditgym check

test:
	python -m pytest -q

demo:
	python -m auditgym demo --seed 2004 --preset interacting

export:
	python -m auditgym export --out episodes

eval:
	python -m auditgym eval --split train,dev

docker-build:
	docker build -t $(IMAGE) .

docker-check: docker-build
	docker run --rm --network none --cpus 2 --memory 4g $(IMAGE)
