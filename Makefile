PYTHON ?= python3

.PHONY: install test docker-build docker-test setup-data

install:
	$(PYTHON) -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
	$(PYTHON) -m pip install facenet-pytorch --no-deps
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest tests/ -v

docker-build:
	docker build -t lfw-verifier .

docker-test: docker-build
	@echo "\n--- smoke test ---"
	docker run --rm lfw-verifier --help
	@echo "\n--- single pair (SAME) ---"
	docker run --rm \
		-v "$(CURDIR)/samples:/app/samples" \
		-v "$(CURDIR)/configs:/app/configs" \
		lfw-verifier --config configs/m3.yaml \
		--img1 samples/Aaron_Peirsol/Aaron_Peirsol_0001.jpg \
		--img2 samples/Aaron_Peirsol/Aaron_Peirsol_0002.jpg
	@echo "\n--- batch (4 sample pairs) ---"
	docker run --rm \
		-v "$(CURDIR)/samples:/app/samples" \
		-v "$(CURDIR)/configs:/app/configs" \
		lfw-verifier --config configs/m3.yaml \
		--pairs samples/pairs.csv

setup-data:
	$(PYTHON) scripts/ingest_lfw.py --config configs/m1.yaml
	$(PYTHON) scripts/make_pairs.py --config configs/m2.yaml --version v1
	$(PYTHON) scripts/make_pairs.py --config configs/m2.yaml --version v2
