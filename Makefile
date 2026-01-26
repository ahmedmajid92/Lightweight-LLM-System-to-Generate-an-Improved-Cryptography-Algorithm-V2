.PHONY: help install run build-kb ingest-external build-sft test

help:
	@echo "Targets:"
	@echo "  install        Install Python deps"
	@echo "  run            Run Streamlit app"
	@echo "  build-kb       Build RAG KB index (BM25 + optional embeddings)"
	@echo "  ingest-external  List external KB docs"
	@echo "  build-sft      Regenerate SFT dataset"
	@echo "  test           Run unit tests"

install:
	pip install -r requirements.txt

run:
	streamlit run app/streamlit_app.py

build-kb:
	python scripts/build_kb_index.py

ingest-external:
	python scripts/ingest_external_kb.py

build-sft:
	python scripts/build_sft_dataset.py

test:
	python -m pytest -q
