train_pipeline:
	python3 train.py

download_embedding_model:
	huggingface-cli download dunzhang/stella_en_1.5B_v5 --local-dir model_artifacts/$(basename dunzhang/stella_en_1.5B_v5) --revision 7816d43c4efd2fead216afbb7522d2093b44b16b

all:
	download_embedding_model
	train_pipeline