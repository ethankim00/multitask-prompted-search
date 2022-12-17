python ./src/mps/datasets/dpr/download_data.py --resource data.retriever.nq --output_dir ./data/dpr
python ./src/mps/datasets/dpr/download_data.py --resource data.retriever.trivia --output_dir ./data/dpr
python ./src/mps/datasets/dpr/download_data.py --resource data.retriever.webq-train --output_dir ./data/dpr
python ./src/mps/datasets/dpr/download_data.py --resource data.retriever.curatedtrec --output_dir ./data/dpr

#python OpenMatch/scripts/nq-dpr/build_train.py --input ./data/dpr/downloads/data/retriever/curatedtrec-train.json --output ./data/pretraining/curated_trecl.jsonl