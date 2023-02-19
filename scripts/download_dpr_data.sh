for dataset in data.retriever.nq data.retriever.trivia data.retriever.webq-train data.retriever.curatedtrec
    do
    python ./src/mps/datasets/dpr/download_data.py --resource $dataset --output_dir ./data/dpr
    done