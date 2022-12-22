for dataset in trivia nq webq curated_trec
    do
    python OpenMatch/scripts/nq-dpr/build_train.py --input ./data/dpr/downloads/data/retriever/$dataset-train.json --output ./data/pretraining/trivia-qa.jsonl --num_workers 8 --mp_chunk_size 250
    done