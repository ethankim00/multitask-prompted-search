

OUTPUT_DIR=./models
TEMPERATURE=0.5

#  android english gaming gis mathematica physics programmers stats tex unix webmasters wordpress
# a
#fiqa arguana scidocs
for train_dataset in  webis-touche2020 dbpedia-entity arguana nq hotpotqa msmarco nfcorpus quora dbpedia scidocs fever climate-fever scifact touche, trec-covid fiqa
do
    python -m src.mps.eval.eval_transfer \
    --source_dataset_group beir \
    --target_dataset $train_dataset \
    --top_k 5 \
    --temperature 0.1
done