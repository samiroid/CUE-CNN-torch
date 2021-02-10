BASE_PATH="/Users/samir/Dev/projects/CUE-CNN-torch/"
WORD_EMBEDDINGS=$BASE_PATH"/raw_data/word_embeddings.txt"
CORPUS=$BASE_PATH"/DATA/small_user_tweets.txt"
OUTPUT_PATH=$BASE_PATH"/DATA/u2v/"

python -m user2vec.u2v -input $CORPUS -emb $WORD_EMBEDDINGS -output $OUTPUT_PATH \
                        -lr 0.001 \
                        -epochs 20 \
                        -neg_samples 2 \
                        -margin 5 
