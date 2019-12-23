#!/usr/bin/env bash
WORDFOLDER=../words/extrinsic
EMBEDFOLDER=../saved_embeddings/extrinsic
languages=("es" "de" "fi" "ru" "tr")
for langId in "${languages[@]}"; do
  # merge words and embeddings with space delimiter
  paste -d ' ' $WORDFOLDER/$langId/$langId.txt $EMBEDFOLDER/$langId/bpe/embeds.vec > $EMBEDFOLDER/$langId/bpe/final_embeds.vec
  paste -d ' ' $WORDFOLDER/$langId/$langId.txt $EMBEDFOLDER/$langId/elmo/embeds.vec > $EMBEDFOLDER/$langId/elmo/final_embeds.vec
  paste -d ' ' $WORDFOLDER/$langId/$langId.txt $EMBEDFOLDER/$langId/fasttext/embeds.vec > $EMBEDFOLDER/$langId/fasttext/final_embeds.vec
  paste -d ' ' $WORDFOLDER/$langId/$langId.txt $EMBEDFOLDER/$langId/muse_supervised/embeds.vec > $EMBEDFOLDER/$langId/muse_supervised/final_embeds.vec
done