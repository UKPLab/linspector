#!/usr/bin/env bash
FOLDER=../embeddings
WORDFOLDER=../words/intrinsic
OUTFOLDER=../saved_embeddings/intrinsic
languages=("es" "de" "fi" "ru" "tr")
for id in {1..2}; do
  for langId in "${languages[@]}"; do
    echo "Extracting ELMO for $langId for part: $id"
    python3 extract_vectors.py \
    --w2vtype "elmo" \
    --lang $langId \
    --embedding "$FOLDER/elmo/$langId" \
    --part $id \
    --infile "$WORDFOLDER/$langId/splitted_$langId_$id.txt" \
    --savedir $OUTFOLDER
    echo "Extracting BPE for $langId for part: $id"
    python3 extract_vectors.py \
    --w2vtype "bpe" \
    --lang $langId \
    --part $id \
    --infile "$WORDFOLDER/$langId/splitted_$langId_$id.txt" \
    --savedir $OUTFOLDER
    echo "Extracting original fasttext for $langId for part: $id"
    python3 extract_vectors.py \
   	--w2vtype "fasttext" \
   	--lang $langId \
   	--embedding "$FOLDER/fasttext/wiki.$langId/wiki.$langId" \
   	--part $id \
   	--infile "$WORDFOLDER/$langId/splitted_$langId_$id.txt" \
   	--savedir $OUTFOLDER
    echo "Extracting muse supervised for $langId for part: $id"
  	python3 extract_vectors.py \
  	--w2vtype "muse_supervised" \
  	--lang $langId \
  	--embedding "$FOLDER/muse_supervised/wiki.multi.$langId/wiki.multi.$langId" \
  	--part $id \
  	--infile "$WORDFOLDER/$langId/splitted_$langId_$id.txt" \
  	--savedir $OUTFOLDER
  done
done
