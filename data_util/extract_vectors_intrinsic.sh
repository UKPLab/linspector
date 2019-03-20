#!/usr/bin/env bash

FOLDER=../embeddings
WORDFOLDER=../words/intrinsic
OUTFOLDER=../saved_embeddings/intrinsic
slang="es"


for id in {1..2}
do
   echo "Extracting ELMO for "$slang" for part: "$id
   python3 extract_vectors.py \
    --w2vtype "elmo" \
    --lang $slang \
    --embedding "$FOLDER/elmo/$slang" \
    --part $id \
    --infile "$WORDFOLDER/$slang/splitted_es_$id.txt" \
    --savedir $OUTFOLDER
done

for id in {1..2}
do
	echo "Extracting original fasttext for "$slang
	python3 extract_vectors.py \
	--w2vtype "fasttext" \
	--lang $slang \
	--embedding "$FOLDER/fasttext/wiki.$slang/wiki.$slang" \
	--part $id \
	--infile "$WORDFOLDER/$slang/splitted_es_$id.txt" \
	--savedir $OUTFOLDER
done

for id in {1..2}
do
	echo "Extracting BPE for "$slang
	python3 extract_vectors.py \
	--w2vtype "bpe" \
	--lang $slang \
	--part $id \
	--infile "$WORDFOLDER/$slang/splitted_es_$id.txt" \
	--savedir $OUTFOLDER
done

for id in {1..2}
do
	echo "Extracting muse supervised for "$slang
	python3 extract_vectors.py \
	--w2vtype "muse_supervised" \
	--lang $slang \
	--embedding "$FOLDER/muse_supervised/wiki.multi.$slang/wiki.multi.$slang" \
	--part $id \
	--infile "$WORDFOLDER/$slang/splitted_es_$id.txt" \
	--savedir $OUTFOLDER
done

slang="fi"
echo "Finnish"

for id in {1..2}
do
   echo "Extracting ELMO for "$slang" for part: "$id
   python3 extract_vectors.py \
    --w2vtype "elmo" \
    --lang $slang \
    --embedding "$FOLDER/elmo/$slang" \
    --part $id \
    --infile "$WORDFOLDER/$slang/splitted_fi_$id.txt" \
    --savedir $OUTFOLDER
done

for id in {1..2}
do
	echo "Extracting original fasttext for "$slang
	python3 extract_vectors.py \
	--w2vtype "fasttext" \
	--lang $slang \
	--embedding "$FOLDER/fasttext/wiki.$slang/wiki.$slang" \
	--part $id \
	--infile "$WORDFOLDER/$slang/splitted_fi_$id.txt" \
	--savedir $OUTFOLDER
done

for id in {1..2}
do
	echo "Extracting BPE for "$slang
	python3 extract_vectors.py \
	--w2vtype "bpe" \
	--lang $slang \
	--part $id \
	--infile "$WORDFOLDER/$slang/splitted_fi_$id.txt" \
	--savedir $OUTFOLDER
done

for id in {1..2}
do
	echo "Extracting muse supervised for "$slang
	python3 extract_vectors.py \
	--w2vtype "muse_supervised" \
	--lang $slang \
	--embedding "$FOLDER/muse_supervised/wiki.multi.$slang/wiki.multi.$slang" \
	--part $id \
	--infile "$WORDFOLDER/$slang/splitted_fi_$id.txt" \
	--savedir $OUTFOLDER
done

echo "Russian"
slang="ru"


for id in {1..2}
do
   echo "Extracting ELMO for "$slang" for part: "$id
   python3 extract_vectors.py \
    --w2vtype "elmo" \
    --lang $slang \
    --embedding "$FOLDER/elmo/$slang" \
    --part $id \
    --infile "$WORDFOLDER/$slang/splitted_ru_$id.txt" \
    --savedir $OUTFOLDER
done

for id in {1..2}
do
	echo "Extracting original fasttext for "$slang
	python3 extract_vectors.py \
	--w2vtype "fasttext" \
	--lang $slang \
	--embedding "$FOLDER/fasttext/wiki.$slang/wiki.$slang" \
	--part $id \
	--infile "$WORDFOLDER/$slang/splitted_ru_$id.txt" \
	--savedir $OUTFOLDER
done

for id in {1..2}
do
	echo "Extracting BPE for "$slang
	python3 extract_vectors.py \
	--w2vtype "bpe" \
	--lang $slang \
	--part $id \
	--infile "$WORDFOLDER/$slang/splitted_ru_$id.txt" \
	--savedir $OUTFOLDER
done

for id in {1..2}
do
	echo "Extracting muse supervised for "$slang
	python3 extract_vectors.py \
	--w2vtype "muse_supervised" \
	--lang $slang \
	--embedding "$FOLDER/muse_supervised/wiki.multi.$slang/wiki.multi.$slang" \
	--part $id \
	--infile "$WORDFOLDER/$slang/splitted_ru_$id.txt" \
	--savedir $OUTFOLDER
done


echo "Turkish"
slang="tr"


for id in {1..2}
do
   echo "Extracting ELMO for "$slang" for part: "$id
   python3 extract_vectors.py \
    --w2vtype "elmo" \
    --lang $slang \
    --embedding "$FOLDER/elmo/$slang" \
    --part $id \
    --infile "$WORDFOLDER/$slang/splitted_tr_$id.txt" \
    --savedir $OUTFOLDER
done

for id in {1..2}
do
	echo "Extracting original fasttext for "$slang
	python3 extract_vectors.py \
	--w2vtype "fasttext" \
	--lang $slang \
	--embedding "$FOLDER/fasttext/wiki.$slang/wiki.$slang" \
	--part $id \
	--infile "$WORDFOLDER/$slang/splitted_tr_$id.txt" \
	--savedir $OUTFOLDER
done

for id in {1..2}
do
	echo "Extracting BPE for "$slang
	python3 extract_vectors.py \
	--w2vtype "bpe" \
	--lang $slang \
	--part $id \
	--infile "$WORDFOLDER/$slang/splitted_tr_$id.txt" \
	--savedir $OUTFOLDER
done

for id in {1..2}
do
	echo "Extracting muse supervised for "$slang
	python3 extract_vectors.py \
	--w2vtype "muse_supervised" \
	--lang $slang \
	--embedding "$FOLDER/muse_supervised/wiki.multi.$slang/wiki.multi.$slang" \
	--part $id \
	--infile "$WORDFOLDER/$slang/splitted_tr_$id.txt" \
	--savedir $OUTFOLDER
done
