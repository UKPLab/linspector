#!/usr/bin/env bash


WORDFOLDER=../words/extrinsic
EMBEDFOLDER=../saved_embeddings/extrinsic
slang="es"

# merge words and embeddings with space delimiter
paste -d ' ' $WORDFOLDER/$slang/$slang.txt $EMBEDFOLDER/$slang/bpe/embeds.vec > $EMBEDFOLDER/$slang/bpe/final_embeds.vec
paste -d ' ' $WORDFOLDER/$slang/$slang.txt $EMBEDFOLDER/$slang/elmo/embeds.vec > $EMBEDFOLDER/$slang/elmo/final_embeds.vec
paste -d ' ' $WORDFOLDER/$slang/$slang.txt $EMBEDFOLDER/$slang/fasttext/embeds.vec > $EMBEDFOLDER/$slang/fasttext/final_embeds.vec
paste -d ' ' $WORDFOLDER/$slang/$slang.txt $EMBEDFOLDER/$slang/muse_supervised/embeds.vec > $EMBEDFOLDER/$slang/muse_supervised/final_embeds.vec


slang="de"

# merge words and embeddings with space delimiter
paste -d ' ' $WORDFOLDER/$slang/$slang.txt $EMBEDFOLDER/$slang/bpe/embeds.vec > $EMBEDFOLDER/$slang/bpe/final_embeds.vec
paste -d ' ' $WORDFOLDER/$slang/$slang.txt $EMBEDFOLDER/$slang/elmo/embeds.vec > $EMBEDFOLDER/$slang/elmo/final_embeds.vec
paste -d ' ' $WORDFOLDER/$slang/$slang.txt $EMBEDFOLDER/$slang/fasttext/embeds.vec > $EMBEDFOLDER/$slang/fasttext/final_embeds.vec
paste -d ' ' $WORDFOLDER/$slang/$slang.txt $EMBEDFOLDER/$slang/muse_supervised/embeds.vec > $EMBEDFOLDER/$slang/muse_supervised/final_embeds.vec


slang="tr"

# merge words and embeddings with space delimiter
paste -d ' ' $WORDFOLDER/$slang/$slang.txt $EMBEDFOLDER/$slang/bpe/embeds.vec > $EMBEDFOLDER/$slang/bpe/final_embeds.vec
paste -d ' ' $WORDFOLDER/$slang/$slang.txt $EMBEDFOLDER/$slang/elmo/embeds.vec > $EMBEDFOLDER/$slang/elmo/final_embeds.vec
paste -d ' ' $WORDFOLDER/$slang/$slang.txt $EMBEDFOLDER/$slang/fasttext/embeds.vec > $EMBEDFOLDER/$slang/fasttext/final_embeds.vec
paste -d ' ' $WORDFOLDER/$slang/$slang.txt $EMBEDFOLDER/$slang/muse_supervised/embeds.vec > $EMBEDFOLDER/$slang/muse_supervised/final_embeds.vec


slang="ru"

# merge words and embeddings with space delimiter
paste -d ' ' $WORDFOLDER/$slang/$slang.txt $EMBEDFOLDER/$slang/bpe/embeds.vec > $EMBEDFOLDER/$slang/bpe/final_embeds.vec
paste -d ' ' $WORDFOLDER/$slang/$slang.txt $EMBEDFOLDER/$slang/elmo/embeds.vec > $EMBEDFOLDER/$slang/elmo/final_embeds.vec
paste -d ' ' $WORDFOLDER/$slang/$slang.txt $EMBEDFOLDER/$slang/fasttext/embeds.vec > $EMBEDFOLDER/$slang/fasttext/final_embeds.vec
paste -d ' ' $WORDFOLDER/$slang/$slang.txt $EMBEDFOLDER/$slang/muse_supervised/embeds.vec > $EMBEDFOLDER/$slang/muse_supervised/final_embeds.vec


slang="fi"

# merge words and embeddings with space delimiter
paste -d ' ' $WORDFOLDER/$slang/$slang.txt $EMBEDFOLDER/$slang/bpe/embeds.vec > $EMBEDFOLDER/$slang/bpe/final_embeds.vec
paste -d ' ' $WORDFOLDER/$slang/$slang.txt $EMBEDFOLDER/$slang/elmo/embeds.vec > $EMBEDFOLDER/$slang/elmo/final_embeds.vec
paste -d ' ' $WORDFOLDER/$slang/$slang.txt $EMBEDFOLDER/$slang/fasttext/embeds.vec > $EMBEDFOLDER/$slang/fasttext/final_embeds.vec
paste -d ' ' $WORDFOLDER/$slang/$slang.txt $EMBEDFOLDER/$slang/muse_supervised/embeds.vec > $EMBEDFOLDER/$slang/muse_supervised/final_embeds.vec
