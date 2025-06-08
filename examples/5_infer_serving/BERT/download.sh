#!/bin/bash

BERT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd -P)
mkdir -p $BERT_DIR/bert_large_tf1_ckpt_mode-qa_ds-squad11_insize-384_19.03.1

cd $BERT_DIR/bert_large_tf1_ckpt_mode-qa_ds-squad11_insize-384_19.03.1
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/dle/bert_large_tf1_ckpt_mode-qa_ds-squad11_insize-384/19.03.1_amp/files?redirect=true&path=bert_config.json' -O 'bert_config.json'
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/dle/bert_large_tf1_ckpt_mode-qa_ds-squad11_insize-384/19.03.1_amp/files?redirect=true&path=model.ckpt.index' -O 'model.ckpt.index'
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/dle/bert_large_tf1_ckpt_mode-qa_ds-squad11_insize-384/19.03.1_amp/files?redirect=true&path=model.ckpt.meta' -O 'model.ckpt.meta'
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/dle/bert_large_tf1_ckpt_mode-qa_ds-squad11_insize-384/19.03.1_amp/files?redirect=true&path=tf_bert_squad_1n_fp16_gbs32.190516222326.log' -O 'tf_bert_squad_1n_fp16_gbs32.190516222326.log'
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/dle/bert_large_tf1_ckpt_mode-qa_ds-squad11_insize-384/19.03.1_amp/files?redirect=true&path=vocab.txt' -O 'vocab.txt'
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/dle/bert_large_tf1_ckpt_mode-qa_ds-squad11_insize-384/19.03.1_amp/files?redirect=true&path=model.ckpt.data-00000-of-00001' -O 'model.ckpt.data-00000-of-00001'
