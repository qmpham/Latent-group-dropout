import sentencepiece as spm
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str)
parser.add_argument("--file", type=str, nargs="+")
args = parser.parse_args()

spsrc = spm.SentencePieceProcessor()
spsrc.Load(args.model)
for f in args.file:
  print("tokenizing %s"%f)
  with open(f,"r") as f_r:
    with open("%s.bpe"%f,"w") as f_w:
      for line in f_r.readlines():

        print(" ".join(spsrc.EncodeAsPieces(line.strip())),file=f_w)

