import argparse
import numpy as np
import transformers

def decode(in_file: str, out_file: str, tokenizer: transformers.AutoTokenizer) -> int:
    mem = np.memmap(in_file, mode="r", dtype="uint16")
    tokens = len(mem)
    with open(out_file, "a") as f:
        for token in mem:
            f.write(tokenizer.decode([token]))
    return tokens

def encode(in_file: str, out_file: str, tokenizer: transformers.AutoTokenizer) -> int:
    with open(in_file, "r", encoding="utf-8") as f:
        text = f.read()
    tokens = tokenizer.encode(text)
    with open(out_file, "wb") as f:
        for token in tokens:
            f.write(np.uint16(token))
    return len(tokens)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset Creator')
    parser.add_argument('--in_file', type=str, help='input file to use', required=True)
    parser.add_argument('--out_file', type=str, help='output file to use', required=True)
    parser.add_argument('--model', type=str, help='model tokenizer to use', required=True)
    args = parser.parse_args()

    encode(args.in_file, args.out_file, transformers.AutoTokenizer.from_pretrained(args.model))