from tqdm import tqdm
import nltk

in_path = "../data/processed/all.txt"
out_path = "../data/processed/segmented.txt"

with open(in_path) as file:
    total_lines = sum(1 for line in file)

with open(out_path, "w") as out_file, open(in_path) as in_file:
    for line in tqdm(in_file, total=total_lines):
        line = line.strip()

        if len(line) == 0:
            out_file.write("\n")
            continue

        sentences = nltk.sent_tokenize(line)
        sentences = "\n".join(sentences)
        out_file.write(f"{sentences}[PAR]\n")
