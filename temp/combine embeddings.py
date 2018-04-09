import numpy

file_path1 = "../comm/citeseer_test_again_word.embedding"
file_path2 = "../comm/citeseer_poisson_model1_iter_750.embedding"
output_file = "../comm/combined.embedding"

embedding1 = {}
embedding2 = {}

with open(file_path1, 'r') as f1, open(file_path2, 'r') as f2:
    count = 0
    for line1 in f1:
        if count == 0:
            firstline1 = line1.strip().split()
        else:
            tokens1 = line1.strip().split()
            embed1 = []
            embed1.extend(tokens1[1:])
            embedding1.update({tokens1[0]: embed1})

        count += 1

    count = 0
    for line2 in f2:
        if count == 0:
            firstline2 = line2.strip().split()
        else:
            tokens2 = line2.strip().split()
            embed2 = []
            embed2.extend(tokens2[1:])
            embedding2.update({tokens2[0]: embed2})

        count += 1


with open(output_file, 'w') as f:
    f.write("{} {}\n".format(firstline1[0], int(firstline1[1])+int(firstline2[1])))
    for val in embedding1:
        new_line = []
        new_line.extend(embedding1[val])
        new_line.extend(embedding2[val])
        line = "{} {}\n".format(val, " ".join(new_line))
        f.write(line)
