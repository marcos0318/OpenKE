import pickle

data_names = ["FB15k-237-betae",  "FB15k-237-q2b",  "FB15k-betae", "FB15k-q2b",    "NELL-betae",  "NELL-q2b"]


for data_name in data_names:
    path = "./data/" + data_name + "/"

    # read the id2rel.pkl and write to relation2id.txt
    fin = open(path + "id2rel.pkl", "rb")
    id2rel = pickle.load(fin)

    fout = open(path + "relation2id.txt", "w")

    num_rel = len(id2rel)
    fout.write(str(num_rel) + "\n\r")
    for i in range(num_rel):
        fout.write(id2rel[i] + "\t" + str(i) + "\n")

    # read the id2rel.pkl and write to relation2id.txt
    fin = open(path + "id2ent.pkl", "rb")
    id2ent = pickle.load(fin)

    fout = open(path + "entity2id.txt", "w")

    num_ent = len(id2ent)
    fout.write(str(num_ent) + "\n\r")
    for i in range(num_ent):
        fout.write(id2ent[i] + "\t" + str(i) + "\n")


    # re-format the train.txt/valid.txt/test.txt edges to train2id.txt test2id.txt valid2id.txt

    fin = open(path + "train.txt", "r")
    fout = open(path + "train2id.txt", "w")

    all_data = fin.readlines()
    num_data = len(all_data)
    fout.write(str(num_data) + "\n\r")

    for line in all_data:
        line_tuple = line.strip().split("\t")

        out_line = "  ".join([line_tuple[0], line_tuple[2], line_tuple[1]]) + "\n\r"
        fout.write(out_line)


    fin = open(path + "valid.txt", "r")
    fout = open(path + "valid2id.txt", "w")

    all_data = fin.readlines()
    num_data = len(all_data)
    fout.write(str(num_data) + "\n\r")

    for line in all_data:
        line_tuple = line.strip().split("\t")

        out_line = "  ".join([line_tuple[0], line_tuple[2], line_tuple[1]]) + "\n\r"
        fout.write(out_line)

    fin = open(path + "test.txt", "r")
    fout = open(path + "test2id.txt", "w")

    all_data = fin.readlines()
    num_data = len(all_data)
    fout.write(str(num_data) + "\n\r")

    for line in all_data:
        line_tuple = line.strip().split("\t")

        out_line = "  ".join([line_tuple[0], line_tuple[2], line_tuple[1]]) + "\n\r"
        fout.write(out_line)






