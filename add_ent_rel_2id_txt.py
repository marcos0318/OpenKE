import pickle

data_names = ["FB15k-237-betae",  "FB15k-237-q2b",  "FB15k-betae", "FB15k-q2b",    "NELL-betae",  "NELL-q2b"]


for data_name in data_names:
    path = "./data/" + data_name + "/"

    # read the id2rel.pkl and write to relation2id.txt
    fin = open(path + "id2rel.pkl", "rb")
    id2rel = pickle.load(fin)

    fout = open(path + "relation2id.txt", "w")

    num_rel = len(id2rel)
    fout.write(str(num_rel) + "\n")
    for i in range(num_rel):
        fout.write(id2rel[i] + "\t" + str(i) + "\n")

    # read the id2rel.pkl and write to relation2id.txt
    fin = open(path + "id2ent.pkl", "rb")
    id2ent = pickle.load(fin)

    fout = open(path + "entity2id.txt", "w")

    num_ent = len(id2ent)
    fout.write(str(num_ent) + "\n")
    for i in range(num_ent):
        fout.write(id2ent[i] + "\t" + str(i) + "\n")




