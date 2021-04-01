import openke
from openke.config import Trainer, Tester
from openke.module.model import TransE
from openke.module.loss import MarginLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader, TestDataLoader


check_points = {
    "./data/FB15k-betae/":  "./checkpoint/transe_batae_fb15k.ckpt",
    "./data/FB15k-237-betae/":  './checkpoint/transe_batae_fb237.ckpt',
    "./data/NELL-betae/":  './checkpoint/transe_batae_nell.ckpt',
}



for data_path, check_point_path in check_points.items():
    # dataloader for training
    train_dataloader = TrainDataLoader(
        in_path=data_path,
        nbatches=25,
        threads=8,
        sampling_mode="normal",
        bern_flag=1,
        filter_flag=1,
        neg_ent=25,
        neg_rel=0)

    # define the model
    transe = TransE(
        ent_tot=train_dataloader.get_ent_tot(),
        rel_tot=train_dataloader.get_rel_tot(),
        dim=400,
        p_norm=1,
        norm_flag=True)

    # test the model
    transe.load_checkpoint('./checkpoint/transe_batae_fb15k.ckpt')

    print(data_path, "ent")
    print(transe.ent_embeddings)
    print(transe.ent_embeddings.weight.shape)

    print(data_path, "rel")
    print(transe.rel_embeddings)
    print(transe.rel_embeddings.weight.shape)



