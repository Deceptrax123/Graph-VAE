from torch_geometric.loader import DataLoader
from torch_geometric.graphgym import init_weights
from torch_geometric.nn import VGAE
from dataset import MolecularGraphDataset
import torch
from Model.encoder import DrugEncoder
from Model.spectral_encoder import SpectralDrugEncoder
from torch.utils.data import ConcatDataset
import torch.multiprocessing as tmp
from torch import nn
import os
import gc
import wandb
from dotenv import load_dotenv


def train_epoch():

    epoch_loss = 0

    for step, graphs in enumerate(train_loader):
        z = model.encode(graphs.x_s, edge_index=graphs.edge_index_s)

        model.zero_grad()
        loss = model.recon_loss(z, graphs.edge_index_s) + \
            (model.kl_loss()/graphs.x_s.size(0))

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        del graphs
        del z

    return epoch_loss/train_steps


def test_epoch():
    epoch_loss = 0

    for step, graphs in enumerate(test_loader):
        z = model.encode(graphs.x_s, edge_index=graphs.edge_index_s)

        loss = model.recon_loss(z, graphs.edge_index_s)

        epoch_loss += loss.item()

        del graphs
        del z

    return epoch_loss/test_steps


def training_loop():

    for epoch in range(EPOCHS):
        model.train(True)

        train_loss = train_epoch()
        model.eval()

        with torch.no_grad():
            test_loss = test_epoch()

            print("Epoch {epoch}: ".format(epoch=epoch+1))
            print("Train Loss: {loss}".format(loss=train_loss))
            print("Test Loss: {loss}".format(loss=test_loss))

            wandb.log({
                "Train Loss": train_loss,
                "Test Reconstruction Loss": test_loss,
                "Learning Rate": optimizer.param_groups[0]['lr']
            })

            if (epoch+1) % 10 == 0:
                path = "weights/reactant_1/model{epoch}.pth".format(
                    epoch=epoch+1)

                torch.save(model.encoder.state_dict(), path)

        # Update learning rate
        scheduler.step()


if __name__ == '__main__':
    tmp.set_sharing_strategy('file_system')
    load_dotenv(".env")

    # Set the training and testing folds
    train_folds = ['fold1', 'fold2', 'fold3',
                   'fold4', 'fold5', 'fold6']
    test_folds = ['fold7', 'fold8']

    params = {
        'batch_size': 32,
        'shuffle': True,
        'num_workers': 0
    }

    wandb.init(
        project="Drug VAE",
        config={
            "Architecture": "VAE",
            "Dataset": "TDC Dataset TWOSIDES",
        }
    )

    train_set1 = MolecularGraphDataset(
        fold_key=train_folds[0], root=os.getenv("graph_files")+"/fold1"+"/data/", start=0)
    train_set2 = MolecularGraphDataset(fold_key=train_folds[1], root=os.getenv("graph_files")+"/fold2/"
                                       + "/data/", start=7500)
    train_set3 = MolecularGraphDataset(fold_key=train_folds[2], root=os.getenv("graph_files")+"/fold3/"
                                       + "/data/", start=15000)
    train_set4 = MolecularGraphDataset(fold_key=train_folds[3], root=os.getenv("graph_files")+"/fold4/"
                                       + "/data/", start=22500)
    train_set5 = MolecularGraphDataset(fold_key=train_folds[4], root=os.getenv("graph_files")+"/fold5/"
                                       + "/data/", start=30000)
    train_set6 = MolecularGraphDataset(fold_key=train_folds[5], root=os.getenv("graph_files")+"/fold6/"
                                       + "/data/", start=37500)

    test_set1 = MolecularGraphDataset(fold_key=test_folds[0], root=os.getenv("graph_files")+"/fold7/"
                                      + "/data/", start=45000)
    test_set2 = MolecularGraphDataset(fold_key=test_folds[1], root=os.getenv(
        "graph_files")+"/fold8"+"/data/", start=52500)

    train_set = ConcatDataset(
        [train_set1, train_set2, train_set3, train_set4, train_set5, train_set6])

    test_set = ConcatDataset([test_set1, test_set2])

    train_loader = DataLoader(train_set, **params, follow_batch=['x_s', 'x_t'])
    test_loader = DataLoader(test_set, **params, follow_batch=['x_s', 'x_t'])

    # Import Models
    encoder = SpectralDrugEncoder(in_features=train_set[0].x_s.size(1))
    for m in encoder.modules():
        init_weights(m)

    # Set up VGAE
    model = VGAE(encoder=encoder)

    # Hyperparameters
    EPOCHS = 10000
    LR = 0.005
    BETAS = (0.9, 0.999)

    optimizer = torch.optim.Adam(params=model.parameters(), lr=LR, betas=BETAS)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, verbose=True)

    train_steps = (len(train_set)+params['batch_size']-1)//params['batch_size']
    test_steps = (len(test_set)+params['batch_size']-1)//params['batch_size']

    training_loop()
