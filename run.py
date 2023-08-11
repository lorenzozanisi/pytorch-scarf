from dataset import ExampleDataset, MyDataset
from pipeline.data import get_data
import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import pickle
import matplotlib.pylab as plt
from scarf.loss import NTXent
from scarf.model import SCARF, Classifier, FineTuneClassifier
import yaml
import numpy as np
import copy
import coloredlogs
import verboselogs
import pipeline.pipeline_tools as pt
import pandas as pd
from pipeline.utils import output_dict
from pipeline.data import DataConverter
import logging
from pipeline.utils import train_keys_store
from typing import Union
from example.utils import train_epoch, dataset_embeddings, valid_epoch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
import torch.multiprocessing as mp


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def train_classifier_fn(model, train_loader, valid_loader, lr=None, epochs=None, patience=None,**cfg):
    # instantiate optimiser
    opt = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=1.0e-4)
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    missed_loss = []
    missed_acc = []

    count = 0
    best_loss = np.inf
    for epoch in range(epochs):

        logging.debug(f"Train Step: {epoch}")                
        loss, acc = model.train_step(
            train_loader, opt, epoch=epoch
        )
        train_loss.append(loss.item())
        train_acc.append(acc)

        validation_loss, validation_accuracy = model.validation_step(
            valid_loader)
        val_loss.append(validation_loss)
        val_acc.append(validation_accuracy)
        logging.debug(
            f"Classifier training. Epoch {epoch}, train loss {loss}, val_loss {val_loss} "
            )

        if epoch==0:
            best_model = model
        if val_loss[-1]<best_loss:
            best_model = copy.deepcopy(model)
        else:
            count+=1
            if count>patience:
                break

    return best_model  # [train_loss, val_loss], [train_acc, val_acc]

def run(cfg, use_pretrained_embeddings=False):

    cfg['use_classifier'] = True
    cfg['dropout'] = 0 
    encoder_depth = 4
    head_depth = 2    
    cfg['model_size'] = encoder_depth+head_depth
    cfg['flux'] = 'efiitg_gb'
    cfg['classifier_type'] = 'SingleClassifier'
    print(cfg)

    print('Loading data')
    (
        train_sample,
        train_classifier,
        valid_sample,
        valid_classifier,
        unlabelled_pool,
        holdout_set,
        holdout_classifier,
        scaler,
    ) = get_data(cfg) 

    logging.debug(f"{train_classifier.data['stable_label']}")
    finetune_fracs = np.arange(0.1,1,0.2)
    finetune_names = [f'finetune_{frac}' for frac in finetune_fracs]
    batch_size = 128
    results = {name: copy.deepcopy(output_dict) for name in ['full dataset']+finetune_names}
    # --- train CLS on full dataset

    devices = dict(
        zip(cfg['fluxes'], ['cuda' for i in range(len(cfg['fluxes']))])
    )    

    converter = DataConverter(cfg['gkmodel'])
    train_loader = converter.pandas_to_numpy_data(train_classifier)
    valid_loader = converter.pandas_to_numpy_data(valid_classifier)
    test_loader = converter.pandas_to_numpy_data(holdout_classifier)

    fulldataclassifier = pt.get_classifier_model(**cfg)
    fulldataclassifier = train_classifier_fn(model = fulldataclassifier, train_loader=train_loader, valid_loader=valid_loader,**cfg)
    _, fulldatalosses = fulldataclassifier.predict(test_loader)

    results['full dataset']["accuracy"].append(fulldatalosses[1])
    results['full dataset']["precision"].append(fulldatalosses[2])
    results['full dataset']["recall"].append(fulldatalosses[3])
    results['full dataset']["f1"].append(fulldatalosses[4])
    results['full dataset']["auc"].append(fulldatalosses[5])
    logging.debug(f"full data results {results['full dataset']}")
    
    # --- now pretrain only on input space using SCARF
    train_keys = train_keys_store(cfg['gkmodel'])       
    train_classifier_ssl_inputs = train_classifier.data[train_keys].values
    train_classifier_ssl_targets = train_classifier.data['stable_label'].values # -- seems unnecessary for ssl?
    valid_classifier_ssl_inputs = valid_classifier.data[train_keys].values
    valid_classifier_ssl_targets = valid_classifier.data['stable_label'].values # -- seems unnecessary for ssl?
    test_classifier_ssl_inputs = holdout_classifier.data[train_keys].values
    test_classifier_ssl_targets = holdout_classifier.data['stable_label'].values # -- seems unnecessary for ssl?    

    train_ds = ExampleDataset(
                            train_classifier_ssl_inputs,
                            )
    valid_ds = ExampleDataset(
                            valid_classifier_ssl_inputs,
                            )
    test_ds = ExampleDataset(
                            test_classifier_ssl_inputs,
                            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    emb_dim = 512

    model = SCARF(
        input_dim=train_ds.shape[1],
        emb_dim=emb_dim,
        encoder_depth=encoder_depth, 
        head_depth=2,        
        corruption_rate=0.2,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=0.0001)
    ntxent_loss = NTXent(device=device)
    
    epochs = 1000
    loss_history = []
    valid_loss_history = []
    for epoch in range(1, epochs + 1):
        epoch_loss = train_epoch(model, ntxent_loss, train_loader, optimizer, device, epoch)
        loss_history.append(epoch_loss)
        # valid_loss = valid_epoch(model, ntxent_loss, valid_loader, optimizer, device, epoch)
        # valid_loss_history.append(epoch_loss)

    plt.plot(np.arange(epochs), loss_history)
    plt.savefig('/home/ir-zani1/rds/rds-ukaea-ap001/ir-zani1/qualikiz/pytorch-scarf/plots/pretrain_loss.png')
    plt.close()

    
    # --- finetune using progressively higher fractions of dataset

    if use_pretrained_embeddings:
        train_embeddings = dataset_embeddings(model, train_loader, device)
        valid_embeddings = dataset_embeddings(model, valid_loader, device)
        test_embeddings = dataset_embeddings(model, test_loader, device)

        train_embeddings = pd.DataFrame(train_embeddings)
        valid_embeddings = pd.DataFrame(valid_embeddings)
        test_embeddings = pd.DataFrame(test_embeddings)

        train_data_finetune = MyDataset(train_embeddings, pd.DataFrame(train_classifier_ssl_targets))
        valid_data_finetune = MyDataset(valid_embeddings,pd.DataFrame(valid_classifier_ssl_targets))
        test_data_finetune = MyDataset(test_embeddings, pd.DataFrame(test_classifier_ssl_targets))

    else:
        train_data_finetune = MyDataset(pd.DataFrame(train_classifier_ssl_inputs), pd.DataFrame(train_classifier_ssl_targets))
        valid_data_finetune = MyDataset(pd.DataFrame(valid_classifier_ssl_inputs),pd.DataFrame(valid_classifier_ssl_targets))
        test_data_finetune = MyDataset(pd.DataFrame(test_classifier_ssl_inputs), pd.DataFrame(test_classifier_ssl_targets))
        emb_dim = len(train_keys)

    valid_loader = DataLoader(valid_data_finetune, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data_finetune, batch_size=batch_size, shuffle=False)

    # --- temporary!
    # train_data_finetune = MyDataset(pd.DataFrame(train_classifier_ssl_inputs), pd.DataFrame(train_classifier_ssl_targets))
    # valid_data_finetune = MyDataset(pd.DataFrame(valid_classifier_ssl_inputs), pd.DataFrame(valid_classifier_ssl_targets))
    # test_data_finetune = MyDataset(pd.DataFrame(test_classifier_ssl_inputs), pd.DataFrame(test_classifier_ssl_targets))
    # valid_loader = DataLoader(valid_data_finetune, batch_size=batch_size, shuffle=False)
    # test_loader = DataLoader(test_data_finetune, batch_size=batch_size, shuffle=False)
    for frac in finetune_fracs:
        # logging.debug(f'Finetune frac {frac}')
        if not use_pretrained_embeddings:
            finetuneclassifier = FineTuneClassifier(inputs=512,encoder=model.encoder, dropout=0, base_model_size=head_depth, device=device)
        else:
            finetuneclassifier = Classifier(inputs=emb_dim, dropout=0, model_size=head_depth, device=device) # --batch_size- make the overall model with the same num of params as the vanilla fulldata cls
        temp_train_classifier = copy.deepcopy(train_data_finetune)
        temp_train_classifier.sample(frac=frac) #--- sample is inplace

        #finetuneclassifier = Classifier(inputs=15, dropout=0, model_size=2, device=device)
        
        train_loader = DataLoader(temp_train_classifier, batch_size=batch_size, shuffle=True)

        finetuneclassifier = train_classifier_fn(model = finetuneclassifier, train_loader=train_loader, valid_loader=valid_loader,**cfg)
        _, fulldatalosses = finetuneclassifier.predict(test_loader)

        results[f'finetune_{frac}']["accuracy"].append(fulldatalosses[1])
        results[f'finetune_{frac}']["precision"].append(fulldatalosses[2])
        results[f'finetune_{frac}']["recall"].append(fulldatalosses[3])
        results[f'finetune_{frac}']["f1"].append(fulldatalosses[4])
        results[f'finetune_{frac}']["auc"].append(fulldatalosses[5]) 
        logging.debug(f"results for finetune frac of {frac}: {results[f'finetune_{frac}']}")
    with open('/home/ir-zani1/rds/rds-ukaea-ap001/ir-zani1/qualikiz/pytorch-scarf/results.pkl', 'wb') as f:
         pickle.dump(results,f)
    plt.scatter(finetune_fracs, [results[f'finetune_{frac}']["f1"] for frac in finetune_fracs], label='With pretraining') 
    plt.scatter([1], results['full dataset']['f1'], color='red', label='No pretraining')      
    plt.ylabel('F1')
    plt.xlabel('Fraction of dataset used')
    plt.legend()
    plt.savefig('/home/ir-zani1/rds/rds-ukaea-ap001/ir-zani1/qualikiz/pytorch-scarf/plots/f1s.png')
    plt.close()

    plt.scatter(finetune_fracs, [results[f'finetune_{frac}']["accuracy"] for frac in finetune_fracs], label='With pretraining') 
    plt.scatter([1], results['full dataset']['accuracy'], color='red', label='No pretraining')      
    plt.ylabel('Accuracy')
    plt.xlabel('Fraction of dataset used')
    plt.legend()
    plt.savefig('/home/ir-zani1/rds/rds-ukaea-ap001/ir-zani1/qualikiz/pytorch-scarf/plots/accuracies.png')
    plt.close()

    plt.scatter(finetune_fracs, [results[f'finetune_{frac}']["precision"] for frac in finetune_fracs], label='With pretraining') 
    plt.scatter([1], results['full dataset']['precision'], color='red', label='No pretraining')      
    plt.ylabel('Precision')
    plt.xlabel('Fraction of dataset used')
    plt.legend()
    plt.savefig('/home/ir-zani1/rds/rds-ukaea-ap001/ir-zani1/qualikiz/pytorch-scarf/plots/precisions.png')
    plt.close()

    plt.scatter(finetune_fracs, [results[f'finetune_{frac}']["recall"] for frac in finetune_fracs], label='With pretraining') 
    plt.scatter([1], results['full dataset']['recall'], color='red', label='No pretraining')      
    plt.ylabel('Recall')
    plt.xlabel('Fraction of dataset used')
    plt.legend()
    plt.savefig('/home/ir-zani1/rds/rds-ukaea-ap001/ir-zani1/qualikiz/pytorch-scarf/plots/recalls.png')
    plt.close()





if __name__=='__main__':
    logging.getLogger("PIL").setLevel(
        logging.WARNING)  
    verboselogs.install()
    logger = logging.getLogger(__name__)
    coloredlogs.install(level="DEBUG")
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--default_config",
                        help="config file", required=True)
    parser.add_argument(
        "--scaler_choice",
        help="Choice of scaler to use, StandardScaler or LogScaler. ",
        default="StandardScaler",
        required=False,
    )

    parser.add_argument(
        "--uncert_type",
        help="Choice of uncertainty to, 'epistemic' or 'epistemicaleatoric'. ",
        default="epistemicaleatoric",
        required=False,
    )
    parser.add_argument(
        "--resume_id",
        help="Wandb id of the run to resume",
        default=None,
        required=False,
    )    

    parser.add_argument(
        "--use_classifier",
        help="Whether to use the classifier. If False, unstable points only will be considered.",
        default=False,
        action="store_true",
        required=False,
        dest="use_classifier",
    )
    parser.add_argument(
        "--use_regressor",
        help="Whether to use the regressor. If False, unstable points only will be considered.",
        default=False,
        action="store_true",
        required=False,
        dest="use_regressor",
    )    
    parser.add_argument(
        "--robust",
        default=False,
        help="Use robust acquisition function",
        action="store_true",
        required=False,
        dest="robust",
    )

    parser.add_argument(
        "--with_LCB_rejection",
        help="Deprecated.",
        default=False,
        action="store_true",
        required=False,
        dest="with_LCB_rejection",
    )
    parser.add_argument(
        "--from_scratch",
        help="Whether to retrain models from scratch",
        default=False,
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--use_all_outputs",
        help="Makes sense only if training on the leading flux only. Whether to also use zeros for the outputs.",
        default=False,
        action="store_true",
        required=False,
    )

    parser.add_argument(
        "--labels_unavailable",
        help="If True, the pool is assumed to not have labels",
        default=False,
        action="store_true",
        required=False,
    )
    parser.add_argument(
        "--train_size",
        help="The training set size",
        default=100000,
        required=False,
        type=int,
    )
    parser.add_argument(
        "--valid_size",
        help="The validation set size",
        default=20000,
        required=False,
        type=int,
    )
    parser.add_argument(
        "--test_size",
        help="The test set size",
        default=50000,
        required=False,
        type=int,
    )
    parser.add_argument(
        "--patience", help="The patience value", default=30, required=False, type=int
    )
    parser.add_argument(
        "--candidate_size",
        help="The subsample of candidates sampled from the pool to speed up the acquisition",
        default=10_000,
        required=False,
        type=int,
    )
    parser.add_argument(
        "--batch_size", help="The batch size", default=256, required=False, type=int
    )

    parser.add_argument(
        "--gkmodel",
        help="The gyrokinetic model. 'TGLF' and 'QLK' currently supported",
        default="QLK15D",
        required=False,
    )  # 'TGLF','QLK15D', 'QLK12D'

    parser.add_argument(
        "--gkmodel_output_dir",
        help="Where the gyrokinetic model outputs will be saved",
        default="/home/ir-zani1/rds/rds-ukaea-ap001/ir-zani1/qualikiz/Active_Continual_Learning/data/TGLF/tglf_runs/",
        required=False,
    ) 

    parser.add_argument(
        "--regressor_type", default="ParallelDeepEnsemble", required=False
    )  # 'SingleRegressor','EnsembleRegressor', 'DeepEnsemble','ParallelDeepEnsemble'
    parser.add_argument(
        "--classifier_type", default="SingleClassifier", required=False
    )  # 'SingleClassifier','ParallelEnsembleClassifier'
    parser.add_argument(
        "--uncertainty_estimate",
        help="The uncertainty estimate. This is fixed to be 'MCDropout' as this keyword is called only if regressor_type=='SingleRegressor'",
        default="MCDropout",
        required=False,
    )  # DEPRECATED 'MCDropout','Laplace','
    parser.add_argument(
        "--model_size",
        help="Size of the MLP(s). 4x512 for 'shallow_wide' and 8x512 for 'deep",
        default=8,
        required=False,
        type=int
    )  # 'shallow_wide','deep'
    parser.add_argument(
        "--loss_function",
        help="Loss function for the regressor. 'NLL' for GaussianNLL and 'MSE' for MSELoss",
        default="NLL",
        required=False,
    )  # 'shallow_wide','deep'
    parser.add_argument(
        "--num_estimators",
        help="Number of models in the ensemble",
        default=5,
        required=False,
        type=int,
    )
    parser.add_argument(
        "--Nbootstraps",
        help="Number of instances running in parallel. May be very slow for ensembles.",
        default=1,
        required=False,
        type=int,
    )  #
    parser.add_argument(
        "--top_keep",
        help="Acquisition batch size",
        default=256,
        required=False,
        type=int,
    )
    parser.add_argument(
        "--dropout", help="Dropout rate", default=0.1, required=False, type=float
    )
    parser.add_argument(
        "--lam", help="Deprecated", default=1, required=False, type=float
    )
    parser.add_argument(
        "--iterations",
        help="Number of Active Learning iterations",
        default=10,
        required=False,
        type=int,
    )
    parser.add_argument(
        "--epochs",
        help="Number of training epochs",
        default=100,
        required=False,
        type=int,
    )
    parser.add_argument(
        "--lr", help="Learning rate", default=0.0001, required=False, type=float
    )
    parser.add_argument(
        "--weight_decay",
        help="L2 regularization",
        default=1.0e-4,
        required=False,
        type=float,
    )
    parser.add_argument(
        "--beta", default=0.1, help="Deprecated", required=False, type=float
    )  # --- LCB hyperparameter to play with
    parser.add_argument(
        "--sample_size_debug",
        help="Smaller samples for debugging purposes, between 0 and 1",
        default=1,
        required=False,
        type=float,
    )
    parser.add_argument(
        "--scale",
        help="If true, data is rescaled at each acquisition according to the new distribution inclusive o f new data",
        default=False,
        action="store_true",
        required=False,
        dest="scale",
    )
    parser.add_argument(
        "--project_name",
        help="The wandb project name",
        default="active-learning-parallelensembles",
        required=False,
    )

    parser.add_argument(
        "--classifier_retrain_iterations",
        help="After how many iterations the classifier is retrainer",
        default=10,
        required=False,
        type=int,
    )

    parser.add_argument(
        "--acquisition_schedule",
        help="After how many iterations to double top_keep",
        default=None,
        required=False,
        type=int,
    )

    parser.add_argument(
        "--use_classifier_for_acquisition",
        help="If true, apply active learning for training the classifier",
        default=False,
        action="store_true",
        required=False,
        dest="use_classifier_for_acquisition",
    )

    args = parser.parse_args()
    with open(args.default_config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    input_cfg = vars(args)
    cfg.update(input_cfg)

    run(cfg=cfg, use_pretrained_embeddings=False)