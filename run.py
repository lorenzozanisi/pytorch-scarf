from dataset import ExampleDataset
from pipeline.data import get_data
import argparse
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from scarf.loss import NTXent
from scarf.model import SCARF
import yaml
from pipeline.Models import Classifier
from pipeline_tools import train_classifier_fn
import numpy as np
import copy
from pipeline.utiles import results
from pipeline.data import DataConverter


class FineTuneClassifier(Classifier):
    def __init__(self, encoder, inputs=15, outputs=1, device=None,model_size: int = 8, dropout: float = 0.1):
        super.__init__(inputs=15, outputs=1, device=None,model_size = 8, dropout  = 0.1)

        self.model = torch.nn.Sequential(encoder, self.model)
    


def run(cfg):
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

    finetune_fracs = np.arange(0.1,1,0.1)
    finetune_names = [f'finetune_{frac}' for frac in finetune_fracs]

    results = {name: results for name in ['full dataset']+finetune_names}
    # --- train CLS on full dataset
    fulldataclassifier = Classifier(dropout=0, model_size=6)
    converter = DataConverter(gkmodel=cfg['gkmodel'])
    fulldata_train_loader = converter.pandas_to_numpy_data(
                train_classifier, batch_size=batch_size, shuffle=True
            )
    valid_loader = converter.pandas_to_numpy_data(
                valid_classifier, batch_size=batch_size, shuffle=True
            )        
    test_loader = converter.pandas_to_numpy_data(
                holdout_classifier, batch_size=batch_size, shuffle=True
            )            
    fulldataclassifier, _ = train_classifier_fn(fulldata_train_loader,valid_loader,model=fulldataclassifier,**cfg)    
    _, fulldatalosses = fulldataclassifier.predict(test_loader)

    results['full dataset']["accuracy"].append(fulldatalosses[1])
    results['full dataset']["precision"].append(fulldatalosses[2])
    results['full dataset']["recall"].append(fulldatalosses[3])
    results['full dataset']["f1"].append(fulldatalosses[4])
    results['full dataset']["auc"].append(fulldatalosses[5])
    
    
    # --- now pretrain only on input space using SCARF
    train_classifier_ssl = copy.deepcopy(train_classifier)
    train_classifier_ssl.data = train_classifier_ssl.data.drop(columns='efiitg_gb')
    train_ds = ExampleDataset(train_classifier_ssl.data, target="stable_label")

    batch_size = 128
    epochs = 5000
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = SCARF(
        input_dim=train_ds.shape[1],
        emb_dim=512,
        encoder_depth=4, 
        head_depth=2,        
        corruption_rate=0.5,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=0.001)
    ntxent_loss = NTXent()

    for epoch in range(1, epochs + 1):
        for anchor, positive in train_loader:
                anchor, positive = anchor.to(device), positive.to(device)

                # reset gradients
                optimizer.zero_grad()

                # get embeddings
                emb, emb_corrupted = model(anchor, positive)

                # compute loss
                loss = ntxent_loss(emb, emb_corrupted)
                loss.backward()

                # update model weights
                optimizer.step()

    
    # --- finetune using progressively higher fractions of dataset

    for frac in finetune_fracs:
        this_model = copy.deepcopy(model)
        finetuneclassifier = FineTuneClassifier(this_model.encoder,dropout=0, model_size=2) # --- make the overall model with the same num of params as the vanilla fulldata cls
        finetune_train_classifier = copy.deepcopy(train_classifier)
        finetune_train_classifier.data = finetune_train_classifier.data.sample(frac=frac)
        train_loader = converter.pandas_to_numpy_data(
                    finetune_train_classifier, batch_size=batch_size, shuffle=True
                )
        finetuneclassifier, _ = train_classifier_fn(train_loader,valid_loader,model=finetuneclassifier,**cfg)

        results[f'finetune_{frac}']["accuracy"].append(fulldatalosses[1])
        results[f'finetune_{frac}']["precision"].append(fulldatalosses[2])
        results[f'finetune_{frac}']["recall"].append(fulldatalosses[3])
        results[f'finetune_{frac}']["f1"].append(fulldatalosses[4])
        results[f'finetune_{frac}']["auc"].append(fulldatalosses[5])        
    
if __name__=='__main__':

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
        default=5000,
        required=False,
        type=int,
    )
    parser.add_argument(
        "--valid_size",
        help="The validation set size",
        default=20_000,
        required=False,
        type=int,
    )
    parser.add_argument(
        "--test_size",
        help="The test set size",
        default=50_000,
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
        "--acquisition",
        help="The acquisition function, choose between 'uncertainty' and 'random'",
        default="uncertainty",
        required=True,
    )  # 'uncertainty','random'

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
        "--classifier_type", default="ParallelEnsembleClassifier", required=False
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

    run(cfg=cfg)