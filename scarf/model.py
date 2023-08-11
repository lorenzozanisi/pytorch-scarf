import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np


class MLP(torch.nn.Sequential):
    """Simple multi-layer perceptron with ReLu activation and optional dropout layer"""

    def __init__(self, input_dim, hidden_dim, n_layers, dropout=0.0):
        layers = []
        in_dim = input_dim
        for _ in range(n_layers - 1):
            layers.append(torch.nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(torch.nn.Dropout(dropout))
            in_dim = hidden_dim

        layers.append(torch.nn.Linear(in_dim, hidden_dim))

        super().__init__(*layers)


class SCARF(nn.Module):
    def __init__(
        self,
        input_dim,
        emb_dim,
        encoder_depth=4,
        head_depth=2,
        corruption_rate=0.6,
        encoder=None,
        pretraining_head=None,
    ):
        """Implementation of SCARF: Self-Supervised Contrastive Learning using Random Feature Corruption.
        It consists in an encoder that learns the embeddings.
        It is done by minimizing the contrastive loss of a sample and a corrupted view of it.
        The corrupted view is built by remplacing a random set of features by another sample randomly drawn independently.

            Args:
                input_dim (int): size of the inputs
                emb_dim (int): dimension of the embedding space
                encoder_depth (int, optional): number of layers of the encoder MLP. Defaults to 4.
                head_depth (int, optional): number of layers of the pretraining head. Defaults to 2.
                corruption_rate (float, optional): fraction of features to corrupt. Defaults to 0.6.
                encoder (nn.Module, optional): encoder network to build the embeddings. Defaults to None.
                pretraining_head (nn.Module, optional): pretraining head for the training procedure. Defaults to None.
        """
        super().__init__()

        if encoder:
            self.encoder = encoder
        else:
            self.encoder = MLP(input_dim, emb_dim, encoder_depth)

        if pretraining_head:
            self.pretraining_head = pretraining_head
        else:
            self.pretraining_head = MLP(emb_dim, emb_dim, head_depth)

        # initialize weights
        self.encoder.apply(self._init_weights)
        self.pretraining_head.apply(self._init_weights)
        self.corruption_len = int(corruption_rate * input_dim)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)

    def forward(self, anchor, random_sample):
        batch_size, m = anchor.size()

        # 1: create a mask of size (batch size, m) where for each sample we set the
        # jth column to True at random, such that corruption_len / m = corruption_rate
        # 3: replace x_1_ij by x_2_ij where mask_ij is true to build x_corrupted

        corruption_mask = torch.zeros_like(anchor, dtype=torch.bool)
        for i in range(batch_size):
            corruption_idx = torch.randperm(m)[: self.corruption_len]
            corruption_mask[i, corruption_idx] = True

        positive = torch.where(corruption_mask, random_sample, anchor)

        # compute embeddings
        emb_anchor = self.encoder(anchor)
        emb_anchor = self.pretraining_head(emb_anchor)

        emb_positive = self.encoder(positive)
        emb_positive = self.pretraining_head(emb_positive)

        return emb_anchor, emb_positive

    def get_embeddings(self, input):
        return self.encoder(input)


class Classifier(nn.Module):
    #ToDo Adapt to multiclass 
    def __init__(self, inputs=15, outputs=1, device=None,model_size: int = 8, dropout: float = 0.1):
        super().__init__()
        self.type = "classifier"
        self.dropout = dropout
        self.model_size = model_size
        self.inputs = inputs
        self.outputs = outputs 
        layers = [nn.Linear(self.inputs, 512), nn.Dropout(p=dropout), nn.ReLU()]
        for i in range(model_size-2):
            layers.append(nn.Linear(512,512))
            layers.append(nn.Dropout(p=dropout))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(512,1))
        layers.append(nn.Sigmoid())
        self.model = nn.Sequential(*layers)
        self.set_device(device)

    def set_device(self, device):
        self.device = device
        self.model.to(device)

    def shrink_perturb(self, lam, loc, scale):
        if lam != 1:
            with torch.no_grad():
                for param in self.model.parameters():
                    loc_tensor = loc * torch.ones_like(param)
                    scale_tensor = scale * torch.ones_like(param)
                    noise_dist = torch.distributions.Normal(loc_tensor, scale)
                    noise = noise_dist.sample()

                    param_update = (param * lam) + noise
                    param.copy_(param_update)

    def reset_weights(self):
        self.model.apply(weight_reset)

    def forward(self, x):
        y_hat = self.model(x)


        return y_hat

    def train_step(self, dataloader, optimizer, epoch=None, disable_tqdm=False):
        # Initalise loss function

        size = len(dataloader.dataset)
        num_batches = len(dataloader)

        losses = []
        correct = 0
        for batch, (X, y, idx) in enumerate(
            dataloader
        ):

            L_pos_class = len(y[y==1])
            L_neg_class = len(y[y==0])
            
            if L_pos_class>1 and L_neg_class>1:
                w_neg = (L_pos_class+L_neg_class)/(2*L_neg_class)
                w_pos = (L_pos_class+L_neg_class)/(2*L_pos_class)
                idx_pos = torch.where(y[y==1])[0]
                idx_neg = torch.where(y[y==0])[0]
                weights = torch.zeros(len(y))
                weights[idx_pos] = w_pos
                weights[idx_neg] = w_neg
                weights = torch.Tensor(weights).to(self.device)
                BCE = nn.BCELoss(weight=weights.unsqueeze(-1).float())
            else:
                BCE = nn.BCELoss()

            BCE = nn.BCELoss()
            X = X.to(self.device)
            y = y.to(self.device)
            y_hat = self.forward(X.float())
            # for some reason, no need to unsqueeze y anymore...
            loss = BCE(y_hat, y.float())

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            

            # calculate train accuracy
            pred_class = torch.round(y_hat.squeeze())  # torch.round(y_hat.squeeze())
            correct += torch.sum(
                pred_class == y.float()
            ).item()  # torch.sum(pred_class == y.float()).item()


        correct /= size
        average_loss = np.mean(losses)
        #logging.debug(f"TRAIN accuracy: {correct:>7f}, loss: {average_loss:>7f}")
        return average_loss, correct

    def validation_step(self, dataloader):
        size = len(dataloader.dataset)
        # Initalise loss function
        BCE = nn.BCELoss()

        test_loss = []
        correct = 0

        true_pos, true_neg, false_pos, false_neg = 0,0,0,0
        with torch.no_grad():
            for X, y, idx in dataloader:
                X = X.to(self.device)
                y = y.to(self.device)
                y_hat = self.forward(X.float())
                # for some reason, no need to unsqueeze y anymore...
                test_loss.append(BCE(y_hat, y.float()).item())

                # calculate test accuracy
                pred_class = torch.round(
                    y_hat.squeeze()
                )  # torch.round(y_hat.squeeze())
                correct += torch.sum(
                    pred_class == y.float()
                ).item()  # torch.sum(pred_class == y.float()).item()

                
                pred_true_idx = torch.where(pred_class == 1)[0]
                pred_false_idx = torch.where(pred_class == 0)[0]
              #  logging.debug(f'TRUE AND FALSE {pred_true_idx}, {pred_false_idx}')
                if len(pred_true_idx)>0 and len(pred_false_idx)>0:
                    true_pos += torch.sum(pred_class[pred_true_idx] == y[pred_true_idx].float()).item()
                    true_neg += torch.sum(pred_class[pred_false_idx] == y[pred_false_idx].float()).item()

                    false_pos += torch.sum(pred_class[pred_true_idx] != y[pred_true_idx].float()).item()
                    false_neg += torch.sum(pred_class[pred_false_idx] != y[pred_false_idx].float()).item()

        correct /= size
        average_loss = np.mean(test_loss)       
       #
       # logging.debug(f"Val accuracy: {correct:>7f}, loss: {average_loss:>7f}")
        try:    
            precision = true_pos / (true_pos + false_pos)
            recall = true_pos / (true_pos + false_neg)
            f1 = 2 * precision * recall / (precision + recall)  
        #    logging.debug(f"Val precision {precision:>7f}, recall: {recall:>7f}, F1: {f1:>7f}")
        except:
            pass
        #    logging.debug(f'Valid step failed with: {true_pos},{true_neg}, {false_neg}, {false_pos}')

        return average_loss, correct

    def predict(self, dataloader):

        if not isinstance(dataloader, DataLoader):
            dataloader = DataLoader(dataloader, batch_size=len(dataloader), shuffle=False)
        size = len(dataloader.dataset)
        pred = []
        y_true = []
        losses = []
        correct = 0
        true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0

        BCE = nn.BCELoss()

        for batch, (x, y, idx) in enumerate(dataloader):
            x = x.to(self.device)
            y = y.to(self.device)
            y_hat = self.forward(x.float())
            pred_class = torch.round(y_hat.squeeze())

            pred.append(pred_class.detach().cpu().numpy())
            y_true.append(y.detach().cpu().numpy())
            loss = BCE(y_hat, y.float()).item()
            losses.append(loss)

            # calculate test accuracy
            correct += torch.sum(
                pred_class == y.float()
            ).item()


            pred_true_idx = torch.where(pred_class == 1)[0]
            pred_false_idx = torch.where(pred_class == 0)[0]
            if len(pred_true_idx)>0 and len(pred_false_idx)>0:
                true_pos += torch.sum(pred_class[pred_true_idx] == y[pred_true_idx].float()).item()
                true_neg += torch.sum(pred_class[pred_false_idx] == y[pred_false_idx].float()).item()

                false_pos += torch.sum(pred_class[pred_true_idx] != y[pred_true_idx].float()).item()
                false_neg += torch.sum(pred_class[pred_false_idx] != y[pred_false_idx].float()).item()
        
        average_loss = np.sum(losses) / size

        correct /= size

#        logging.debug(f"TEST accuracy: {correct:>7f}, loss: {average_loss:>7f}")

        try:
            precision = true_pos / (true_pos + false_pos)
            recall = true_pos / (true_pos + false_neg)
#            logging.debug(f'Precision, recall, {precision}, {recall}')
            f1 = 2 * precision * recall / (precision + recall)  
#            logging.debug(f'F1 score {f1}')        
#            logging.debug(f"Test precision {precision:>7f}, recall: {recall:>7f}, F1: {f1:>7f}")
        except:
#            print('TEST: no good values this time round.')
            precision = np.nan
            recall = np.nan
            f1 = np.nan

        pred = np.hstack(np.asarray(pred, dtype=object).flatten())
        y_true = np.asarray(y_true, dtype=object).flatten()

        try:
            roc_auc = roc_auc_score(y_true.astype(int), pred)

        except:
            roc_auc = 0
#            logging.info("ROC AUC score not available need to fix")

        return pred, [average_loss, correct, precision, recall, f1, roc_auc]



class FineTuneClassifier(Classifier):
    def __init__(self, encoder, inputs=512, outputs=1, device=None,base_model_size: int = 8, dropout: float = 0.1):
        print('INPUTS ARE', inputs)
        super().__init__(inputs=inputs, outputs=outputs, device=device,model_size = base_model_size, dropout  = dropout)
        self.encoder = encoder
        self.model = nn.Sequential(self.encoder, self.model)
        print(self.model)
    def forward(self,x):
         yhat = self.model.forward(x)
         return yhat
