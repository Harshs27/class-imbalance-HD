"""
Functions for
1. Learning
2. Sampling 
"""

import copy
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn import preprocessing
import sys
import torch
import torch.nn as nn
import torchvision

# local imports
import ngm.utils.neural_view as neural_view
import ngm.utils.data_processing as dp

######################################################################
# Functions for NGM learning
######################################################################

def product_weights_MLP(model):
    """
    Reads the input model (MLP) and returns the normalized
    product of the neural network weight matrices. 
    """
    # global device
    for i, (n, p) in enumerate(model.MLP.named_parameters()):
        if i==0:
            if 'weight' in n:
                W = torch.abs(p).t()#.to(device) # DxH
                # Normalizing the weight using L2-norm
                W = torch.nn.functional.normalize(W)
        else: # i > 0
            if 'weight' in n:
                curr_W = torch.abs(p).t()
                # Normalizing the current weight using L2-norm
                curr_W = torch.nn.functional.normalize(curr_W)
                W = torch.matmul(W, curr_W)
                # Normalizing the running weight product using L2-norm
                W = torch.nn.functional.normalize(W)
    return W


def forward_NGM(X, model, S, structure_penalty='hadamard', lambd=0.1):
    """Pass the input X through the NGM model
    to obtain the X_pred. 

    LOSS = reg_loss + lambd * structure_loss

    The 'hadamard' ||prodW * Sc|| is more theoretically sound as it just 
    focuses on the terms needed to zero out and completely drop the 
    non-zero terms. 
    The 'diff' ||prodW-S|| also tries to make the non-zero terms go to 1.

    Args:
        X (torch.Tensor BxD): Input data
        model (torch.nn.object): The MLP model for NGM's `neural' view
        S (pd.DataFrame): Adjacency matrix from graph G
        structure_penalty (str): 'hadamard':||prodW * Sc||, 'diff':||prodW-S||
        lambd (float): reg_loss + lambd * structure_loss
            Recommended lambd=1 as the losses are scaled to the same range.
    
    Returns:
        (list): [
            Xp (torch.Tensor BxD): The predicted X
            loss (torch.scalar): The NGM loss 
            reg_loss (torch.scalar): The regression term loss
            structure_loss (torch.scalar): The structure penalty loss
        ]
    """
    # 1. Running the NGM model 
    Xp = model.MLP(X)
    # 2. Calculate the regression loss
    mse = nn.MSELoss() 
    reg_loss = mse(Xp, X)
    # 3. Initialize the structure loss
    structure_loss = torch.zeros(1)[0]
    if lambd > 0:
        # 3.2 Get the product of weights (L2 normalized) of the MLP
        prod_W = product_weights_MLP(model)
        # print(f'check prod_w in cuda {prod_W, S}')
        D = prod_W.shape[-1]
        # 3.3 Calculate the penalty
        if structure_penalty=='hadamard':
            # Using the L2 norm for high structure penalty
            structure_loss = torch.linalg.norm(prod_W*S, ord=2)
        elif structure_penalty=='diff':
            struct_mse = nn.MSELoss() 
            structure_loss = struct_mse(prod_W, S)
        # 3.4 Scale the structure loss
        structure_loss = structure_loss/(D**2)
        # Adding the log scaling
        structure_loss = torch.log(structure_loss)
    # 4. Calculate the total loss = reg_loss + lambd * struct_loss
    loss = reg_loss + lambd * structure_loss
    return Xp, loss, reg_loss, structure_loss


def learning(
    G, 
    X,
    lambd=1.0,
    hidden_dim=20,
    epochs=1200, 
    lr=0.001,
    norm_type='min_max',
    k_fold=1,
    structure_penalty='hadamard',
    VERBOSE=True, 
    BATCH_SIZE=None,
    USE_CUDA=True
    ):
    """Learn the distribution over a conditional independence graph. 
    1. Fit a MLP (autoencoder) to learn the data representation from X->X. 
    2. The input-output path of dependence structure of the MLP 
       should match the conditional independence structure of the
       input graph. This is achieved using a regularization term.
    3. Return the learned model representing the NGM

    Normalize X and select the best model using K-fold CV. 

    Fit the MLP on the input data X to get the `neural' view of NGM 
    while maintaining the conditional independence structure defined 
    by the complement structure matrix Sc. Does cross-validation to 
    get better generalization.

    Args:
        G (nx.Graph): Conditional independence graph.
        X (pd.DataFrame): Samples(M) x Features(D).
        lambd (float): reg_loss + lambd * structure_loss
            Recommended lambd=1 as the losses are scaled to the same range.
        hidden_dim (int): The size of the hidden unit of the MLP. 
            Each layer will have the same value.
        epochs (int): The training epochs number.
        lr (float): Learning rate for the optimizer.
        norm_type (str): min_max/mean
        k_fold (int): #splits for the k-fold CV.
        structure_penalty (str): 'hadamard':||prodW * Sc||, 'diff':||prodW-S||
        VERBOSE (bool): if True, prints to output.
        BATCH_SIZE (int): If None, take all data
        
    Returns:
        model_NGM (list): [
            model (torch.nn.object): A MLP model for NGM's `neural' view,
            scaler (sklearn object): Learned normalizer for the input data,
            feature_means (pd.Series): [feature:mean val]
        ]
    """
    # Select the device for training
    # global device
    device = torch.device("cuda") if USE_CUDA else torch.device("cpu") 
    print(f'Using "{device}" compute')
    # Get the graph structure
    S = nx.to_pandas_adjacency(G)

    # Arrange the columns of X to match the adjacency matrix
    X = X[S.columns]
    feature_means = X.mean()
    print(f'Means of selected features {feature_means, len(feature_means)}')
    # Normalize the data
    print(f'Normalizing the data: {norm_type}')
    # X, scaler = dp.process_data_for_CI_graph(X, norm_type, drop_duplicate=False)
    X, scaler = dp.normalize_table(X, norm_type)
    # Converting the data to torch 
    X = dp.convertToTorch(np.array(X), req_grad=False)
    # Get the frame of the graph structure and send to device
    if structure_penalty=='hadamard':
        # Get the complement of S (binarized)
        S = (S==0).astype(int)
        S = dp.convertToTorch(np.array(S), req_grad=False).to(device)
    elif structure_penalty=='diff':
        # Binarize the adjacency matrix S
        S = (S!=0).astype(int)
        S = dp.convertToTorch(np.array(S), req_grad=False).to(device)
    else:
        print(f'Structure penalty type {structure_penalty} is not defined')
        sys.exit(0)
    # print(f'Sending the data to {device}')
    # X = X.to(device)
    # print(f'The data is in {device}')
    M, D = X.shape
    # Splitting into k-fold for cross-validation 
    n_splits = k_fold if k_fold > 1 else 2
    kf = KFold(n_splits=n_splits, shuffle=True)
    # For each fold, collect the best model and the test-loss value
    results_Kfold = {}
    for _k, (train, test) in enumerate(kf.split(X)):
        if _k >= k_fold: # No CV if k_fold=1
            continue
        if VERBOSE: print(f'Fold num {_k}')
        if VERBOSE: print(f'Sending the data to {device}')
        X_train, X_test = X[train].to(device), X[test].to(device) # KxD, (M-K)xD
        if VERBOSE: print(f'The data is in {device}, grad should be False: {X_train.requires_grad}')
        # Initialize the MLP model
        if VERBOSE: print(f'Initializing the NGM model')
        # Send model to device
        model = neural_view.DNN(I=D, H=hidden_dim, O=D)
        model = model.to(device)
        optimizer = neural_view.get_optimizers(model, lr=lr)
        if VERBOSE: print(f'NGM model initialized {model}')

        # TODO: Add base initialization only on the regression loss
        # model = base_initialization_NGM(model, X_train)

        # Defining optimization & model tracking parameters
        best_test_loss = np.inf
        PRINT = int(epochs/10) # will print only 10 times
        lambd_increase = int(epochs/10)
        # updating with the best model and loss for the current fold
        results_Kfold[_k] = {}

        # Training the NGM model
        # For each epoch, go through the entire batch of data
        for e in range(epochs):
            # TODO: Keep increasing the lambd penalty as epochs proceed
            # if not e % lambd_increase:
            #     lambd *= 10 # increase in lambd value
            #     print(f'epoch={e}, lambda={lambd}')
            if BATCH_SIZE is None:
                X_train_batch, X_test_batch = X_train, X_test
            else:
                # randomly sample train & test points
                X_train_batch = X_train[np.random.choice(len(X_train), BATCH_SIZE, replace=False)]
                X_test_batch = X_test[np.random.choice(len(X_test), BATCH_SIZE, replace=False)]
            
            # reset the grads to zero
            optimizer.zero_grad()
            # calculate the loss for train data
            _, loss_train, reg_loss_train, struct_loss_train = forward_NGM(
                X_train_batch, 
                model, 
                S,
                structure_penalty,
                lambd=lambd
            )
            # calculate the backward gradients
            loss_train.backward()
            # updating the optimizer params with the grads
            optimizer.step()
            # Printing output
            if not e%PRINT and VERBOSE: 
                print(f'\nFold {_k}: epoch:{e}/{epochs}')
                print(f'Train: loss={dp.t2np(loss_train)}, reg={dp.t2np(reg_loss_train)}, struct={dp.t2np(struct_loss_train)}')
            if e==0 or e%100==99 or (not e%PRINT):# EVERY 100th epoch, save the best model.
                with torch.no_grad(): # prediction on test 
                    model.eval()
                    _, loss_test, reg_loss_test, struct_loss_test = forward_NGM(
                        X_test_batch, 
                        model, 
                        S,
                        structure_penalty, 
                        lambd=lambd 
                    )
                model.train()
                if VERBOSE: print(f'Test: loss={dp.t2np(loss_test)}, reg={dp.t2np(reg_loss_test)}, struct={dp.t2np(struct_loss_test)}')
                # Updating the best model for this fold
                _loss_test = dp.t2np(loss_test)
                # if e==0 or (_loss_test < best_test_loss and e%100==99): # EVERY 100th epoch, update the model.
                if _loss_test < best_test_loss: 
                    results_Kfold[_k]['best_model_updates'] = f'Fold {_k}: epoch:{e}/{epochs}:\n\
                        Train: loss={dp.t2np(loss_train)}, reg={dp.t2np(reg_loss_train)}, struct={dp.t2np(struct_loss_train)}\n\
                        Test: loss={dp.t2np(loss_test)}, reg={dp.t2np(reg_loss_test)}, struct={dp.t2np(struct_loss_test)}'
                    # if VERBOSE and not e%PRINT or e==epochs-1:
                        # print(f'Fold {_k}: epoch:{e}/{epochs}: Updating the best model with test loss={_loss_test}')
                    best_model_kfold = copy.deepcopy(model)
                    best_test_loss = _loss_test
                # else: # loss increasing, reset the model to the previous best
                #     # print('re-setting to the previous best model')
                #     model = best_model_kfold
                #     optimizer = neural_view.get_optimizers(model, lr=lr)
        results_Kfold[_k]['test_loss'] = best_test_loss
        results_Kfold[_k]['model'] = best_model_kfold
        if VERBOSE: print('\n')
    # Select the model from the results Kfold dictionary 
    # with the best score on the test fold.
    best_loss = np.inf
    for _k in results_Kfold.keys():
        curr_loss = results_Kfold[_k]['test_loss']
        if curr_loss < best_loss:
            model = results_Kfold[_k]['model']
            best_loss = curr_loss
            best_model_details = results_Kfold[_k]["best_model_updates"]

    print(f'Best model selected: {best_model_details}')
    # Checking the structure of the prodW and Sc
    prod_W = dp.t2np(product_weights_MLP(model))
    print(f'Structure Check: prodW={prod_W}, S={S}')
    # print(f'Structure Check: prodW={prod_W}, S={(np.array(S)!=0).astype(int)}')
    return [model, scaler, feature_means]



def fit_regression_direct(
    model_NGM, 
    Xy, 
    target_feature, 
    VERBOSE=True,
    BATCH_SIZE=1000,
    USE_CUDA=True
    ):
    """Directly run model.predict()

    Args:
        model_NGM (list): [
            model (torch.nn.object): A MLP model for NGM's `neural' view,
            scaler (sklearn object): Learned normalizer for the input data,
            feature_means (pd.Series): [feature:mean val]
        ]
        node_feature_dict (dict): {'name':value}.
        unknown_val (str): The marker for the unknown value.
        lr (float): Learning rate for the optimizer.
        max_itr (int): For the convergence.
        VERBOSE (bool): enable/disable print statements.
        reg_loss_th (float): The threshold for reg loss convergence.

    Returns:
        Xpred (pd.DataFrame): Predictions for the unobserved features.
            {'feature name': pred-value} 
    """
    device = torch.device("cuda") if USE_CUDA else torch.device("cpu") 
    print(f'Using "{device}" compute')
    B = min(BATCH_SIZE, Xy.shape[0])
    numB = int(np.ceil(Xy.shape[0]/B))
    # Get the NGM params
    model, scaler, feature_means = model_NGM
    model = model.to(device)
    # Get the feature names and input dimension 
    D = len(feature_means)
    feature_names = feature_means.index
    # Arrange the columns of input data to match feature means
    Xy = Xy[feature_names]
    # Eval mode
    model.eval()
    # Freeze the model weights
    for p in model.parameters():
        p.requires_grad = False
    
    # initialize the target feature as the mean value (SHOULD NOT MATTER)
    Xy[target_feature] = feature_means[target_feature]  # BxD
    # Scale the input and create a tensor
    Xi_all = dp.convertToTorch(scaler.transform(Xy), req_grad=False).to(device)
    print(Xi_all.shape) # B_allxD
    # Minimizing for the regression loss for the known values.
    Xp_batch = []
    for b in range(numB):
        # print(f'Batch {b}/{numB}')
        # Creating the tensor input to the MLP model
        Xi = Xi_all[b*B:, :] if b==numB-1 else Xi_all[b*B:(b+1)*B, :]
        # Running the NGM model 
        Xp = model.MLP(Xi)
        # Collect the predictions
        Xp_batch.extend(dp.t2np(Xp))
    # inverse normalize the prediction
    Xpred = dp.inverse_norm_table(Xp_batch, scaler)
    Xpred = pd.DataFrame(Xpred, columns=feature_names)
    return Xpred



######################################################################
# Images: Manifold learning & Kaleidoscopic sampling
######################################################################

def forward_DNN(X, model, model_type='MLP', image_metadata=None):
    # 1. Running the NN model, X = B x D
    if model_type in ['MLP', 'Transformer']:
        Xp = model.MLP(X)
        # print(f'cHECK: {X.shape, Xp.shape}')
    elif model_type in ['CNN', 'CNN-bottleneck', 'UNet']:
        # print(f'{X.shape, *image_metadata}')
        Xp = model.CNN(X.reshape(-1, *image_metadata))
        Xp = Xp.reshape(Xp.shape[0], -1)
        # print(Xp, Xp.shape)
    else:
        print(f'model type {model_type} is not defined')
    # 2. Calculate the regression loss
    mse = nn.MSELoss() 
    # D = X.shape[-1]
    reg_loss = mse(Xp, X)#/D  # since NN o/p is (0, 1), reg_loss in(0, 1)
    return Xp, reg_loss


def set_model_grad(model, req_grad=True):
    for _name, _param in model.named_parameters():
        _param.requires_grad = req_grad
        # print(_name, _param.requires_grad)


def manifold_learning_image(
    X,
    hidden_dim=20,
    epochs=1200, 
    lr=0.001,
    NORM=None,
    k_fold=5,
    VERBOSE=True, 
    BATCH_SIZE=None,
    USE_CUDA=None,   
    image_metadata=[1, 28, 28], 
    model_type='MLP',  # 'CNN'
    pretrained_model=None,
    ):
    """Fit an autoencoder to learn the data representation from X->X. 
    Return the learned model representing the Manifold. 
    Does cross-validation to get better generalization.

    Args:
        X (pd.DataFrame): Samples(M) x Features(D).
        hidden_dim (int): The size of the hidden unit of the MLP. 
            Each layer will have the same value.
        epochs (int): The training epochs number.
        lr (float): Learning rate for the optimizer.
        NORM (str): min_max/mean/None
        k_fold (int): #splits for the k-fold CV.
        VERBOSE (bool): if True, prints to output.
        BATCH_SIZE (int): If None, take all data
        USE_CUDA (str): None/"cuda:x"
        
    Returns:
        model_NN (list): [
            model (torch.nn.object): A MLP model for NGM's `neural' view,
            scaler (sklearn object): Learned normalizer for the input data,
            feature_means (pd.Series): [feature:mean val]
        ]
    """ 
    # Select the device for training
    # While using USE_CUDA_DEVICES, change USE_CUDA="cuda"
    device = torch.device(USE_CUDA) if USE_CUDA else torch.device("cpu") 
    print(f'Using "{device}" compute')
    feature_means = X.mean()
    print(f'Means of selected features {feature_means, len(feature_means)}')
    # Normalize the data
    print(f'Normalizing the data: {NORM}')
    X, scaler = dp.normalize_table(X, NORM)
    # Converting the data to torch 
    X = dp.convertToTorch(np.array(X), req_grad=False, device=device)
    print(X)
    M, D = X.shape
    # Initialize the MLP model
    if pretrained_model: 
        if VERBOSE: print(f'Using the Pre-trained model as initial seed')
        model, scaler, feature_means = pretrained_model
        model.train()
        for _name, _param in model.named_parameters():
            _param.requires_grad = True
            print(_name, _param.requires_grad)
    else:
        if VERBOSE: print(f'Initializing a DNN model')
        model = neural_view.DNN(I=D, H=hidden_dim, O=D, model_type=model_type, image_metadata=image_metadata)
    model = model.to(device)
    optimizer = neural_view.get_optimizers(model, lr=lr)
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Manifold model initialized {model} with params {total_params}')
    # Defining optimization & model tracking parameters
    min_loss = np.inf
    PRINT = int(epochs/10) # will print only 10 times
    epoch_vs_loss = []  # [(epoch, loss), ...]
    # For each epoch, go through a batch of data
    X_batch = X
    for e in range(epochs):
        if not e%PRINT:
            print(f'epoch {e}')
        # Stochastic sampling of learning datapoints
        if BATCH_SIZE:
            X_batch = X[np.random.choice(len(X), BATCH_SIZE, replace=False)]
        # Converting the data to torch 
        # X_batch = dp.convertToTorch(np.array(X_batch), req_grad=False)
        # reset the grads to zero
        optimizer.zero_grad()
        # calculate the loss for train data
        Xp, loss = forward_DNN(X_batch, model, model_type, image_metadata) 
        # calculate the backward gradients
        loss.backward()
        # updating the optimizer params with the grads
        optimizer.step()
        loss = dp.t2np(loss)
        # Xp = Xp.detach()
        # X_batch = X_batch.detach()
        if not e%PRINT and VERBOSE: 
            with torch.no_grad(): # prediction on test 
                model.eval()
                print(f'\n epoch:{e}/{epochs}, loss={loss}')
                print(f'Train images {X_batch.shape}') 
                disp_seed = np.random.choice(1000, size=1)
                dp.tensor_to_images(X_batch, image_metadata, display_N=5, seed=disp_seed)
                # dp.pixel_embeddings_to_images(X_batch, image_metadata, display_N=5, seed=disp_seed)
                # Visualize some test images
                print(f'Recovered images {Xp.shape}')
                # dp.pixel_embeddings_to_images(Xp, image_metadata, display_N=5, seed=disp_seed)
                dp.tensor_to_images(Xp, image_metadata, display_N=5, seed=disp_seed)
            model.train()
        if e==0 or e==epochs-1 or e%100==99 or (not e%PRINT):
            # EVERY 100th epoch, save the best model.
            if loss < min_loss: 
                print(f'Updating the best model')
                best_model = copy.deepcopy(model)
                set_model_grad(best_model, req_grad=False)
                min_loss = loss
        # Record the loss trend for analysis
        epoch_vs_loss.append([e, loss])
    if VERBOSE: 
        with torch.no_grad(): # prediction on test 
            model.eval()
            print('\n Train data fit')
            dp.tensor_to_images(Xp, image_metadata, display_N=48)
    epoch_vs_loss = np.array(epoch_vs_loss)
    dp.plot_xy(epoch_vs_loss[:, 0], epoch_vs_loss[:, 1], 'epochs', 'loss')
    # print(f'Freeze the model weights {model}')
    # set_model_grad(model, req_grad=False)
    return [model, scaler, feature_means]
    # gc.collect()
    # torch.cuda.empty_cache()
    # return [best_model, scaler, feature_means]


def get_kaleidoscopic_samples(
        model, 
        NUM_SAMPLES=1000, 
        SAMPLING_RUNS=300, 
        model_type='MLP', 
        image_metadata=[1,28,28],
        SHOW_EVERY=1,
        NOISE='uniform',
        USE_CUDA=False
    ):
    # Select the device for training
    # While using USE_CUDA_DEVICES, change USE_CUDA="cuda"
    device = torch.device(USE_CUDA) if USE_CUDA else torch.device("cpu") 
    print(f'Using "{device}" compute')
    C, D, D = image_metadata
    mean, std = 0.5, 0.5
    if NOISE=='normal':
        X_in = np.clip(np.random.normal(mean, std, size=[NUM_SAMPLES, C*D*D]), 0, 1)
    elif NOISE=='uniform':
        X_in = np.clip(np.random.uniform(0, 1, size=[NUM_SAMPLES, C*D*D]), 0, 1)
    X_in = dp.convertToTorch(np.array(X_in), req_grad=False, device=device)
    with torch.no_grad(): 
        model = model.to(device)
        model.eval()
        for _name, _param in model.named_parameters():
            _param.requires_grad = False
            print(_name, _param.requires_grad)
        disp_seed = np.random.choice(1000, size=1)
        for r in range(SAMPLING_RUNS):
            X_in, _ = forward_DNN(
                X_in, model, model_type, image_metadata
            )
            if r%SHOW_EVERY==0:
                print(f'Sampling run {r}/{SAMPLING_RUNS}')
                dp.tensor_to_images(X_in, image_metadata, display_N=48, seed=disp_seed)
    return 


# ********************************************************************


######################################################################
# Synthetic: Manifold learning & Kaleidoscopic sampling
######################################################################

def manifold_learning_MLP(
    X,
    hidden_dim=20,
    epochs=1200, 
    lr=0.001,
    NORM=None,
    VERBOSE=True, 
    BATCH_SIZE=None,
    USE_CUDA=True, 
    pretrained_model=None, 
    ):
    """Manifold learning using MLP. 
    1. Fit a MLP (autoencoder) to learn the data representation from X->X. 
    2. Return the learned model representing the model

    Args:
        X (pd.DataFrame): Samples(M) x Features(D).
        hidden_dim (int): The size of the hidden unit of the MLP. 
            Each layer will have the same value.
        epochs (int): The training epochs number.
        lr (float): Learning rate for the optimizer.
        VERBOSE (bool): if True, prints to output.
        BATCH_SIZE (int): If None, take all data.
        USE_CUDA (bool): If True, use GPU. 
        pretrained_model (object): if input model, use it as init. 
        
    Returns:
        List: [
            model (torch.nn.object): A MLP model after manifold learning,
            scaler (sklearn object): Learned normalizer for the input data,
            feature_means (pd.Series): [feature:mean val]
        ]
    """
    # Select the device for training
    device = torch.device("cuda") if USE_CUDA else torch.device("cpu") 
    print(f'Using "{device}" compute')
    feature_names = X.columns
    feature_means = X.mean()
    X, scaler = dp.normalize_table(X, NORM)
    # Converting the data to torch 
    X = dp.convertToTorch(np.array(X), req_grad=False).to(device)
    M, D = X.shape
    if VERBOSE: print(f'Data is in {device}, grad is False: {X.requires_grad}')
    # Initialize the MLP model
    if pretrained_model: 
        if VERBOSE: print(f'Using the Pre-trained model as initial seed')
        model, scaler, feature_means = pretrained_model
        model = model.to(device)
        model.train()
        for _name, _param in model.named_parameters():
            _param.requires_grad = True
            print(_name, _param.requires_grad)
    else:
        if VERBOSE: print(f'Initializing a MLP model')
        model = neural_view.DNN(I=D, H=hidden_dim, O=D)
        model = model.to(device)
    # Defining optimization & model tracking parameters
    optimizer = neural_view.get_optimizers(model, lr=lr)
    if VERBOSE: print(f'MLP model initialized {model}')
    PRINT = int(epochs/10) # will print only 10 times
    # Fit the MLP model via stochastic optimization
    # For each epoch, go through the entire batch of data
    min_loss = np.inf
    epoch_vs_loss = []  # [(epoch, loss), ...]
    for e in range(epochs):
        # Stochastic sampling of learning datapoints
        X_batch = X
        if BATCH_SIZE:
            X_batch = X[np.random.choice(len(X), BATCH_SIZE, replace=False)]         
        # reset the grads to zero
        optimizer.zero_grad()
        # calculate the optimization loss
        Xp, loss = forward_DNN(X_batch, model) 
        # calculate the backward gradients
        loss.backward()
        # updating the optimizer params with the grads
        optimizer.step()
        loss = loss.detach()
        # Printing output
        if not e%PRINT and VERBOSE: 
            # curr_Xp = pd.DataFrame(dp.t2np(Xp), columns=feature_names)
            print(f'\n epoch:{e}/{epochs}, loss={dp.t2np(loss)}')#, val={dp.t2np(Xp)}')
        if e==0 or e==epochs-1 or e%100==99 or (not e%PRINT):
            # EVERY 100th epoch, save the best model.
            if loss < min_loss: 
                best_model = copy.deepcopy(model)
                min_loss = loss
        # Record the loss trend for analysis
        epoch_vs_loss.append([e, dp.t2np(loss)])
    if VERBOSE: print('\n')
    epoch_vs_loss = np.array(epoch_vs_loss)
    dp.plot_xy(epoch_vs_loss[:, 0], epoch_vs_loss[:, 1], 'epochs', 'loss')
    print('Clear the RAM after training')
    # gc.collect()
    # torch.cuda.empty_cache()
    return [best_model, scaler, feature_means]


def kaleidoscopic_sampling_synthetic(
    model_NN,
    NUM_SAMPLES=1000, 
    SAMPLING_RUNS=5, 
    model_type='MLP', 
    image_metadata=[1,28,28]
    ):
    # Get the NN params
    model, scaler, feature_means = model_NN
    feature_names = feature_means.index
    C, W, H = image_metadata
    mean, std = 0.5, 0.5
    X_in = np.clip(np.random.normal(mean, std, size=[NUM_SAMPLES, C*W*H]), 0, 1)
    input_noise = pd.DataFrame(X_in, columns=feature_names)
    X_in = dp.convertToTorch(np.array(X_in), req_grad=False, use_cuda=False)
    with torch.no_grad(): 
        model.eval()
        recovered_samples = {}
        for r in range(SAMPLING_RUNS):
            print(f'kaleioscopic sampling run: {r}/{SAMPLING_RUNS}')
            X_in, _ = forward_DNN( #ggn_loss_function(
                X_in, model, model_type, image_metadata
            )
            # dp.pixel_embeddings_to_images(
            #     X_in, image_metadata, display_N=48, seed=27
            # )

            recovered_samples[r] = pd.DataFrame(np.array(X_in), columns=feature_names)
    return recovered_samples, input_noise
# ********************************************************************