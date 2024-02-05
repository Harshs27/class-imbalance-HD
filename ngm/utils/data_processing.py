"""
Data processing and post-processing
functions for Manifold Learning.
"""

from sklearn import preprocessing
from torchvision import datasets, transforms

import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision

from ngm.main import forward_DNN


# **************** Functions for Class imbalance ********************
def complete_graph_from_list(L):
    G=nx.complete_graph(len(L))
    nx.relabel_nodes(G,dict(enumerate(L)), copy = False)
    return G


# ************** plotting functions for 1D, 2D ***********************
def plot_loss_vs_epoch(epoch_vs_loss, epoch_vs_test_loss):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    plt.plot(epoch_vs_loss[:, 0], epoch_vs_loss[:, 1], 'b', label='train')
    plt.plot(epoch_vs_test_loss[:, 0], epoch_vs_test_loss[:, 1], 'o', label='test')
    plt.title('Loss trend', fontsize=30)
    plt.xlabel('epochs', fontsize=25)
    plt.ylabel('loss', fontsize=25)
    plt.legend(loc='upper right', fontsize=20)
    ax.tick_params(axis='both', labelsize=25)
    plt.show()


def plot_xy(x, y, x_label='', y_label='', title='', fig=None, scale_y_axis=False, title_font=30):
    # setting the axes at the centre
    if fig is None: fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['left'].set_position('zero') # 'center'
    ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    if scale_y_axis: ax.set_ylim([0, 1])
    # plot the function
    plt.plot(x, y, 'b')
    if len(title)>0: plt.title(title, fontsize=title_font, y=1.05)
    plt.xlabel(x_label, fontsize=25)
    plt.ylabel(y_label, fontsize=25)
    ax.tick_params(axis='both', labelsize=25)
    # fig.tight_layout()


def evaluate_model_loss(model_NN, x, model_type='MLP', image_metadata=[1,1,1]):
    model, _, _ = model_NN
    x = convertToTorch(np.array(x), req_grad=False)
    fx, _ = forward_DNN(x, model, model_type, image_metadata)
    # print(x, fx.shape)
    D = x.shape[-1]
    loss = torch.linalg.norm(x-fx, ord=2, dim=1)**2/D
    # mse = nn.MSELoss()
    # loss = mse(x, fx)
    loss = t2np(loss)
    return loss


def plot_loss_function_NN(model_NN, x, title='', SAVE=False, scale_y_axis=False, model_type='MLP', image_metadata=[1,1,1]):
    """Vary x and see the loss value. Gives an idea of 
    how well the NN fits the input data. 
    """
    loss = evaluate_model_loss(model_NN, x, model_type=model_type, image_metadata=image_metadata)
    # print(loss)
    plot_xy(x, loss, x_label='input', y_label='loss', title=title, scale_y_axis=scale_y_axis)
    # show the plot
    if SAVE: plt.savefig('plot.jpg', dpi=300)
    plt.show()
    return loss


def evaluate_model_output(model_NN, x, model_type='MLP', image_metadata=[1,1,1]):
    model, _, _ = model_NN
    x = convertToTorch(np.array(x), req_grad=False)
    fx, _ = forward_DNN(x, model, model_type=model_type, image_metadata=image_metadata)
    fx = t2np(fx)
    return fx


def plot_output_NN(model_NN, x, title='', SAVE=False, scale_y_axis=False, model_type='MLP', image_metadata=[1,1,1]):
    """Vary x and see the loss value. Gives an idea of 
    how well the NN fits the input data. 
    """
    fx = evaluate_model_output(model_NN, x, model_type=model_type, image_metadata=image_metadata)
    print(x, fx)
    # plot_xy(x, fx, x_label='input', y_label='output', title=title)

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(1, 1, 1)
    # ax.spines['left'].set_position('zero') # 'center'
    # ax.spines['bottom'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    # ax.xaxis.set_ticks_position('bottom')
    # ax.yaxis.set_ticks_position('left')
    if scale_y_axis: ax.set_ylim([0, 1])
    # plot the function
    plt.plot(x, fx, 'b')
    if len(title)>0: plt.title(title, fontsize=30)
    plt.xlabel('input', fontsize=25)
    plt.ylabel('output', fontsize=25)
    ax.tick_params(axis='both', labelsize=25)

    # show the plot
    if SAVE: plt.savefig('plot.jpg', dpi=300)
    plt.show()
    return fx


def plot_loss_function_and_samples_NN(func, points, model_NN, title='', SAVE=False, scale_y_axis=False, title_font=30, model_type='MLP', image_metadata=[1,1,1]):
    x, loss = func
    recovered_samples, input_noise = points
    loss_noise = evaluate_model_loss(model_NN, input_noise, model_type=model_type, image_metadata=image_metadata)
    loss_samples = evaluate_model_loss(model_NN, recovered_samples, model_type=model_type, image_metadata=image_metadata)
    # print(x, loss, recovered_samples, input_noise, loss_noise, loss_samples)
    plot_xy(x, loss, x_label='Input', y_label='Loss', title=title, scale_y_axis=scale_y_axis, title_font=title_font)
    # Plot the points. Add legend
    plt.scatter(input_noise, loss_noise, color='red', marker='o', s=100, label='noise')
    # print(input_noise)
    for pt, l in zip(np.array(input_noise).reshape(-1), loss_noise):
        # print(f' (x, fx) = {pt, l}')
        plt.vlines(x=pt, ymax=l, ymin=0, colors='orange', ls='--', lw=1)
    plt.scatter(recovered_samples, loss_samples, marker='o', s=100, color='green', label='samples')
    plt.legend(loc='upper right', fontsize=20)#, bbox_to_anchor=(1.1, 0.5), labelspacing=3)
    # show the plot
    if SAVE: plt.savefig('plot.jpg', dpi=300)
    plt.show()


def plot_xyz(X, Y, Z, x_label='X', y_label='Y', z_label='Z', title=''):
    # fig = plt.figure(figsize=(5, 5))
    ax = plt.figure(figsize=(10, 10)).add_subplot(projection='3d')
    # Plot the 3D surface
    ax.plot_surface(X, Y, Z, edgecolor='royalblue', lw=0.5, alpha=0.7) #rstride=8, cstride=8)
    # Plot projections of the contours for each dimension.  By choosing offsets
    # that match the appropriate axes limits, the projected contours will sit on
    # the 'walls' of the graph.
    ax.contour(X, Y, Z, zdir='z', offset=-0.1, cmap='coolwarm')
    ax.contour(X, Y, Z, zdir='x', offset=-0.2, cmap='coolwarm')
    # ax.contour(X, Y, Z, zdir='y', offset=0, cmap='coolwarm')
    ax.set(xlim=(-0.2, 1.1), ylim=(-0.0, 1.1), zlim=(-0.1, 1.0))
    # ax.set(xlim=(-0.2, 1.1), ylim=(-0.0, 1.1), zlim=(-2, 2.0))
    plt.title(title, fontsize=30, y=0.97)
    ax.set_xlabel(x_label, fontsize=25, rotation=0, labelpad=15)
    ax.set_ylabel(y_label, fontsize=25, rotation=0, labelpad=15)
    ax.set_zlabel(z_label, fontsize=25, rotation=0, labelpad=15)
    # plt.zlabel(z_label, fontsize=25)
    ax.tick_params(axis='x', labelsize=20)
    ax.tick_params(axis='y', labelsize=20)
    ax.tick_params(axis='z', labelsize=20)
    # ax.view_init(elev=20., azim=-60)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.subplots_adjust(top=1.0)
    return ax
         

def plot_loss_function_NN_2D(model_NN, grid_range, title='', SAVE=False, model_type='MLP', image_metadata=[1,1,2]):
    """Vary x and see the loss value. Gives an idea of 
    how well the NN fits the input data. 
    """
    x, y = grid_range
    pt_xy = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1)], axis=1)
    loss = evaluate_model_loss(model_NN, pt_xy, model_type=model_type, image_metadata=image_metadata)
    loss = loss.reshape(x.shape)
    ax = plot_xyz(x, y, loss, x_label='X', y_label='Y', z_label='loss', title=title)
    ax.view_init(elev=20., azim=-60)
    # show the plot
    if SAVE: plt.savefig('plot.jpg', dpi=300)
    plt.show()
    return loss


# def plot_output_NN_2D(model_NN, grid_range, title='', SAVE=False):
#     """Vary x and see the loss value. Gives an idea of 
#     how well the NN fits the input data. 
#     """
#     x, y = grid_range
#     pt_xy = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1)], axis=1)
#     # loss = evaluate_model_loss(model_NN, pt_xy)
#     fx = evaluate_model_output(model_NN, pt_xy)
#     print(f'Check {pt_xy, fx}')
#     fx = fx.reshape(x.shape)
#     ax = plot_xyz(x, y, fx, x_label='X', y_label='Y', z_label='loss', title=title)
#     ax.view_init(elev=20., azim=-60)
#     # show the plot
#     if SAVE: plt.savefig('plot.jpg', dpi=300)
#     plt.show()
#     return loss
#     return fx


def plot_loss_function_and_samples_NN_2D(func, points, model_NN, title='', SAVE=False, model_type='MLP', image_metadata=[1,1,2]):
    grid_range, loss = func
    x, y = grid_range
    pt_xy = np.concatenate([x.reshape(-1, 1), y.reshape(-1, 1)], axis=1)
    recovered_samples, input_noise = points
    loss_noise = evaluate_model_loss(model_NN, input_noise, model_type=model_type, image_metadata=image_metadata)
    loss_samples = evaluate_model_loss(model_NN, recovered_samples, model_type=model_type, image_metadata=image_metadata)
    ax = plot_xyz(x, y, loss, x_label='X', y_label='Y', z_label='loss', title=title)
    ax.scatter(
        input_noise['d0'], input_noise['d1'], loss_noise, 
        color='red', marker='o',  label='noise', s=100
    )
    ax.scatter(
        recovered_samples['d0'], recovered_samples['d1'], loss_samples,
        marker='o', s=100, color='green', label='samples'
    )  
    plt.legend(loc='upper right', fontsize=20, bbox_to_anchor=(1, 0.9))#, labelspacing=3)
    ax.view_init(elev=20, azim=-60)  # Again as we want to change the viewpoint of scatter
    # show the plot
    if SAVE: plt.savefig('plot.jpg', dpi=300)
    plt.show()


# ************** Preparing NGM inputs (image analysis ) **************

def get_label_image_dict(dataloader):
    """ Get the input data pair (X, G) for NGM from the 
    images in the dataloader. For each pixel a naming 
    convention is followed. This can later be used to 
    define a graph connectivity structure.

    Args:
        dataloader (object): Torch oject iterator
    
    Returns
        image_X (pd.DataFrame): Images(B) x Features(D)
        image_metadata (list): [channel, width, height]
    """
    image_X = {}
    for i, data in enumerate(dataloader):
        # get the inputs; data is a list of [inputs, labels]
        images, labels = data
        # Convert images to numpy
        images = np.array(images)
        labels = np.array(labels)
        B, C, W, H = images.shape  # batch, channel, row, col
        # Reshape the image
        images = images.reshape(B, -1)
        for img, label in zip(images, labels):
            if label in image_X.keys():
                image_X[label].append(img)
            else:
                image_X[label] = [img]
    # Name the dimensions of the flattened images
    pixel_names = [] 
    for c in range(C):  # Go over the number of channels
        for w in range(W):  # row
            for h in range(H):  # col
                pixel_names.append(
                    ['c'+str(c+1)+'_w'+str(w+1)+'_h'+str(h+1)]
                )
    pixel_names = np.array(pixel_names).reshape(-1)
    # Convert to pandas dataframe format
    for label in image_X.keys():
        image_X[label] = pd.DataFrame(np.vstack(image_X[label]), columns = pixel_names)
    image_metadata = [C, W, H]  # used later for reconstruction
    # pixel_G = get_image_connectivity_graph(image_X.columns)
    return image_X, image_metadata


def prepare_image_input_manifold_learning(dataloader):
    """ Get the input data pair (X, G) for NGM from the 
    images in the dataloader. For each pixel a naming 
    convention is followed. This can later be used to 
    define a graph connectivity structure.

    Args:
        dataloader (object): Torch oject iterator
    
    Returns
        image_X (pd.DataFrame): Images(B) x Features(D)
        image_metadata (list): [channel, width, height]
    """
    image_X = []
    for i, data in enumerate(dataloader):
        # get the inputs; data is a list of [inputs, labels]
        images, labels = data
        # Convert images to numpy
        images = np.array(images)
        # print(images.shape, labels.shape)
        B, C, W, H = images.shape  # batch, channel, row, col
        # Reshape the image
        image_X.append(images.reshape(B, -1)) 
    # Name the dimensions of the flattened images
    pixel_names = [] 
    for c in range(C):  # Go over the number of channels
        for w in range(W):  # row
            for h in range(H):  # col
                pixel_names.append(
                    ['c'+str(c+1)+'_w'+str(w+1)+'_h'+str(h+1)]
                )
    pixel_names = np.array(pixel_names).reshape(-1)
    # Convert to pandas dataframe format
    image_X = pd.DataFrame(np.vstack(image_X), columns = pixel_names)
    image_metadata = [C, W, H]  # used later for reconstruction
    # pixel_G = get_image_connectivity_graph(image_X.columns)
    return image_X, image_metadata


def get_image_connectivity_graph(pixel_names):
    """ Define the graph for image pixels connectivity pattern.

    Args:
        pixel_names (np.array(str)): Contains channel number and 
            pixel number information c*_p* 
    
    Returns:
        pixel_G (nx.Graph): Pixel connectivity graph.
    """
    nodes = len(pixel_names)
    pixel_G = nx.complete_graph(pixel_names)
    return pixel_G
# ******************************************************************


# ************** PyTorch image data handling **************************
def load_image_data(dataset_name, data_path, batch_size):
    print(f'Loading {dataset_name} data')
    if dataset_name=='MNIST':
        # The output of torchvision datasets are PILImage images of range [0, 1]. 
        transform=transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.5,), (0.5,))
        ])
        trainset = datasets.MNIST(
            data_path, train=True, download=True, transform=transform
        )
        testset = datasets.MNIST(
            data_path, train=False, transform=transform
        )
    elif dataset_name=='CIFAR':
        transform = transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        trainset = datasets.CIFAR10(
            data_path, train=True, download=True, transform=transform
        )
        testset = datasets.CIFAR10(
            data_path, train=False, transform=transform
        )
    else:
        print(f'dataset {dataset_name} not available')
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    def get_details(dataloader):
        total_batches = 0
        for i, (images, labels) in enumerate(dataloader):
            total_batches += 1
        print(f'Batch: images & labels size {images.shape, labels.shape}')
        print(f'Total batches = {total_batches}')
        return None
    print('Train data details:')
    get_details(trainloader)
    print(f'Test data details:')
    get_details(testloader)
    return trainloader, testloader 


def imshow(img):  # function to show an image
    # img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def visualize_image_dataset(dataloader, display_N=4):
    # get some random training images
    dataiter = iter(dataloader)
    images, labels = next(dataiter)
    # show images (Increase grid size?)
    imshow(
        torchvision.utils.make_grid(images[:display_N])
    )
    # range of values in an image, verify the range 
    print(f'Pixels range (min, max) = {images[0].min(), images[0].max()}')
    return None


def pixel_embeddings_to_images(Xs, image_metadata, display_N=10, seed=None):
    """ The input image embeddings are converted to images.
    Each embedding is a flattened pixelated representation.

    Args:
        Xs (pd.DataFrame): samples (B) x dimensions (D)
        image_metadata (list): [C, W, H]
    """
    C, W, H = image_metadata
    # for img_embedding in Xs:  # iterate through images
    Xs = np.array(Xs).reshape(Xs.shape[0], C, W, H)
    Xs = convertToTorch(Xs)
    # randomly choose N images to display
    size = min(Xs.shape[0], display_N)
    if seed: np.random.seed(seed)
    idx = np.random.choice(range(Xs.shape[0]), size=size, replace=False)
    imshow(
        torchvision.utils.make_grid(Xs[idx])
    )
    return None


def tensor_to_images(Xs, image_metadata, display_N=10, seed=None):
    """ The input image embeddings are converted to images.
    Each embedding is a flattened pixelated representation.

    Args:
        Xs (Tensor): samples (B) x dimensions (D)
        image_metadata (list): [C, W, H]
    """
    # randomly choose N images to display
    size = min(Xs.shape[0], display_N)
    if seed: np.random.seed(seed)
    idx = np.random.choice(range(Xs.shape[0]), size=size, replace=False)
    Xs = Xs[idx].detach().cpu()
    C, W, H = image_metadata
    Xs = Xs.reshape(Xs.shape[0], C, W, H)
    imshow(torchvision.utils.make_grid(Xs))
    return None


# ********************* General data manipulation functions**************
def series2df(series):
    "Convert a pd.Series to pd.Dataframe and set the index as header."
    # Convert the series to dictionary.
    series_dict = {n:v for n, v in zip(series.index, series.values)}
    # Create the dataframe from series and transpose.
    df = pd.DataFrame(series_dict.items()).transpose()
    # Set the index row as header and drop it from values.
    df.columns = df.iloc[0]
    df = df.drop(df.index[0])
    return df


def t2np(x):
    "Convert torch to numpy"
    return x.detach().cpu().numpy()


def convertToTorch(data, req_grad=False, device="cpu"):
    """Convert data from numpy to torch variable, if the req_grad
    flag is on then the gradient calculation is turned on.
    """
    if not torch.is_tensor(data):
        data = torch.from_numpy(data).type(torch.FloatTensor)
    data.requires_grad = req_grad
    return data.to(device)


def old_convertToTorch(data, req_grad=False, use_cuda=False):
    """Convert data from numpy to torch variable, if the req_grad
    flag is on then the gradient calculation is turned on.
    """
    if not torch.is_tensor(data):
        dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        data = torch.from_numpy(data).type(dtype) # .astype(np.float, copy=False)
    data.requires_grad = req_grad
    return data


def normalize_table(X, method='min_max'):
    """Normalize the input data X.

    Args:
        X (pd.Dataframe): Samples(M) x Features(D).
        methods (str): min_max/mean 

    Returns:
        Xnorm (pd.Dataframe): Samples(M) x Features(D).
        scaler (object): The scaler to scale X
    """
    if method=='min_max':
        scaler = preprocessing.MinMaxScaler()
    elif method=='mean':
        scaler = preprocessing.StandardScaler()
    else: # none
        print(f'Scaler not applied')
        scaler = None
    # Apply the scaler on the data X
    Xnorm = scaler.fit_transform(X) if scaler else X
    # Convert back to pandas dataframe
    Xnorm = pd.DataFrame(Xnorm, columns=X.columns)
    return Xnorm, scaler


def inverse_norm_table(Xnorm, scaler):
    """
    Apply the inverse transform on input normalized
    data to get back the original data.
    """
    return scaler.inverse_transform(Xnorm) if scaler else Xnorm

# ***********************************************************************