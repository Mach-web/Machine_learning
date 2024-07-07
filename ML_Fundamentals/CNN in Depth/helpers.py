import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize_filters(filters):
    
    fig, subs = plt.subplots(1, len(filters), figsize=(10, 5))
    
    for i, ax in enumerate(subs.flatten()):    
        
        ax.imshow(filters[i], cmap='gray')
        ax.set_title('Filter %s' % str(i+1))
        ax.axis("off")
        
        width, height = filters[i].shape
        
        for x in range(width):
            
            for y in range(height):
                
                ax.annotate(str(filters[i][x][y]), xy=(y,x),
                            horizontalalignment='center',
                            verticalalignment='center',
                            color='white' if filters[i][x][y]<0 else 'black')


def show_feature_maps(input_img, feature_maps, filters):
    
    # Setup visualization grid
    gs_kw = dict(height_ratios=[2, 0.5, 2])
    fig, subs_dict = plt.subplot_mosaic(
        '''
        L....AAAA....
        M.B..C..D..E.
        NFFFGGGHHHIII
        ''',
        figsize=(15, 10),
        gridspec_kw=gs_kw
    )
    
    # plot original image
    subs_dict['A'].imshow(input_img, cmap='gray')
    subs_dict['A'].axis("off")

    # visualize all filters
    for i, p in enumerate(['B', 'C', 'D', 'E']):
        ax = subs_dict[p]
        ax.imshow(filters[i], cmap='gray')
        ax.set_title('Filter %s' % str(i+1))
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Visualize feature maps corresponding to each filter
    for i, p in enumerate(['F', 'G', 'H', 'I']):
        ax = subs_dict[p]
        ax.imshow(np.squeeze(feature_maps[0, i].data.numpy()), cmap="gray")
        ax.set_title("Output %s" % str(i + 1))
        ax.axis("off")
    
    subs_dict['L'].text(0, 0.5, "Input image", {'va': 'center'}, rotation=90, fontsize=15)
    subs_dict['L'].axis("off")
    subs_dict['M'].text(0, 0.5, "Conv layer filters", {'va': 'center'},rotation=90, fontsize=15)
    subs_dict['M'].axis("off")
    subs_dict['N'].text(0, 0.5, "Feature maps", {'va': 'center'}, rotation=90, fontsize=15)
    subs_dict['N'].axis("off")
    
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    fig.tight_layout()

    
def show_feature_maps_full(input_img, model, filters):

    # Setup visualization grid
    gs_kw = dict(height_ratios=[2, 0.5, 2, 2, 2])
    fig, subs_dict = plt.subplot_mosaic(
        """
        1....AAAA....
        2.B..C..D..E.
        3FFFGGGHHHIII
        4LLLMMMNNNOOO
        5PPPQQQRRRSSS
        """,
        figsize=(15, 12),
        gridspec_kw=gs_kw,
    )

    # plot original image
    subs_dict["A"].matshow(input_img, cmap="gray")
    subs_dict["A"].axis("off")

    # visualize all filters
    for i, p in enumerate(["B", "C", "D", "E"]):
        ax = subs_dict[p]
        ax.matshow(filters[i], cmap="gray")
        ax.set_title("Filter %s" % str(i + 1))
        ax.set_xticks([])
        ax.set_yticks([])

    # Visualize feature maps corresponding to each filter
    # Get feature maps (only convolution)
    x = (
        torch.from_numpy(input_img)
        # Add one dimension for batch
        .unsqueeze(0)
        # Add dimension for n_channels
        .unsqueeze(1)
    )
    feature_maps = model.conv_layer(x)

    for i, p in enumerate(["F", "G", "H", "I"]):
        ax = subs_dict[p]
        ax.matshow(np.squeeze(feature_maps[0, i].data.numpy()), cmap="gray")
        ax.set_title("Output %s" % str(i + 1))
        ax.axis("off")

    # Visualize feature maps after activation
    feature_maps = model.activation(model.conv_layer(x))
    for i, p in enumerate(["L", "M", "N", "O"]):
        ax = subs_dict[p]
        ax.matshow(np.squeeze(feature_maps[0, i].data.numpy()), cmap="gray")
        ax.set_title("Output %s" % str(i + 1))
        ax.axis("off")

    # Visualize feature maps after max pool
    final_res = model(x)
    for i, p in enumerate(["P", "Q", "R", "S"]):

        data = np.squeeze(final_res[0, i].data.numpy())

        ax = subs_dict[p]
        ax.matshow(data, cmap="gray")
        ax.set_title("Output %s" % str(i + 1))
        ax.axis("off")

        half_x_diff = (feature_maps[0, i].data.shape[1] - data.shape[1]) / 2
        half_y_diff = (feature_maps[0, i].data.shape[0] - data.shape[0]) / 2
        ax.set_xlim((lim - half_x_diff for lim in subs_dict["L"].get_xlim()))
        ax.set_ylim((lim - half_y_diff for lim in subs_dict["L"].get_ylim()))

    subs_dict["1"].text(
        0, 0.5, "Input image", {"va": "center"}, rotation=90, fontsize=15
    )
    subs_dict["1"].axis("off")
    subs_dict["2"].text(
        0, 0.5, "Conv layer filters", {"va": "center"}, rotation=90, fontsize=15
    )
    subs_dict["2"].axis("off")
    subs_dict["3"].text(
        0, 0.5, "Feature maps\nafter conv", {"va": "center"}, rotation=90, fontsize=15
    )
    subs_dict["3"].axis("off")
    subs_dict["4"].text(
        0, 0.5, "Feature maps\nafter ReLU", {"va": "center"}, rotation=90, fontsize=15
    )
    subs_dict["4"].axis("off")
    subs_dict["5"].text(
        0,
        0.5,
        "Feature maps\nafter MaxPool",
        {"va": "center"},
        rotation=90,
        fontsize=15,
    )
    subs_dict["5"].axis("off")

    plt.subplots_adjust(wspace=0.01, hspace=0.01)

    fig.tight_layout()