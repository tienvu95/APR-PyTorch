

# @article{gaurav2018hypernetsgithub,
#   title={HyperNetworks(Github)},
#   author={{Mittal}, G.},
#   howpublished = {\url{https://github.com/g1910/HyperNetworks}},
#   year={2018}
# }

#aim of this HyperNetwork

#take the embedding of u,i,j
#objective function = same as the adversarial training, to miminize the worst case cross entropy loss

# how should we proceed with model architecture

#objective of hypernet is similar to adv_loss

import torch
import torch.nn as nn
from keras.layers import Concatenate

def HyperNetwork(u):
    # mlp_vector=torch.stack((u,i,j))

    linear_relu_stack = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    return linear_relu_stack(u)
    # num_layer=len(layers)
    #
    # user_input=Input(shape=(1,),dtype='int32')
    # item_input_pos=Input(shape=(1,),dtype='int32')
    # item_input_neg = Input(shape=(1,), dtype='int32')
    #
    # ## in this context each point is projected to have 10 dimensional coordinates?
    # MF_embedding_user=Embedding(input_dim=num_users,output_dim=mf_dim,embeddings_initializer='random_normal',
    #                             name='mf_user_embedding',embeddings_regularizer=l2(reg_mf),input_length=1)
    # MF_embedding_item = Embedding(input_dim=num_items, output_dim=mf_dim, embeddings_initializer='random_normal',
    #                               name='mf_item_embedding',embeddings_regularizer=l2(reg_mf), input_length=1)
    # MLP_embedding_user=Embedding(input_dim=num_users,output_dim=layers[0],embeddings_initializer='random_normal',
    #                              name='mlp_user_embedding', embeddings_regularizer=l2(reg_mf),input_length=1)
    # MLP_embedding_item = Embedding(input_dim=num_items, output_dim=layers[0], embeddings_initializer='random_normal',
    #                                name='mlp_item_embedding',embeddings_regularizer=l2(reg_mf), input_length=1)
    #
    # mf_user_latent=Flatten()(MF_embedding_user(u))
    # mf_item_latent_pos=Flatten()(MF_embedding_item(i))
    # mf_item_latent_neg = Flatten()(MF_embedding_item(j))

    ## merge = deprecated use keras.layers.Concatenate(axis=-1) instead
    # x_ui = torch.mul(u, i).sum(dim=1)
    # x_uj = torch.mul(u, j).sum(dim=1)
    # ## convert negative layer to negative, lambda layers should be re-written as subclass layer if too complex (eg: multiply by scale?)
    # prefer_neg = Lambda(lambda x: -x)(prefer_neg)
    ## basically just merge 2 layer
    # mf_vector = Concatenate()([x_ui, x_uj])

    ## flat matrix to an array
    # mlp_user_latent=Flatten()(MLP_embedding_user(user_input))
    # mlp_item_latent_pos=Flatten()(MLP_embedding_item(item_input_pos))
    # mlp_item_latent_neg=Flatten()(MLP_embedding_item(item_input_neg))
    # mlp_item_latent_neg=Lambda(lambda x:-x)(mlp_item_latent_neg)
    # mlp_vector=Concatenate()([u,i,j])
    # for idx in range(1,num_layer):
    #     #set up hidden layer of the network, why tanh and have to regularize?? L1 consider weight, l2 consider square of weight
    #     layer=Dense(layers[idx],kernel_regularizer=l2(0.0000),activation='tanh',name="layer%d" %idx)
    #     mlp_vector=layer(mlp_vector)
    # # for idx in range(1,num_layer):
    #     #set up hidden layer of the network, why tanh and have to regularize?? L1 consider weight, l2 consider square of weight
    # # layer=Dense(1,kernel_regularizer=l2(0.0000),activation='tanh')
    # # mf_vector=layer(mf_vector)
    #
    #
    # ## concatenation of NBPR layer and DNCR layer
    # # predict_vector=Concatenate()([mf_vector,mlp_vector])
    #
    #
    # #set up prediction --> final layer, sigmoid activate to give binary output
    # delta=Dense(1,kernel_regularizer=l2(0.0000),activation='tanh',name='final')(mlp_vector)
    # model=Model(inputs=[u,i,j],outputs=delta)
