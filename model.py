import torch.nn as nn
import torch
import math

class Simple_NeRF(nn.Module):
    def __init__(self, position=10, direction=4, dim=128):
        super(Simple_NeRF, self).__init__()

        b0 = nn.Sequential(nn.Linear(6 * position + 3, dim), nn.ReLU(), 
                                     nn.Linear(dim, dim), nn.ReLU(),
                                     nn.Linear(dim, dim), nn.ReLU(),
                                     nn.Linear(dim, dim), nn.ReLU())
        b1 = nn.Sequential(nn.Linear(6 * position + dim + 3, dim), nn.ReLU(), 
                                     nn.Linear(dim, dim), nn.ReLU(),
                                     nn.Linear(dim, dim), nn.ReLU(),
                                     nn.Linear(dim, dim + 1), nn.ReLU())
        b2 = nn.Sequential(nn.Linear(6 * direction + dim + 3, dim // 2), nn.ReLU())
        b3 = nn.Sequential(nn.Linear(dim // 2, 3), nn.Sigmoid())
        self.blocks = nn.ParameterList([b0, b1, b2, b3])

        self.position = position
        self.direction = direction
        self.ReLU = nn.ReLU()
    
    # Positional encoding of tensor with sin and cos
    # Args: x = 2D input tensor to be encoded (3 cols)
    # L = specifies length of encoding
    #
    # Returns: positional encoding of tensor
    @staticmethod
    def pos_encoding(x, L):
        tensors = [x]
        for l in range(L):
            tensors.append(torch.sin((2 ** l) * math.pi * x))
            tensors.append(torch.cos((2 ** l) * math.pi * x))
        final_encoding = torch.cat(tensors, 1)
        return final_encoding

    # Forward pass of Simple_NeRF model following Fig. 7 of paper
    # Args: x = positions of points (3 cols), 
    # d = directions of points (3 cols)
    #
    # Returns: [color, density]
    def forward(self, x, d):

        # positionally encode both x and d
        x_encoded = self.pos_encoding(x, self.position)
        d_encoded = self.pos_encoding(d, self.direction)


        b0_result = self.blocks[0](x_encoded)

        b1_input = torch.cat((b0_result, x_encoded), 1)
        b1_result = self.blocks[1](b1_input)

        b2_input = torch.cat((b1_result[:, :-1], d_encoded), 1)
        density = self.ReLU(b1_result[:, -1])
        b2_result = self.blocks[2](b2_input)

        color = self.blocks[3](b2_result)

        return [color, density]
        
        emb_x = self.positional_encoding(o, self.embedding_dim_pos)
        emb_d = self.positional_encoding(d, self.embedding_dim_direction)
        tmp = self.block2(torch.cat((self.block1(emb_x), emb_x), dim=1))
        h, sigma = tmp[:, :-1], self.relu(tmp[:, -1])
        c = self.block4(self.block3(torch.cat((h, emb_d), dim=1)))
        return c, sigma
