"""
FEWER-TOKEN NEURAL SPEECH CODEC WITH TIME-INVARIANT CODES

For more details: https://arxiv.org/pdf/2310.00014

Authors
 * Salima Mdhaffar 2025
"""

# Adapted from https://github.com/jik876/hifi-gan/ and https://github.com/coqui-ai/TTS/
# MIT License

# Copyright (c) 2020 Jungil Kong

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio import transforms
import typing as tp
import speechbrain as sb
from speechbrain.nnet.CNN import Conv1d, Conv2d, ConvTranspose1d
from torch.nn.utils import spectral_norm
from torch.nn.utils import weight_norm

FeatureMapType = tp.List[torch.Tensor]
LogitsType = torch.Tensor
DiscriminatorOutput = tp.Tuple[tp.List[LogitsType], tp.List[FeatureMapType]]

LRELU_SLOPE = 0.1

CONV_NORMALIZATIONS = frozenset([
    'none', 'weight_norm', 'spectral_norm', 'time_layer_norm', 'layer_norm',
    'time_group_norm'
])

def get_norm_module(module: nn.Module,
                    causal: bool=False,
                    norm: str='none',
                    **norm_kwargs) -> nn.Module:
    """Return the proper normalization module. If causal is True, this will ensure the returned
    module is causal, or return an error if the normalization doesn't support causal evaluation.
    """
    assert norm in CONV_NORMALIZATIONS
    if norm == 'layer_norm':
        assert isinstance(module, nn.modules.conv._ConvNd)
        return ConvLayerNorm(module.out_channels, **norm_kwargs)
    elif norm == 'time_group_norm':
        if causal:
            raise ValueError("GroupNorm doesn't support causal evaluation.")
        assert isinstance(module, nn.modules.conv._ConvNd)
        return nn.GroupNorm(1, module.out_channels, **norm_kwargs)
    else:
        return nn.Identity()

def apply_parametrization_norm(module: nn.Module,
                               norm: str='none') -> nn.Module:
    assert norm in CONV_NORMALIZATIONS
    if norm == 'weight_norm':
        return weight_norm(module)
    elif norm == 'spectral_norm':
        return spectral_norm(module)
    else:
        # We already check was in CONV_NORMALIZATION, so any other choice
        # doesn't need reparametrization.
        return module

class NormConv2d(nn.Module):
    """Wrapper around Conv2d and normalization applied to this conv
    to provide a uniform interface across normalization approaches.
    """

    def __init__(self,
                 *args,
                 norm: str='none',
                 norm_kwargs: tp.Dict[str, tp.Any]={},
                 **kwargs):
        super().__init__()
        self.conv = apply_parametrization_norm(nn.Conv2d(*args, **kwargs), norm)
        self.norm = get_norm_module(
            self.conv, causal=False, norm=norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x

def get_2d_padding(kernel_size: tp.Tuple[int, int],
                   dilation: tp.Tuple[int, int]=(1, 1)):
    return (((kernel_size[0] - 1) * dilation[0]) // 2, (
        (kernel_size[1] - 1) * dilation[1]) // 2)

#####################################
############Encoder##################
#####################################

class GlobalTokenEncoder(nn.Module):
    """ #Apply further convolutions + LeakyReLU to h (Time-Invariant Extractor) """
    def __init__(self, in_channels=128, hidden_channels=64, out_channels=128, kernel_size=3, stride=1):
        super().__init__()
        self.pad = (kernel_size - stride) // 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size, stride, self.pad, bias=False),
            nn.LeakyReLU(LRELU_SLOPE),
            nn.Conv1d(hidden_channels, hidden_channels, kernel_size, stride, self.pad, bias=False),
            nn.LeakyReLU(LRELU_SLOPE),
            nn.Conv1d(hidden_channels, out_channels, kernel_size, stride, self.pad, bias=False),
            nn.LeakyReLU(LRELU_SLOPE),
        )
        self.fn = nn.Sequential(
            nn.Linear(out_channels, out_channels),
            nn.LeakyReLU(LRELU_SLOPE),
            nn.BatchNorm1d(out_channels),
        )
    def forward(self, x):
        """
        x --- [B, in_channels, T]
        out -- [B, out_channels]
        """
        x = self.conv(x)
        x = torch.mean(x, dim=2)
        x = self.fn(x)
        return x

class ResBlock1(torch.nn.Module):
    """
    Residual Block Type 1, which has 3 convolutional layers in each convolution block.

    Arguments
    ---------
    channels : int
        number of hidden channels for the convolutional layers.
    kernel_size : int
        size of the convolution filter in each layer.
    dilation : list
        list of dilation value for each conv layer in a block.
    """

    def __init__(self, channels, kernel_size=3, dilation=(1, 3, 5)):
        super().__init__()
        self.convs1 = nn.ModuleList(
            [
                Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation[0],
                    padding="same",
                    skip_transpose=True,
                    weight_norm=True,
                ),
                Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation[1],
                    padding="same",
                    skip_transpose=True,
                    weight_norm=True,
                ),
                Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation[2],
                    padding="same",
                    skip_transpose=True,
                    weight_norm=True,
                ),
            ]
        )

        self.convs2 = nn.ModuleList(
            [
                Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=1,
                    padding="same",
                    skip_transpose=True,
                    weight_norm=True,
                ),
                Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=1,
                    padding="same",
                    skip_transpose=True,
                    weight_norm=True,
                ),
                Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=1,
                    padding="same",
                    skip_transpose=True,
                    weight_norm=True,
                ),
            ]
        )

    def forward(self, x):
        """Returns the output of ResBlock1

        Arguments
        ---------
        x : torch.Tensor (batch, channel, time)
            input tensor.

        Returns
        -------
        The ResBlock outputs
        """

        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c1(xt)
            xt = F.leaky_relu(xt, LRELU_SLOPE)
            xt = c2(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        """This functions removes weight normalization during inference."""
        for layer in self.convs1:
            layer.remove_weight_norm()
        for layer in self.convs2:
            layer.remove_weight_norm()


class ResBlock2(torch.nn.Module):
    """
    Residual Block Type 2, which has 2 convolutional layers in each convolution block.

    Arguments
    ---------
    channels : int
        number of hidden channels for the convolutional layers.
    kernel_size : int
        size of the convolution filter in each layer.
    dilation : list
        list of dilation value for each conv layer in a block.
    """

    def __init__(self, channels, kernel_size=3, dilation=(1, 3)):
        super().__init__()
        self.convs = nn.ModuleList(
            [
                Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation[0],
                    padding="same",
                    skip_transpose=True,
                    weight_norm=True,
                ),
                Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation[1],
                    padding="same",
                    skip_transpose=True,
                    weight_norm=True,
                ),
            ]
        )

    def forward(self, x):
        """Returns the output of ResBlock1

        Arguments
        ---------
        x : torch.Tensor (batch, channel, time)
            input tensor.

        Returns
        -------
        The ResBlock outputs
        """

        for c in self.convs:
            xt = F.leaky_relu(x, LRELU_SLOPE)
            xt = c(xt)
            x = xt + x
        return x

    def remove_weight_norm(self):
        """This functions removes weight normalization during inference."""
        for layer in self.convs:
            layer.remove_weight_norm()

class HiFiCodecEncoder(nn.Module):
    """
    Encoder follows a similar structure as HifiCodec.
    Encoder is composed of a 1D convolutional layer followed by 4 convolutional modules and a final 1D convolutional layer.
    Each convolutional module consists of three residual units and one downsampling layer.
    All of these 4 modules indicate a total downsampling of 320 times.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        resblock_type,
        resblock_dilation_sizes,
        resblock_kernel_sizes,
        upsample_kernel_sizes,
        upsample_initial_channel,
        upsample_factors,
        inference_padding=5,
        cond_channels=0,
        conv_post_bias=True,
    ):
        super().__init__()
        self.inference_padding = inference_padding
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_factors)
        # initial convolutional layers
        self.conv_pre =nn.Conv1d(
            in_channels=1,
            out_channels=32,
            kernel_size=7,
            stride=1,
            padding=3,
        )
        resblock = ResBlock1 if resblock_type == "1" else ResBlock2
        # downsampling layers
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(
            list(
                (
                    list(zip(upsample_factors, upsample_kernel_sizes))))
        ):
            self.ups.append(
                nn.Conv1d(
                    in_channels=32 * (2**i),
                    out_channels=32 * (2 ** (i + 1)),
                    kernel_size=k,
                    stride=u,
                    padding=((k - u) // 2),
                )
            )
        # MRF blocks
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = 32 * (2 ** (i + 1))
            for _, (k, d) in enumerate(
                zip(list((resblock_kernel_sizes)), list((resblock_dilation_sizes)))
            ):
                self.resblocks.append(resblock(ch, k, d))
        # post convolution layer
        self.conv_post = nn.Conv1d(
            in_channels=512,
            out_channels=512,
            kernel_size=3,
            stride=1,
            padding=1,
            #skip_transpose=True,
            #bias=conv_post_bias,
            #weight_norm=True,
        )
        if cond_channels > 0:
            self.cond_layer = Conv1d(
                in_channels=cond_channels,
                out_channels=upsample_initial_channel,
                kernel_size=1,
            )
        self.global_encoder = GlobalTokenEncoder(in_channels=128, hidden_channels=64, out_channels=128, kernel_size=3, stride=1)

    def forward(self, x, g=None):
        """
        Arguments
        ---------
        x : torch.Tensor (batch, channel, time)
            feature input tensor.
        g : torch.Tensor (batch, 1, time)
            global conditioning input tensor.

        Returns
        -------
        The generator outputs
        """

        #First 1D conv
        print(x.shape)
        o = self.conv_pre(x)
        print(o.shape)
        global_features = None # Diff (Salima)
        '''if hasattr(self, "cond_layer"):
            o = o + self.cond_layer(g)'''
        #Applies a LeakyReLU activation and Upsamples the feature map (e.g., to increase time resolution in audio)
        for i in range(self.num_upsamples):
            print(i)
            o = F.leaky_relu(o, LRELU_SLOPE)
            o = self.ups[i](o)
            print(o.shape)
            # Runs multiple residual blocks (num_kernels times). 
            z_sum = None
            for j in range(self.num_kernels):
                if z_sum is None:
                    z_sum = self.resblocks[i * self.num_kernels + j](o)
                else:
                    z_sum += self.resblocks[i * self.num_kernels + j](o)
            o = z_sum / self.num_kernels
            print(o.shape)
            # Diff (Salima)
            if i == self.num_upsamples//2 - 1:
                mid_features = o
                global_features = self.global_encoder(mid_features) #Apply further convolutions + LeakyReLU to h
        o = F.leaky_relu(o)
        o = self.conv_post(o)
        o = torch.tanh(o)
        return o, global_features

    def remove_weight_norm(self):
        """This functions removes weight normalization during inference."""

        for layer in self.ups:
            layer.remove_weight_norm()
        for layer in self.resblocks:
            layer.remove_weight_norm()
        self.conv_pre.remove_weight_norm()
        self.conv_post.remove_weight_norm()

########################################
#############Quantizer##################
########################################
class Quantizer_module(torch.nn.Module):
    def __init__(self, n_e, e_dim):
        super(Quantizer_module, self).__init__()
        self.embedding = nn.Embedding(n_e, e_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)

    def forward(self, x):
        # compute Euclidean distance
        d = torch.sum(x ** 2, 1, keepdim=True) + torch.sum(self.embedding.weight ** 2, 1) \
            - 2 * torch.matmul(x, self.embedding.weight.T)
        min_indicies = torch.argmin(d, 1)
        z_q = self.embedding(min_indicies)
        return z_q, min_indicies

class Quantizer(torch.nn.Module):
    """
    Takes as input the output of the Encoder
    n_codes: Number of codes in each codebook 
    residul_layer: Number of residual quantization layers (1–4)
    
    global_code_num: Number of sub-groups to split the global representation vector.
    Example: if the vector is 128-dimension and the number of sub-groups is 8, we will
    get 8 chunks with 16 dimension.
    """
    def __init__(self, n_codes, residul_layer, global_code_num, codebook_loss_lambda, commitment_loss_lambda):
        super(Quantizer, self).__init__()

        
        self.residul_layer = residul_layer
        self.global_code_num = global_code_num
        # Quantizer for the first layer (works on full 512-dim vector)
        self.quantizer_modules = Quantizer_module(n_codes, 512)
        # Additional layers for residual quantization
        if residul_layer >= 2:
            self.quantizer_modules2 = Quantizer_module(n_codes, 512)
        if residul_layer >= 3:
            self.quantizer_modules3 = Quantizer_module(n_codes, 512)
        if residul_layer == 4:
            self.quantizer_modules4 = Quantizer_module(n_codes, 512)

        # Quantizer for the global representation
        self.quantizer_modules_globaltokens = nn.ModuleList([
            Quantizer_module(n_codes, 128//global_code_num)
            for _ in range(global_code_num)
        ])

        self.vq_loss_fn = VQVectorLoss(codebook_weight=codebook_loss_lambda,commitment_weight=commitment_loss_lambda)


    def forward_frame(self, inp, idx):
        """
        inp: Latent vector (or residual error) of shape [B, C, T]
        """
        # Transposed to [B, T, C] so quantization operates over time-steps 
        inp = inp.transpose(1, 2)
        #Reshapes to [B*T, 512] to quantize all time steps in a flat batch.
        x = inp.reshape(-1, 512)

        min_indices = []
        z_q = []

        # Select quantizer layer based on idx
        if idx == 0:
            quantizer = self.quantizer_modules
        elif idx == 1:
            quantizer = self.quantizer_modules2
        elif idx == 2:
            quantizer = self.quantizer_modules3
        elif idx == 3:
            quantizer = self.quantizer_modules4
        else:
            raise ValueError(f"Invalid idx {idx}")
        
        # Apply quantizer directly to full vector
        z_q, min_indices = quantizer(x)  # [B*T, 512], [B*T]
        
        # Reshape quantized vector back to [B, T, 512]
        z_q = z_q.reshape(inp.shape)

        # VQ loss: codebook + commitment
        loss = self.vq_loss_fn(z_q, inp)

        z_q = inp + (z_q - inp).detach()   # STE

        return z_q, loss, [min_indices]
    
    def forward_global_representation(self, inp):
        """
        inp: is the global_representation extracted from the encoder after applying conv + LeakyRelu
        inp is assumed to be [B, 128] or [B, 1, 128]

        Return:
        z_q: quantized version
        loss: VQ loss for this vector
        min_indices: indices used (can be used for conditioning / control)
        """

        x = inp.reshape(-1, 128)  # [B * 1, 128]
        # Splits the 128-dim vector into self.global_code_num equal chunks
        x = torch.split(x, 128 // self.global_code_num, dim=-1) 
        # x is a list of chunks
        min_indices = []
        z_q = []

        #Each chunk goes through a separate Quantizer_module
        #Loops over each chunk of the vector (_x) and its corresponding quantizer module (m)
        for _x, m in zip(x, self.quantizer_modules_globaltokens):
            _z_q, _min_indices = m(_x)
            # _z_q is the quantized output for that chunk
            # _min_indicies: the selected code index
            
            #Stores all quantized chunks and indices
            z_q.append(_z_q)
            min_indices.append(_min_indices)
            
        #Concatenates the quantized chunks back into a [B, 128] vector
        #Reshapes it to the same shape as inp
            
        z_q = torch.cat(z_q, -1).reshape(inp.shape)

        # Compute modular VQ loss
        loss = self.vq_loss_fn(z_q, inp)

        # Straight-through estimator (STE)
        z_q = inp + (z_q - inp).detach()

        return z_q, loss, min_indices

    def forward(self, inp, global_style):
        """
        inp: the input latent representation with shape [B, C, T] (Batch × Channels × Time)
        global_style: the global embedding (e.g. style or speaker vector), shape [B, 128]
        
        Return:
        quantized_out: final quantized latent representation [B, C, T]
        loss: total VQ loss
        all_indices: list of selected codebook indices per residual step
        global_style_quantized: quantized version of style embedding [B, 128]
        global_style_tokens: index tokens selected during quantization
        """

        quantized_out = 0.0 #Accumulates the output from all quantization steps
        residual = inp #Keeps track of the residual error after each quantization step
        all_losses = [] #Stores the loss from each residual layer
        all_indices = [] #Stores the codebook indices used at each step

        #Residual Vector Quantization (RVQ)
        for i in range(self.residul_layer):
            # current residual vector [B, C, T] or latent representation if residul_layer = 0
            quantized, loss, indices = self.forward_frame(residual, i)
            #quantized: the quantized approximation of this residual , loss: the VQ loss (codebook + commitment), indices: selected codebook entries
            #Remove quantized part from residual (standard residual quantization)
            quantized = quantized.transpose(1, 2)
            residual = residual - quantized
            #Accumulate quantized outputs into quantized_out
            quantized_out = quantized_out + quantized

            #Store all the indices (e.g., to reconstruct or decode later)
            all_indices.extend(indices)
            #Store loss for each layer
            all_losses.append(loss)

        all_losses = torch.stack(all_losses)
        loss = torch.mean(all_losses)

        #Time invariant Quantization (GVQ)
        global_style_quantized, loss_gst_vq, global_style_tokens = self.forward_global_representation(global_style)
        loss += loss_gst_vq
        return quantized_out, loss, all_indices, global_style_quantized, global_style_tokens


class CodebookLoss(nn.Module):
    def __init__(self, weight=1.0):
        super(CodebookLoss, self).__init__()
        self.weight = weight

    def forward(self, z_q, x):
        """
        Encourages codebook vectors to move closer to encoder output.
        z_q: quantized vectors
        x: encoder output (detached)
        """
        loss = torch.mean((z_q - x.detach()) ** 2)
        return self.weight * loss

class CommitmentLoss(nn.Module):
    def __init__(self, weight=0.25):
        super(CommitmentLoss, self).__init__()
        self.weight = weight

    def forward(self, z_q, x):
        """
        Encourages encoder output to commit to codebook vectors.
        z_q: quantized vectors (detached)
        x: encoder output
        """
        loss = torch.mean((z_q.detach() - x) ** 2)
        return self.weight * loss


class VQVectorLoss(nn.Module):
    def __init__(self, codebook_weight=1.0, commitment_weight=0.25):
        super(VQVectorLoss, self).__init__()
        self.codebook_loss = CodebookLoss(weight=codebook_weight)
        self.commitment_loss = CommitmentLoss(weight=commitment_weight)

    def forward(self, z_q, x):
        """
        Compute total vector quantization loss (codebook + commitment).
        Args:
            z_q: quantized vector (output of codebook lookup)
            x: encoder output (pre-quantization)
        Returns:
            total_loss: scalar VQ loss
        """
        cb_loss = self.codebook_loss(z_q, x)
        commit_loss = self.commitment_loss(z_q, x)
        return cb_loss + commit_loss


########################################
#############Decoder####################
########################################
class HiFiCodecDecoder(nn.Module):
    """
    Decoder adopts a symmetric structure to the encoder, utilizing transpose convolutions for upsampling.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        resblock_type,
        resblock_dilation_sizes,
        resblock_kernel_sizes,
        upsample_kernel_sizes,
        upsample_initial_channel,
        upsample_factors,
        inference_padding=5,
        cond_channels=0,
        conv_post_bias=True,
    ):
        super().__init__()
        self.inference_padding = inference_padding
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_factors)
        # initial convolutional layers
        self.conv_pre = Conv1d(
            in_channels=512,
            out_channels=512,
            kernel_size=7,
            stride=1,
            padding="same",
            skip_transpose=True,
            weight_norm=True,
        )
        resblock = ResBlock1 if resblock_type == "1" else ResBlock2
        # downsampling layers
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(
            zip(upsample_factors, upsample_kernel_sizes)
        ):
            self.ups.append(
                ConvTranspose1d(
                    in_channels=upsample_initial_channel // (2**i),
                    out_channels=upsample_initial_channel // (2 ** (i + 1)),
                    kernel_size=k,
                    stride=u,
                    padding=(k - u) // 2,
                    skip_transpose=True,
                    weight_norm=True,
                )
            )
        # MRF blocks
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for _, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(resblock(ch, k, d))
        # post convolution layer
        self.conv_post = Conv1d(
            in_channels=ch,
            out_channels=1,
            kernel_size=7,
            stride=1,
            padding="same",
            skip_transpose=True,
            bias=conv_post_bias,
            weight_norm=True,
        )
        if cond_channels > 0:
            self.cond_layer = Conv1d(
                in_channels=cond_channels,
                out_channels=upsample_initial_channel,
                kernel_size=1,
            )

    def forward(self, x, global_features):
        """
        Arguments
        ---------
        x : torch.Tensor (batch, channel, time)
            feature input tensor.
        g : torch.Tensor (batch, 1, time)
            global conditioning input tensor.

        Returns
        -------
        The generator outputs
        """

        #First 1D conv
        o = self.conv_pre(x)
        print(o.shape)
        #Applies a LeakyReLU activation and Upsamples the feature map (e.g., to increase time resolution in audio)
        for i in range(self.num_upsamples):
            o = F.leaky_relu(o, LRELU_SLOPE)
            o = self.ups[i](o)
            # Runs multiple residual blocks (num_kernels times). 
            z_sum = None
            for j in range(self.num_kernels):
                if z_sum is None:
                    z_sum = self.resblocks[i * self.num_kernels + j](o)
                else:
                    z_sum += self.resblocks[i * self.num_kernels + j](o)
            o = z_sum / self.num_kernels
            # Diff (Salima)
            if o.shape[-2] == global_features.shape[-1]:
                o += global_features.unsqueeze(-1).repeat(1, 1, o.shape[-1])
            print(o.shape)
        o = F.leaky_relu(o)
        o = self.conv_post(o)
        o = torch.tanh(o)
        print(o.shape)
        return o

    def remove_weight_norm(self):
        """This functions removes weight normalization during inference."""

        for layer in self.ups:
            layer.remove_weight_norm()
        for layer in self.resblocks:
            layer.remove_weight_norm()
        self.conv_pre.remove_weight_norm()
        self.conv_post.remove_weight_norm()

    @torch.no_grad()
    def inference(self, c, padding=True):
        """The inference function performs a padding and runs the forward method.

        Arguments
        ---------
        c : torch.Tensor (batch, channel, time)
            feature input tensor.
        padding : bool
            Whether to pad tensor before forward.

        Returns
        -------
        The generator outputs
        """
        if padding:
            c = torch.nn.functional.pad(
                c, (self.inference_padding, self.inference_padding), "replicate"
            )
        return self.forward(c)

###########################################
#######Discriminators#####################
##########################################

class DiscriminatorP(torch.nn.Module):
    """HiFiGAN Periodic Discriminator
    Takes every Pth value from the input waveform and applies a stack of convolutions.
    Note:
        if period is 2
        waveform = [1, 2, 3, 4, 5, 6 ...] --> [1, 3, 5 ... ] --> convs -> score, feat

    Arguments
    ---------
    period : int
       Take every a new value every `period`
    kernel_size : int
        Size of 1-d kernel for conv stack
    stride : int
        Stride of conv stack
    """

    def __init__(self, period, kernel_size=5, stride=3):
        super().__init__()
        self.period = period

        self.convs = nn.ModuleList(
            [
                Conv2d(
                    in_channels=1,
                    out_channels=32,
                    kernel_size=(kernel_size, 1),
                    stride=(stride, 1),
                    padding="same",
                    skip_transpose=True,
                    weight_norm=True,
                ),
                Conv2d(
                    in_channels=32,
                    out_channels=128,
                    kernel_size=(kernel_size, 1),
                    stride=(stride, 1),
                    padding="same",
                    skip_transpose=True,
                    weight_norm=True,
                ),
                Conv2d(
                    in_channels=128,
                    out_channels=512,
                    kernel_size=(kernel_size, 1),
                    stride=(stride, 1),
                    padding="same",
                    skip_transpose=True,
                    weight_norm=True,
                ),
                Conv2d(
                    in_channels=512,
                    out_channels=1024,
                    kernel_size=(kernel_size, 1),
                    stride=(stride, 1),
                    padding="same",
                    skip_transpose=True,
                    weight_norm=True,
                ),
                Conv2d(
                    in_channels=1024,
                    out_channels=1024,
                    kernel_size=(kernel_size, 1),
                    stride=1,
                    padding="same",
                    skip_transpose=True,
                    weight_norm=True,
                ),
            ]
        )
        self.conv_post = Conv2d(
            in_channels=1024,
            out_channels=1,
            kernel_size=(3, 1),
            stride=1,
            padding="same",
            skip_transpose=True,
            weight_norm=True,
        )

    def forward(self, x):
        """
        Arguments
        ---------
        x : torch.Tensor (batch, 1, time)
            input waveform.

        Returns
        -------
        Scores and features
        """

        feat = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            feat.append(x)
        x = self.conv_post(x)
        feat.append(x)
        x = torch.flatten(x, 1, -1)

        return x, feat


class MultiPeriodDiscriminator(torch.nn.Module):
    """HiFiGAN Multi-Period Discriminator (MPD)
    Wrapper for the `PeriodDiscriminator` to apply it in different periods.
    Periods are suggested to be prime numbers to reduce the overlap between each discriminator.
    """

    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorP(2),
                DiscriminatorP(3),
                DiscriminatorP(5),
                DiscriminatorP(7),
                DiscriminatorP(11),
            ]
        )

    def forward(self, x):
        """Returns Multi-Period Discriminator scores and features

        Arguments
        ---------
        x : torch.Tensor (batch, 1, time)
            input waveform.

        Returns
        -------
        Scores and features
        """

        scores = []
        feats = []
        for _, d in enumerate(self.discriminators):
            score, feat = d(x)
            scores.append(score)
            feats.append(feat)
        return scores, feats


class DiscriminatorS(torch.nn.Module):
    """HiFiGAN Scale Discriminator.
    It is similar to `MelganDiscriminator` but with a specific architecture explained in the paper.
    SpeechBrain CNN wrappers are not used here because spectral_norm is not often used

    Arguments
    ---------
    use_spectral_norm : bool
        if `True` switch to spectral norm instead of weight norm.
    """

    def __init__(self, use_spectral_norm=False):
        super().__init__()
        norm_f = (
            nn.utils.spectral_norm
            if use_spectral_norm
            else nn.utils.weight_norm
        )
        self.convs = nn.ModuleList(
            [
                norm_f(nn.Conv1d(1, 128, 15, 1, padding=7)),
                norm_f(nn.Conv1d(128, 128, 41, 2, groups=4, padding=20)),
                norm_f(nn.Conv1d(128, 256, 41, 2, groups=16, padding=20)),
                norm_f(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
                norm_f(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
                norm_f(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
                norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        """
        Arguments
        ---------
        x : torch.Tensor (batch, 1, time)
            input waveform.

        Returns
        -------
        Scores and features
        """

        feat = []
        for layer in self.convs:
            x = layer(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            feat.append(x)
        x = self.conv_post(x)
        feat.append(x)
        x = torch.flatten(x, 1, -1)
        return x, feat


class MultiScaleDiscriminator(torch.nn.Module):
    """HiFiGAN Multi-Scale Discriminator.
    Similar to MultiScaleMelganDiscriminator but specially tailored for HiFiGAN as in the paper.
    """

    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList(
            [
                DiscriminatorS(use_spectral_norm=True),
                DiscriminatorS(),
                DiscriminatorS(),
            ]
        )
        self.meanpools = nn.ModuleList(
            [nn.AvgPool1d(4, 2, padding=2), nn.AvgPool1d(4, 2, padding=2)]
        )

    def forward(self, x):
        """
        Arguments
        ---------
        x : torch.Tensor (batch, 1, time)
            input waveform.

        Returns
        -------
        Scores and features
        """

        scores = []
        feats = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                x = self.meanpools[i - 1](x)
            score, feat = d(x)
            scores.append(score)
            feats.append(feat)
        return scores, feats


class HifiganDiscriminator(nn.Module):
    """HiFiGAN discriminator wrapping MPD and MSD.

    Example
    -------
    >>> inp_tensor = torch.rand([4, 1, 8192])
    >>> hifigan_discriminator= HifiganDiscriminator()
    >>> scores, feats = hifigan_discriminator(inp_tensor)
    >>> len(scores)
    8
    >>> len(feats)
    8

    """

    def __init__(self):
        super().__init__()
        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()

    def forward(self, x):
        """Returns list of list of features from each layer of each discriminator.

        Arguments
        ---------
        x : torch.Tensor
            input waveform.

        Returns
        -------
        Features from each discriminator layer
        """

        scores, feats = self.mpd(x)
        scores_, feats_ = self.msd(x)
        return scores + scores_, feats + feats_


class DiscriminatorSTFT(nn.Module):
    """STFT sub-discriminator.
    Args:
        filters (int): Number of filters in convolutions
        in_channels (int): Number of input channels. Default: 1
        out_channels (int): Number of output channels. Default: 1
        n_fft (int): Size of FFT for each scale. Default: 1024
        hop_length (int): Length of hop between STFT windows for each scale. Default: 256
        kernel_size (tuple of int): Inner Conv2d kernel sizes. Default: ``(3, 9)``
        stride (tuple of int): Inner Conv2d strides. Default: ``(1, 2)``
        dilations (list of int): Inner Conv2d dilation on the time dimension. Default: ``[1, 2, 4]``
        win_length (int): Window size for each scale. Default: 1024
        normalized (bool): Whether to normalize by magnitude after stft. Default: True
        norm (str): Normalization method. Default: `'weight_norm'`
        activation (str): Activation function. Default: `'LeakyReLU'`
        activation_params (dict): Parameters to provide to the activation function.
        growth (int): Growth factor for the filters. Default: 1
    """

    def __init__(self,
                 filters: int,
                 in_channels: int=1,
                 out_channels: int=1,
                 n_fft: int=1024,
                 hop_length: int=256,
                 win_length: int=1024,
                 max_filters: int=1024,
                 filters_scale: int=1,
                 kernel_size: tp.Tuple[int, int]=(3, 9),
                 dilations: tp.List=[1, 2, 4],
                 stride: tp.Tuple[int, int]=(1, 2),
                 normalized: bool=True,
                 norm: str='weight_norm',
                 activation: str='LeakyReLU',
                 activation_params: dict={'negative_slope': 0.2}):
        super().__init__()
        assert len(kernel_size) == 2
        assert len(stride) == 2
        self.filters = filters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.normalized = normalized
        self.activation = getattr(torch.nn, activation)(**activation_params)
        self.spec_transform = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window_fn=torch.hann_window,
            normalized=self.normalized,
            center=False,
            pad_mode=None,
            power=None)
        spec_channels = 2 * self.in_channels
        self.convs = nn.ModuleList()
        self.convs.append(
            NormConv2d(
                spec_channels,
                self.filters,
                kernel_size=kernel_size,
                padding=get_2d_padding(kernel_size)))
        in_chs = min(filters_scale * self.filters, max_filters)
        for i, dilation in enumerate(dilations):
            out_chs = min((filters_scale**(i + 1)) * self.filters, max_filters)
            self.convs.append(
                NormConv2d(
                    in_chs,
                    out_chs,
                    kernel_size=kernel_size,
                    stride=stride,
                    dilation=(dilation, 1),
                    padding=get_2d_padding(kernel_size, (dilation, 1)),
                    norm=norm))
            in_chs = out_chs
        out_chs = min((filters_scale**(len(dilations) + 1)) * self.filters,
                      max_filters)
        self.convs.append(
            NormConv2d(
                in_chs,
                out_chs,
                kernel_size=(kernel_size[0], kernel_size[0]),
                padding=get_2d_padding((kernel_size[0], kernel_size[0])),
                norm=norm))
        self.conv_post = NormConv2d(
            out_chs,
            self.out_channels,
            kernel_size=(kernel_size[0], kernel_size[0]),
            padding=get_2d_padding((kernel_size[0], kernel_size[0])),
            norm=norm)

    def forward(self, x: torch.Tensor):
        fmap = []
        # print('x ', x.shape)
        z = self.spec_transform(x)  # [B, 2, Freq, Frames, 2]
        # print('z ', z.shape)
        z = torch.cat([z.real, z.imag], dim=1)
        # print('cat_z ', z.shape)
        z = rearrange(z, 'b c w t -> b c t w')
        for i, layer in enumerate(self.convs):
            z = layer(z)
            z = self.activation(z)
            # print('z i', i, z.shape)
            fmap.append(z)
        z = self.conv_post(z)
        # print('logit ', z.shape)
        return z, fmap

class MultiScaleSTFTDiscriminator(nn.Module):
    """Multi-Scale STFT (MS-STFT) discriminator.
    Args:
        filters (int): Number of filters in convolutions
        in_channels (int): Number of input channels. Default: 1
        out_channels (int): Number of output channels. Default: 1
        n_ffts (Sequence[int]): Size of FFT for each scale
        hop_lengths (Sequence[int]): Length of hop between STFT windows for each scale
        win_lengths (Sequence[int]): Window size for each scale
        **kwargs: additional args for STFTDiscriminator
    """

    def __init__(self,
                 filters: int,
                 in_channels: int=1,
                 out_channels: int=1,
                 n_ffts: tp.List[int]=[1024, 2048, 512, 256, 128],
                 hop_lengths: tp.List[int]=[256, 512, 128, 64, 32],
                 win_lengths: tp.List[int]=[1024, 2048, 512, 256, 128],
                 **kwargs):
        super().__init__()
        assert len(n_ffts) == len(hop_lengths) == len(win_lengths)
        self.discriminators = nn.ModuleList([
            DiscriminatorSTFT(
                filters,
                in_channels=in_channels,
                out_channels=out_channels,
                n_fft=n_ffts[i],
                win_length=win_lengths[i],
                hop_length=hop_lengths[i],
                **kwargs) for i in range(len(n_ffts))
        ])
        self.num_discriminators = len(self.discriminators)

    def forward(self, x: torch.Tensor) -> DiscriminatorOutput:
        logits = []
        fmaps = []
        for disc in self.discriminators:
            logit, fmap = disc(x)
            logits.append(logit)
            fmaps.append(fmap)
        return logits, fmaps

#################################
# GENERATOR LOSSES
#################################

def stft(x, n_fft, hop_length, win_length, window_fn="hann_window"):
    """computes the Fourier transform of short overlapping windows of the input"""
    o = torch.stft(
        x.squeeze(1),
        n_fft,
        hop_length,
        win_length,
    )
    M = o[:, :, :, 0]
    P = o[:, :, :, 1]
    S = torch.sqrt(torch.clamp(M**2 + P**2, min=1e-8))
    return S


class STFTLoss(nn.Module):
    """STFT loss. Input generate and real waveforms are converted
    to spectrograms compared with L1 and Spectral convergence losses.
    It is from ParallelWaveGAN paper https://arxiv.org/pdf/1910.11480.pdf

    Arguments
    ---------
    n_fft : int
        size of Fourier transform.
    hop_length : int
        the distance between neighboring sliding window frames.
    win_length : int
        the size of window frame and STFT filter.
    """

    def __init__(self, n_fft, hop_length, win_length):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length

    def forward(self, y_hat, y):
        """Returns magnitude loss and spectral convergence loss

        Arguments
        ---------
        y_hat : torch.tensor
            generated waveform tensor
        y : torch.tensor
            real waveform tensor

        Returns
        -------
        Magnitude loss and spectral convergence loss
        """

        y_hat_M = stft(y_hat, self.n_fft, self.hop_length, self.win_length)
        y_M = stft(y, self.n_fft, self.hop_length, self.win_length)
        # magnitude loss
        loss_mag = F.l1_loss(torch.log(y_M), torch.log(y_hat_M))
        # spectral convergence loss
        loss_sc = torch.norm(y_M - y_hat_M, p="fro") / torch.norm(y_M, p="fro")
        return loss_mag, loss_sc


class MultiScaleSTFTLoss(torch.nn.Module):
    """Multi-scale STFT loss. Input generate and real waveforms are converted
    to spectrograms compared with L1 and Spectral convergence losses.
    It is from ParallelWaveGAN paper https://arxiv.org/pdf/1910.11480.pdf"""

    def __init__(
        self,
        n_ffts=(1024, 2048, 512),
        hop_lengths=(120, 240, 50),
        win_lengths=(600, 1200, 240),
    ):
        super().__init__()
        self.loss_funcs = torch.nn.ModuleList()
        for n_fft, hop_length, win_length in zip(
            n_ffts, hop_lengths, win_lengths
        ):
            self.loss_funcs.append(STFTLoss(n_fft, hop_length, win_length))

    def forward(self, y_hat, y):
        """Returns multi-scale magnitude loss and spectral convergence loss

        Arguments
        ---------
        y_hat : torch.tensor
            generated waveform tensor
        y : torch.tensor
            real waveform tensor

        Returns
        -------
        Magnitude loss and spectral convergence loss
        """

        N = len(self.loss_funcs)
        loss_sc = 0
        loss_mag = 0
        for f in self.loss_funcs:
            lm, lsc = f(y_hat, y)
            loss_mag += lm
            loss_sc += lsc
        loss_sc /= N
        loss_mag /= N
        return loss_mag, loss_sc


class L1SpecLoss(nn.Module):
    """L1 Loss over Spectrograms as described in HiFiGAN paper https://arxiv.org/pdf/2010.05646.pdf
    Note : L1 loss helps leaning details compared with L2 loss

    Arguments
    ---------
    sample_rate : int
        Sample rate of audio signal.
    hop_length : int
        Length of hop between STFT windows.
    win_length : int
        Window size.
    n_mel_channels : int
        Number of mel filterbanks.
    n_fft : int
        Size of FFT.
    n_stft : int
        Size of STFT.
    mel_fmin : float
        Minimum frequency.
    mel_fmax : float
        Maximum frequency.
    mel_normalized : bool
        Whether to normalize by magnitude after stft.
    power : float
        Exponent for the magnitude spectrogram.
    norm : str or None
        If "slaney", divide the triangular mel weights by the width of the mel band
    mel_scale : str
        Scale to use: "htk" or "slaney".
    dynamic_range_compression : bool
        whether to do dynamic range compression
    """

    def __init__(
        self,
        sample_rate=22050,
        hop_length=256,
        win_length=24,
        n_mel_channels=80,
        n_fft=1024,
        n_stft=1024 // 2 + 1,
        mel_fmin=0.0,
        mel_fmax=8000.0,
        mel_normalized=False,
        power=1.0,
        norm="slaney",
        mel_scale="slaney",
        dynamic_range_compression=True,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mel_channels = n_mel_channels
        self.n_fft = n_fft
        self.n_stft = n_fft // 2 + 1
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.mel_normalized = mel_normalized
        self.power = power
        self.norm = norm
        self.mel_scale = mel_scale
        self.dynamic_range_compression = dynamic_range_compression

    def forward(self, y_hat, y):
        """Returns L1 Loss over Spectrograms

        Arguments
        ---------
        y_hat : torch.tensor
            generated waveform tensor
        y : torch.tensor
            real waveform tensor

        Returns
        -------
        L1 loss
        """
        y_hat_M = mel_spectogram(
            self.sample_rate,
            self.hop_length,
            self.win_length,
            self.n_fft,
            self.n_mel_channels,
            self.mel_fmin,
            self.mel_fmax,
            self.power,
            self.mel_normalized,
            self.norm,
            self.mel_scale,
            self.dynamic_range_compression,
            y_hat,
        )
        # y_M = mel_spectogram(self.mel_params, y)
        y_M = mel_spectogram(
            self.sample_rate,
            self.hop_length,
            self.win_length,
            self.n_fft,
            self.n_mel_channels,
            self.mel_fmin,
            self.mel_fmax,
            self.power,
            self.mel_normalized,
            self.norm,
            self.mel_scale,
            self.dynamic_range_compression,
            y,
        )

        # magnitude loss
        # loss_mag = F.l1_loss(torch.log(y_M), torch.log(y_hat_M))
        loss_mag = F.l1_loss(y_M, y_hat_M)
        return loss_mag

def feature_loss(fmap_r, fmap_g):
    """
    Feature-matching loss, minimizing distance between discriminator feature activations for real vs. fake audio.
    fmap_r: List of feature maps from real audio.
    fmap_g: List of feature maps from generated audio.
    Each element in the list is a set of intermediate layer outputs from a discriminator (e.g., from DiscriminatorP or DiscriminatorS), for each of the multiple discriminators used (MPD, MSD, etc.).
    fmap_r = [d1_features_real, d2_features_real, d3_features_real, ...]
    fmap_g = [d1_features_fake, d2_features_fake, d3_features_fake, ...]
    """
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g): # Loop over discriminators
        for rl, gl in zip(dr, dg): # Loop over layers in each discriminator
            loss += torch.mean(torch.abs(rl - gl))

    return loss * 2

def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    """
    loss against discriminators (multi-scale STFT + multi-period + multi-scale), encouraging realism in the reconstructed audio.
    disc_real_outputs: list of outputs (logits) from discriminators for real waveforms.
    disc_generated_outputs: list of outputs for fake/generated waveforms.
    These are usually lists because you may use multiple discriminators (multi-period, multi-scale, STFT, etc.).
    """
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1 - dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    print(loss)
    return loss, r_losses, g_losses

def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1 - dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses
