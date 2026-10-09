import torch
import torch.nn as nn


class KernelPENodeEncoder(torch.nn.Module):
    """Configurable kernel-based Positional Encoding node encoder.

    The choice of which kernel-based statistics to use is configurable through
    setting of `kernel_type`. Based on this, the appropriate config is selected,
    and also the appropriate variable with precomputed kernel stats is then
    selected from PyG Data graphs in `forward` function.
    E.g., supported are 'RWSE', 'HKdiagSE', 'ElstaticSE'.

    PE of size `dim_pe` will get appended to each node feature vector.
    If `expand_x` set True, original node features will be first linearly
    projected to (dim_emb - dim_pe) size and the concatenated with PE.

    Args:
        dim_emb: Size of final node embedding
        expand_x: Expand node features `x` from dim_in to (dim_emb - dim_pe)
    """

    kernel_type = None  # Instantiated type of the KernelPE, e.g. RWSE

    def __init__(self, dim_in, dim_emb, pecfg, expand_x=True):
        super().__init__()
        if self.kernel_type is None:
            raise ValueError(
                f"{self.__class__.__name__} has to be "
                f"preconfigured by setting 'kernel_type' class"
                f"variable before calling the constructor.",
            )

        dim_pe = pecfg.pe_dim  # Size of the kernel-based PE embedding
        num_rw_steps = pecfg.kernel.times
        norm_type = pecfg.raw_norm_type.lower()  # Raw PE normalization layer type
        # self.pass_as_var = pecfg.pass_as_var  # Pass PE also as a separate variable

        if dim_emb - dim_pe < 1:
            raise ValueError(
                f"PE dim size {dim_pe} is too large for "
                f"desired embedding size of {dim_emb}.",
            )

        if expand_x:
            self.linear_x = nn.Linear(dim_in, dim_emb - dim_pe)
        self.expand_x = expand_x

        if norm_type == "batchnorm":
            self.raw_norm = nn.BatchNorm1d(num_rw_steps)
        else:
            self.raw_norm = None

        self.pe_encoder = nn.Linear(num_rw_steps, dim_pe)

    def forward(self, batch):
        pestat_var = f"pestat_{self.kernel_type}"
        if not hasattr(batch, pestat_var):
            raise ValueError(
                f"Precomputed '{pestat_var}' variable is "
                f"required for {self.__class__.__name__}; set "
                f"config 'posenc_{self.kernel_type}.enable' to "
                f"True, and also set 'posenc.kernel.times' values",
            )

        pos_enc = getattr(batch, pestat_var)  # (Num nodes) x (Num kernel times)
        # pos_enc = batch.rw_landing  # (Num nodes) x (Num kernel times)
        if self.raw_norm:
            pos_enc = self.raw_norm(pos_enc)
        pos_enc = self.pe_encoder(pos_enc)  # (Num nodes) x dim_pe

        # Expand node features if needed
        if self.expand_x:
            h = self.linear_x(batch.x)
        else:
            h = batch.x
        # Concatenate final PEs to input embedding
        batch.x = torch.cat((h, pos_enc), 1)
        # Keep PE also separate in a variable (e.g. for skip connections to input)
        # if self.pass_as_var:
        #     setattr(batch, f'pe_{self.kernel_type}', pos_enc)

        return batch


class RWSENodeEncoder(KernelPENodeEncoder):
    """Random Walk Structural Encoding node encoder."""

    kernel_type = "RWSE"
