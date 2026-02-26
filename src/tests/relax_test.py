import torch

from src.model.sandpile import SandpileModel


def relax(z, z_c, bound_cond):
    """Old relax function"""
    z_new = z.clone()
    d = len(z_new.size())

    # perform relaxation steps until z(r) <= z_c everywhere
    while True:
        # Identify all unstable sites simultaneously
        unstable_mask = z_new > z_c
        if not unstable_mask.any():
            break

        # Cast to z's dtype to calculate sand transfer
        firings = unstable_mask.to(z_new.dtype)

        # Bulk relaxation step (Eq 4)
        z_new -= 2 * d * firings

        # Process dimensional shifts (nearest-neighbor transfers)
        for dim in range(d):
            # Slices for receiving from r - e_i (moving right/up)
            idx_z_plus = [slice(None)] * d
            idx_z_plus[dim] = slice(1, None)
            idx_f_minus = [slice(None)] * d
            idx_f_minus[dim] = slice(None, -1)

            # Slices for receiving from r + e_i (moving left/down)
            idx_z_minus = [slice(None)] * d
            idx_z_minus[dim] = slice(None, -1)
            idx_f_plus = [slice(None)] * d
            idx_f_plus[dim] = slice(1, None)

            # Add sand tumbling from adjacent sites
            z_new[tuple(idx_z_plus)] += firings[tuple(idx_f_minus)]
            z_new[tuple(idx_z_minus)] += firings[tuple(idx_f_plus)]

            # Boundary Condition enforcement
            if bound_cond == "open":
                # Eq 5: Re-add 1 unit for each boundary at index N-1 where sand does not tumble outward
                idx_N = [slice(None)] * d
                idx_N[dim] = -1
                z_new[tuple(idx_N)] += firings[tuple(idx_N)]

        # Zero-out specific boundaries after all transfers are computed
        for dim in range(d):
            # Both Open (Eq 5) and Closed (Eq 6) set z=0 at r_j = 0
            idx_0 = [slice(None)] * d
            idx_0[dim] = 0
            z_new[tuple(idx_0)] = 0

            # Closed (Eq 6) also sets z=0 at r_j = N
            if bound_cond == "closed":
                idx_N = [slice(None)] * d
                idx_N[dim] = -1
                z_new[tuple(idx_N)] = 0

    return z_new


Ns = [10, 20, 30]
ds = [2, 3, 4]
boundaries = ["open", "closed"]
z_c = 5

for N in Ns:
    for d in ds:
        shape = (N,) * d
        z_init = torch.randint(0, 10, size=shape)
        for bound_cond in boundaries:
            model = SandpileModel(
                N, d, z_c, boundary_condition=bound_cond, z_init=z_init
            )
            model.relax()
            z_rel_1 = model.z
            z_rel_2 = relax(z_init, z_c, bound_cond)
            if torch.all(z_rel_1 == z_rel_2):
                print(f"Check passed for {N=}, {d=}, {bound_cond=}.")
            else:
                print(f"Check failed for {N=}, {d=}, {bound_cond=}.")
