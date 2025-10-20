# kawasaki_dynamics.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from numba import njit

lattice_constant = 3.0  # nm
lattice_size = (60, 60, 30)
T = 1.0         # k_B T / J 
J1 = 1.0
J2 = 1/2

# Predefine neighbour offsets as arrays 
first_neighbour_offsets = np.asarray([
    ( 1, 0, 0), 
    (-1, 0, 0),
    ( 0, 1, 0), 
    ( 0,-1, 0),
    ( 0, 0, 1), 
    ( 0, 0,-1)], dtype=np.int32)

# second neighbour offsets: 12 face-diagonals at distance sqrt(2)
second_neighbour_offsets = np.array([
    (-1, -1,  0),
    (-1,  0, -1),
    (-1,  0,  1),
    (-1,  1,  0),
    ( 0, -1, -1),
    ( 0, -1,  1),
    ( 0,  1, -1),
    ( 0,  1,  1),
    ( 1, -1,  0),
    ( 1,  0, -1),
    ( 1,  0,  1),
    ( 1,  1,  0)], dtype=np.int32)


# -----------------------
# JIT kernels
# -----------------------

@njit
def _local_energy_nb(spins, x, y, z, nn1, nn2, J1, J2):
    """
    Local bond energy touching site (x,y,z).
    0.5 factor avoids double counting bonds.
    spins: int8 3D array with values in {-1, +1}
    """
    s = spins[x, y, z]
    nx, ny, nz = spins.shape
    e = 0.0

    # 1st neighbours
    for m in range(nn1.shape[0]):
        dx, dy, dz = nn1[m, 0], nn1[m, 1], nn1[m, 2]
        ii = (x + dx) % nx
        jj = (y + dy) % ny
        kk = (z + dz) % nz
        e += -0.5 * J1 * s * spins[ii, jj, kk]

    # 2nd neighbours
    for m in range(nn2.shape[0]):
        dx, dy, dz = nn2[m, 0], nn2[m, 1], nn2[m, 2]
        ii = (x + dx) % nx
        jj = (y + dy) % ny
        kk = (z + dz) % nz
        e += -0.5 * J2 * s * spins[ii, jj, kk]

    return e


@njit
def _kawasaki_attempt_nb(spins, nn1, nn2, J1, J2, kT):
    """
    One Kawasaki exchange attempt at a random site + random NN1 neighbour.
    Returns 1 if accepted, 0 if rejected/no-op.
    """
    nx, ny, nz = spins.shape

    x = np.random.randint(0, nx)
    y = np.random.randint(0, ny)
    z = np.random.randint(0, nz)

    m = np.random.randint(0, nn1.shape[0])
    dx, dy, dz = nn1[m, 0], nn1[m, 1], nn1[m, 2]
    i2 = (x + dx) % nx
    j2 = (y + dy) % ny
    k2 = (z + dz) % nz

    if spins[x, y, z] == spins[i2, j2, k2]:
        return 0  # no change if same spin

    # energy before
    Eb = _local_energy_nb(spins, x, y, z, nn1, nn2, J1, J2) \
       + _local_energy_nb(spins, i2, j2, k2, nn1, nn2, J1, J2)

    # swap
    s = spins[x, y, z]
    spins[x, y, z] = spins[i2, j2, k2]
    spins[i2, j2, k2] = s

    # energy after
    Ea = _local_energy_nb(spins, x, y, z, nn1, nn2, J1, J2) \
       + _local_energy_nb(spins, i2, j2, k2, nn1, nn2, J1, J2)

    dE = Ea - Eb
    # Heat-bath acceptance
    kT_eff = kT if kT > 1e-12 else 1e-12
    p = 1.0 / (1.0 + np.exp(dE / kT_eff))

    if np.random.random() >= p:
        # reject: swap back
        s = spins[x, y, z]
        spins[x, y, z] = spins[i2, j2, k2]
        spins[i2, j2, k2] = s
        return 0
    
    return 1


@njit
def _kawasaki_mcs_nb(spins, nn1, nn2, J1, J2, kT):
    """
    One Monte Carlo Step (MCS) = nsites attempts.
    Returns number of accepted exchanges.
    """
    nsites = spins.size
    acc = 0
    for _ in range(nsites):
        acc += _kawasaki_attempt_nb(spins, nn1, nn2, J1, J2, kT)
    return acc


@njit
def _interfacial_bonds_nb(spins):
    """
    Count unlike first neighbour bonds (periodic in x,y,z), each counted once using +x,+y,+z.
    """
    nx, ny, nz = spins.shape
    count = 0
    # +x
    for x in range(nx):
        xx = (x + 1) % nx
        for y in range(ny):
            for z in range(nz):
                if spins[x, y, z] != spins[xx, y, z]:
                    count += 1
    # +y
    for x in range(nx):
        for y in range(ny):
            yy = (y + 1) % ny
            for z in range(nz):
                if spins[x, y, z] != spins[x, yy, z]:
                    count += 1
    # +z
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                zz = (z + 1) % nz
                if spins[x, y, z] != spins[x, y, zz]:
                    count += 1
    return count



def ising_model(lattice, steps=1000, sample_points=100, kT=T):
    """
    Each 'step' is one MCS (nsites exchange attempts).
    `lattice` should be int8 with values {-1,+1} for best performance.
    """
    spins = lattice.astype(np.int8, copy=False)
    nx, ny, nz = spins.shape
    E_list = []
    sample_every = max(1, steps // max(1, sample_points))

    for mcs in range(int(steps)):
        acc = _kawasaki_mcs_nb(spins, first_neighbour_offsets, second_neighbour_offsets, J1, J2, kT)

        if (mcs % sample_every) == 0:
            E_list.append(calculate_total_energy(spins))

        if (mcs % max(1, steps // 10)) == 0:
            print(f"{mcs/steps*100:.1f}% complete (accepted {acc} moves)")

    return spins, E_list

def calculate_total_energy(spins, J1=J1, J2=J2):
    E = 0.0
    # first neighbours 
    for dx,dy,dz in first_neighbour_offsets:
        shifted = np.roll(np.roll(np.roll(spins, dx, axis=0), dy, axis=1), dz, axis=2)
        E += -J1 * np.sum(spins * shifted)
    # second neighbours
    for dx,dy,dz in second_neighbour_offsets:
        shifted = np.roll(np.roll(np.roll(spins, dx, axis=0), dy, axis=1), dz, axis=2)
        E += -J2 * np.sum(spins * shifted)
    return 0.5 * E  # bonds counted twice


def compute_interfacial_area(spins, lattice_constant=3.0):
    bonds = _interfacial_bonds_nb(spins)
    area_nm2 = bonds * (lattice_constant ** 2)
    return area_nm2, bonds


def plot_lattice_3d(lattice, downsample_factor=2):
    arr = lattice[::downsample_factor, ::downsample_factor, ::downsample_factor]
    filled = arr != 0
    cmap = colors.ListedColormap(['blue', 'red'])
    norm = colors.BoundaryNorm([-1.5, 0, 1.5], cmap.N)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(filled, facecolors=cmap(norm(arr)), edgecolor=None, alpha=0.5)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('3D Lattice')
    plt.show()

if __name__ == '__main__':
        
    lattice = np.random.choice([-1, 1], size=lattice_size)
    lattice, E = ising_model(lattice, steps=0)
    print("Energy samples:", len(E))
    area_nm2, bonds = compute_interfacial_area(lattice, lattice_constant)

    plt.figure()
    plt.plot(E, marker='o')
    plt.xlabel('Sample idx'); plt.ylabel('Energy')
    plt.title('Energy samples')
    plt.show()

    plot_lattice_3d(lattice, downsample_factor=2)
    print(f"Interfacial bonds: {bonds}, Area ≈ {area_nm2/10e6:.2f} 10^6 nm^2")


