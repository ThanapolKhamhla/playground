import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

# -----------------------
# Lattice & parameters
# -----------------------
lattice_constant = 3.0  # nm
lattice_size = (60, 60, 30)

T = 1.0                 # k_B T / J units; set k_B=J=1
J1 = 1.0
J2 = 1/np.sqrt(2) 

# -----------------------
# Neighbour offsets
# -----------------------
# First neighbours: 6 with |r|=1
first_neighbour_offsets = [(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)]

# Second neighbours: 12 with |r|=sqrt(2)
second_neighbour_offsets = []
for dx in (-1,0,1):
    for dy in (-1,0,1):
        for dz in (-1,0,1):
            if dx*dx + dy*dy + dz*dz == 2:
                second_neighbour_offsets.append((dx,dy,dz))
                
                
def ising_model(lattice, steps=1, sample_points=100):
    E_list = []
    energy_sample_interval = max(1, int(steps) // max(1, sample_points))

    nx, ny, nz = lattice.shape
    nsites = nx * ny * nz
    
    for i in range(int(steps)):
        for _ in range(nsites):
            centre = (np.random.randint(0, nx),
                        np.random.randint(0, ny),
                        np.random.randint(0, nz))
            lattice = kawasaki_exchange(lattice, centre)

        if (i % energy_sample_interval) == 0:
            E_list.append(calculate_total_energy(lattice))

        if (i % max(1, (int(steps) // 10))) == 0:
            print(f"{i / int(steps) * 100:.1f}% complete")

    return lattice, E_list

def kawasaki_exchange(lattice, centre, kT=T):
    x, y, z = centre
    nx, ny, nz = lattice.shape

    # pick random NN1 neighbour
    dx, dy, dz =first_neighbour_offsets[np.random.randint(len(first_neighbour_offsets))]
    i2 = (x + dx) % nx
    j2 = (y + dy) % ny
    k2 = (z + dz) % nz

    if lattice[x,y,z] == lattice[i2,j2,k2]:
        return lattice  # no-op if same spin

    # local energy before
    Eb = local_energy(lattice, (x,y,z)) + local_energy(lattice, (i2,j2,k2))

    # propose swap
    lattice[x,y,z], lattice[i2,j2,k2] = lattice[i2,j2,k2], lattice[x,y,z]

    # local energy after
    Ea = local_energy(lattice, (x,y,z)) + local_energy(lattice, (i2,j2,k2))
    dE = Ea - Eb

    # Glauber / heat-bath acceptance
    p = 1.0 / (1.0 + np.exp(dE / max(kT, 1e-12)))
    if np.random.rand() >= p:
        # reject: swap back
        lattice[x,y,z], lattice[i2,j2,k2] = lattice[i2,j2,k2], lattice[x,y,z]
    return lattice


def local_energy(lattice, pos, J1=J1, J2=J2):
    """
    Compute the energy contribution from a single lattice site.
    The local energy is defined as the sum over interactions with
    its neighbors (with a 1/2 factor so that each bond is shared).
    """
    
    x, y, z = pos
    s = lattice[x, y, z]
    nx, ny, nz = lattice.shape
    energy = 0.0

    # First neighbors:
    for dx,dy,dz in first_neighbour_offsets:
        ii = (x + dx) % nx
        jj = (y + dy) % ny
        kk = (z + dz) % nz
        energy += -J1/2 * s * lattice[ii, jj, kk]   # 1/2 avoids double counting

    # Second neighbors:
    for dx,dy,dz in second_neighbour_offsets:
        ii = (x + dx) % nx
        jj = (y + dy) % ny
        kk = (z + dz) % nz
        energy += -J2/2 * s * lattice[ii, jj, kk]

    return energy


def calculate_total_energy(lattice, J1=J1, J2=J2):
    """Vectorized total energy (count each bond once)."""
    energy = 0.0
    
    # First neighbors:
    for dx,dy,dz in first_neighbour_offsets:
        shifted = np.roll(np.roll(np.roll(lattice, dx, axis=0), dy, axis=1), dz, axis=2)
        energy += -J1 * np.sum(lattice * shifted)
    
    # Second neighbors:
    for dx,dy,dz in second_neighbour_offsets:
        shifted = np.roll(np.roll(np.roll(lattice, dx, axis=0), dy, axis=1), dz, axis=2)
        energy += -J2 * np.sum(lattice * shifted)
        
    return 1/2 * energy  # 1/2 because each bond is counted twice in the sums above

def compute_interfacial_area(lattice, lattice_constant=3.0):
    interface_bonds = 0

    # count each bond once using +x, +y, +z neighbours
    # +x
    diff = lattice != np.roll(lattice, shift=-1, axis=0)
    interface_bonds += np.sum(diff)
    # +y
    diff = lattice != np.roll(lattice, shift=-1, axis=1)
    interface_bonds += np.sum(diff)
    # +z
    diff = lattice != np.roll(lattice, shift=-1, axis=2)
    interface_bonds += np.sum(diff)

    # each counted once by construction
    area_nm2 = interface_bonds * (lattice_constant ** 2)
    return area_nm2, interface_bonds


def plot_lattice_3d(lattice, downsample_factor=2):
    arr = lattice[::downsample_factor, ::downsample_factor, ::downsample_factor]
    filled = arr != 0
    cmap = colors.ListedColormap(['blue', 'red'])
    norm = colors.BoundaryNorm([-1.5, 0, 1.5], cmap.N)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(filled, facecolors=cmap(norm(arr)), edgecolor=None, alpha=0.85)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title('3D Lattice (downsampled)')
    plt.show()


# -----------------------
# Run
# -----------------------
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


