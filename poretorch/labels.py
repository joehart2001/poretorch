"""Labelling the void into discrete pores across periodic boundaries.

Connected-component labelling on the void mask is only half the job: a pore that
leaves one face of the cell and re-enters through the opposite face is one pore,
not two, so components touching across a boundary have to be merged. Doing that
merge while tracking *which* periodic image each piece was reached in also
answers the more interesting question of whether the pore is a closed pocket
or a channel that runs through the material without end.

A pore percolates when it can be traced back to itself in a different periodic
image. The set of lattice vectors realised by such round trips generates a
lattice whose rank is the pore's dimensionality:

=============  ===================================================
Dimensionality Meaning
=============  ===================================================
0              closed pocket, finite in every direction
1              channel, infinite along one direction
2              planar void, infinite within a plane
3              fully connected pore network
=============  ===================================================

This is a stricter test than asking whether a pore touches two opposite faces,
which a small pocket straddling a boundary also passes.
"""

from dataclasses import dataclass, field
from itertools import product

import numpy as np
from scipy import ndimage as ndi

DIMENSIONALITY_NAMES = {0: "closed", 1: "channel", 2: "planar", 3: "network"}


@dataclass(frozen=True)
class Pore:
    """
    A single connected void region.

    Attributes
    ----------
    id
        Label of this pore in the label grid (1-based).
    n_voxels
        Number of void voxels belonging to the pore.
    volume
        Pore volume in cubic Angstrom.
    max_clearance
        Largest empty-sphere radius anywhere in the pore, in Angstrom.
    mean_clearance
        Mean empty-sphere radius over the pore, in Angstrom.
    dimensionality
        Rank of the lattice of periodic round trips: 0 closed, 1 channel,
        2 planar, 3 fully connected.
    percolation_vectors
        Integer lattice vectors along which the pore closes on itself.
    """

    id: int
    n_voxels: int
    volume: float
    max_clearance: float
    mean_clearance: float
    dimensionality: int
    percolation_vectors: tuple[tuple[int, int, int], ...] = ()

    @property
    def is_percolating(self) -> bool:
        """Whether the pore extends without bound in at least one direction."""
        return self.dimensionality > 0

    @property
    def kind(self) -> str:
        """Human-readable dimensionality: closed, channel, planar or network."""
        return DIMENSIONALITY_NAMES[self.dimensionality]

    @property
    def spans_axes(self) -> tuple[bool, bool, bool]:
        """
        Whether the pore runs without bound along each cell axis.

        An axis counts as spanned when some periodic round trip has a non-zero
        component along it.
        """
        spans = [False, False, False]
        for vector in self.percolation_vectors:
            for axis, component in enumerate(vector):
                if component != 0:
                    spans[axis] = True
        return tuple(spans)


@dataclass
class PoreLabels:
    """
    The void partitioned into pores.

    Attributes
    ----------
    labels
        Integer grid shaped ``(nx, ny, nz)``. 0 is solid, positive values are
        pore ids.
    pores
        One :class:`Pore` per label, ordered by id.
    voxel_volume
        Volume of a single voxel in cubic Angstrom.
    """

    labels: np.ndarray
    pores: list[Pore] = field(default_factory=list)
    voxel_volume: float = 1.0

    @property
    def n_pores(self) -> int:
        """Number of distinct pores."""
        return len(self.pores)

    @property
    def total_volume(self) -> float:
        """Combined volume of every pore in cubic Angstrom."""
        return float(sum(pore.volume for pore in self.pores))

    def percolating(self) -> list[Pore]:
        """Pores that extend without bound in at least one direction."""
        return [pore for pore in self.pores if pore.is_percolating]

    def closed(self) -> list[Pore]:
        """Pores that are finite in every direction."""
        return [pore for pore in self.pores if not pore.is_percolating]

    def summary(self) -> dict:
        """
        Aggregate pore statistics.

        Returns
        -------
        dict
            Counts and volumes split by percolating versus closed, plus the
            distribution over dimensionalities.
        """
        percolating = self.percolating()
        closed = self.closed()
        total = self.total_volume
        by_dim = {name: 0 for name in DIMENSIONALITY_NAMES.values()}
        for pore in self.pores:
            by_dim[pore.kind] += 1
        percolating_volume = float(sum(p.volume for p in percolating))
        return {
            "n_pores": self.n_pores,
            "n_percolating": len(percolating),
            "n_closed": len(closed),
            "total_volume_A3": total,
            "percolating_volume_A3": percolating_volume,
            "closed_volume_A3": total - percolating_volume,
            "percolating_volume_fraction": (
                percolating_volume / total if total > 0 else 0.0
            ),
            "counts_by_kind": by_dim,
        }


def label_pores(
    void_mask,
    clearance=None,
    voxel_volume: float = 1.0,
    pbc=(True, True, True),
    connectivity: int = 1,
    min_voxels: int = 1,
) -> PoreLabels:
    """
    Label the void into pores, merging across periodic boundaries.

    Parameters
    ----------
    void_mask
        Boolean grid shaped ``(nx, ny, nz)``, ``True`` in the void. Accepts a
        torch tensor or a numpy array.
    clearance
        Optional clearance grid of the same shape, used for the per-pore radius
        statistics. Accepts a torch tensor or a numpy array.
    voxel_volume
        Volume of one voxel in cubic Angstrom.
    pbc
        Periodicity flags per axis. A non-periodic axis is not merged across.
    connectivity
        Voxel connectivity, 1 for face-sharing (6 neighbours), 2 for edges
        (18), 3 for corners (26). Face-sharing is the conventional choice and
        avoids linking pores that only touch diagonally.
    min_voxels
        Pores smaller than this are discarded and their voxels set to solid.

    Returns
    -------
    PoreLabels
        Label grid and per-pore statistics.
    """
    mask = _to_numpy(void_mask).astype(bool, copy=False)
    if mask.ndim != 3:
        raise ValueError(f"void_mask must be 3D, got shape {mask.shape}")
    if connectivity not in (1, 2, 3):
        raise ValueError(f"connectivity must be 1, 2 or 3, got {connectivity}")

    pbc = _pbc_tuple(pbc)
    structure = ndi.generate_binary_structure(rank=3, connectivity=connectivity)
    raw, n_raw = ndi.label(mask, structure=structure)

    if n_raw == 0:
        return PoreLabels(
            labels=np.zeros(mask.shape, dtype=np.int32),
            pores=[],
            voxel_volume=voxel_volume,
        )

    contacts = _boundary_contacts(raw, pbc, connectivity)
    groups, percolation = _merge_with_images(n_raw, contacts)

    labels, pores = _finalise_labels(
        raw=raw,
        groups=groups,
        percolation=percolation,
        clearance=None if clearance is None else _to_numpy(clearance),
        voxel_volume=voxel_volume,
        min_voxels=min_voxels,
    )
    return PoreLabels(labels=labels, pores=pores, voxel_volume=voxel_volume)


def _to_numpy(array):
    """
    Return a numpy view of a numpy array or torch tensor.

    Parameters
    ----------
    array
        Numpy array or torch tensor, on any device.

    Returns
    -------
    np.ndarray
        Equivalent numpy array.
    """
    if hasattr(array, "detach"):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def _pbc_tuple(pbc) -> tuple[bool, bool, bool]:
    """
    Normalise periodicity flags to a three-tuple of booleans.

    Parameters
    ----------
    pbc
        A single boolean or an iterable of three.

    Returns
    -------
    tuple[bool, bool, bool]
        Per-axis periodicity.
    """
    if isinstance(pbc, (bool, np.bool_)):
        return (bool(pbc),) * 3
    values = tuple(bool(x) for x in _to_numpy(pbc).ravel())
    if len(values) != 3:
        raise ValueError(f"pbc must have three entries, got {len(values)}")
    return values


def _half_offsets(connectivity: int) -> list[tuple[int, int, int]]:
    """
    Enumerate one direction of each neighbour pair for a connectivity.

    Taking half the neighbour offsets avoids visiting every adjacency twice.

    Parameters
    ----------
    connectivity
        Voxel connectivity, 1, 2 or 3.

    Returns
    -------
    list[tuple[int, int, int]]
        Offsets with at most ``connectivity`` non-zero components, keeping only
        the lexicographically positive member of each ``+/-`` pair.
    """
    offsets = []
    for offset in product((-1, 0, 1), repeat=3):
        nonzero = sum(1 for c in offset if c != 0)
        if nonzero == 0 or nonzero > connectivity:
            continue
        if offset > tuple(-c for c in offset):  # keep one of each +/- pair
            offsets.append(offset)
    return offsets


def _boundary_contacts(raw: np.ndarray, pbc, connectivity: int):
    """
    Find adjacencies that cross a periodic boundary, with their lattice shifts.

    Only thin boundary slabs are inspected, so the cost is proportional to the
    cell surface rather than its volume.

    Parameters
    ----------
    raw
        Label grid from non-periodic connected-component labelling.
    pbc
        Per-axis periodicity flags.
    connectivity
        Voxel connectivity used for the labelling.

    Returns
    -------
    list[tuple[np.ndarray, np.ndarray, tuple[int, int, int]]]
        For each wrap pattern, the labels on either side of the boundary and the
        integer lattice shift taken in crossing it.
    """
    shape = raw.shape
    contacts = []

    for offset in _half_offsets(connectivity):
        moving = [a for a in range(3) if offset[a] != 0]
        wrappable = [a for a in moving if pbc[a]]

        # Each non-empty subset of wrappable axes is a distinct boundary slab:
        # those axes wrap, the remaining moving axes stay inside the cell.
        for size in range(1, len(wrappable) + 1):
            for wrapping in _subsets(wrappable, size):
                src, dst = [], []
                valid = True
                for axis in range(3):
                    step = offset[axis]
                    length = shape[axis]
                    if step == 0:
                        src.append(slice(0, length))
                        dst.append(slice(0, length))
                    elif axis in wrapping:
                        if length < 2:
                            # A single voxel across the cell wraps onto
                            # itself, so the interior slice below would
                            # be empty.
                            src.append(slice(0, 1))
                            dst.append(slice(0, 1))
                            continue
                        first = length - 1 if step > 0 else 0
                        last = 0 if step > 0 else length - 1
                        src.append(slice(first, first + 1))
                        dst.append(slice(last, last + 1))
                    else:
                        if length < 2:
                            valid = False
                            break
                        if step > 0:
                            src.append(slice(0, length - 1))
                            dst.append(slice(1, length))
                        else:
                            src.append(slice(1, length))
                            dst.append(slice(0, length - 1))
                if not valid:
                    continue

                a = raw[tuple(src)].ravel()
                b = raw[tuple(dst)].ravel()
                keep = (a > 0) & (b > 0)
                if not keep.any():
                    continue
                shift = tuple(
                    (1 if offset[axis] > 0 else -1) if axis in wrapping else 0
                    for axis in range(3)
                )
                contacts.append((a[keep], b[keep], shift))
    return contacts


def _subsets(items, size):
    """
    Enumerate subsets of a given size.

    Parameters
    ----------
    items
        Sequence to draw from.
    size
        Subset size.

    Yields
    ------
    tuple
        Each subset, as a tuple.
    """
    from itertools import combinations

    yield from combinations(items, size)


def _merge_with_images(n_raw: int, contacts):
    """
    Merge components across boundaries while recording periodic round trips.

    A breadth-first traversal assigns every component the lattice image it was
    first reached in. Meeting an already-visited component in a *different*
    image means the pore closes on itself across the lattice, and the difference
    of the two images is a percolation vector.

    Parameters
    ----------
    n_raw
        Number of raw components.
    contacts
        Boundary adjacencies from :func:`_boundary_contacts`.

    Returns
    -------
    tuple[np.ndarray, dict]
        Group id per raw label shaped ``(n_raw + 1,)`` with entry 0 unused, and
        a mapping from group id to its set of percolation vectors.
    """
    adjacency: dict[int, set] = {}
    for a_labels, b_labels, shift in contacts:
        pairs = np.unique(np.stack([a_labels, b_labels], axis=1), axis=0)
        for a, b in pairs:
            a, b = int(a), int(b)
            # Crossing from a to b advances by +shift. The reverse subtracts it.
            adjacency.setdefault(a, set()).add((b, shift))
            adjacency.setdefault(b, set()).add((a, tuple(-c for c in shift)))

    groups = np.zeros(n_raw + 1, dtype=np.int64)
    percolation: dict[int, set] = {}
    next_group = 0

    for seed in range(1, n_raw + 1):
        if groups[seed] != 0:
            continue
        next_group += 1
        groups[seed] = next_group
        images = {seed: (0, 0, 0)}
        vectors: set[tuple[int, int, int]] = set()
        queue = [seed]
        while queue:
            node = queue.pop()
            here = images[node]
            for neighbour, shift in adjacency.get(node, ()):
                there = tuple(here[a] + shift[a] for a in range(3))
                if neighbour not in images:
                    images[neighbour] = there
                    groups[neighbour] = next_group
                    queue.append(neighbour)
                    continue
                delta = tuple(there[a] - images[neighbour][a] for a in range(3))
                if any(delta):
                    vectors.add(_canonical_vector(delta))
        if vectors:
            percolation[next_group] = vectors

    return groups, percolation


def _canonical_vector(vector) -> tuple[int, int, int]:
    """
    Fix the sign of a lattice vector so ``v`` and ``-v`` are not both stored.

    Parameters
    ----------
    vector
        Integer triple.

    Returns
    -------
    tuple[int, int, int]
        The vector, negated if needed so its first non-zero entry is positive.
    """
    for component in vector:
        if component > 0:
            return tuple(int(c) for c in vector)
        if component < 0:
            return tuple(int(-c) for c in vector)
    return (0, 0, 0)


def _lattice_rank(vectors) -> tuple[int, tuple]:
    """
    Rank of the lattice generated by a set of integer vectors.

    Parameters
    ----------
    vectors
        Iterable of integer triples.

    Returns
    -------
    tuple[int, tuple]
        The rank (0 to 3) and the vectors as a canonical sorted tuple.
    """
    unique = tuple(sorted(set(vectors)))
    if not unique:
        return 0, ()
    matrix = np.array(unique, dtype=np.float64)
    return int(np.linalg.matrix_rank(matrix)), unique


def _finalise_labels(
    raw: np.ndarray,
    groups: np.ndarray,
    percolation: dict,
    clearance,
    voxel_volume: float,
    min_voxels: int,
):
    """
    Renumber merged groups to consecutive pore ids and gather their statistics.

    Parameters
    ----------
    raw
        Raw component label grid.
    groups
        Group id per raw label.
    percolation
        Percolation vectors per group id.
    clearance
        Optional clearance grid for radius statistics.
    voxel_volume
        Volume of one voxel in cubic Angstrom.
    min_voxels
        Minimum pore size to keep.

    Returns
    -------
    tuple[np.ndarray, list[Pore]]
        Final label grid and the per-pore statistics.
    """
    merged = groups[raw]
    n_groups = int(groups.max())
    if n_groups == 0:
        return np.zeros(raw.shape, dtype=np.int32), []

    group_ids = np.arange(1, n_groups + 1)
    sizes = np.bincount(merged.ravel(), minlength=n_groups + 1)[1:]

    keep = sizes >= max(1, min_voxels)
    # Map surviving groups onto consecutive ids. Dropped groups become solid.
    lut = np.zeros(n_groups + 1, dtype=np.int32)
    lut[group_ids[keep]] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)
    labels = lut[merged]

    n_pores = int(keep.sum())
    if n_pores == 0:
        return labels, []

    index = np.arange(1, n_pores + 1)
    counts = np.bincount(labels.ravel(), minlength=n_pores + 1)[1:]

    if clearance is not None:
        max_clearance = ndi.maximum(clearance, labels=labels, index=index)
        mean_clearance = ndi.mean(clearance, labels=labels, index=index)
        max_clearance = np.atleast_1d(max_clearance)
        mean_clearance = np.atleast_1d(mean_clearance)
    else:
        max_clearance = np.full(n_pores, np.nan)
        mean_clearance = np.full(n_pores, np.nan)

    pores = []
    for new_id, group_id in enumerate(group_ids[keep], start=1):
        rank, vectors = _lattice_rank(percolation.get(int(group_id), ()))
        pores.append(
            Pore(
                id=int(new_id),
                n_voxels=int(counts[new_id - 1]),
                volume=float(counts[new_id - 1] * voxel_volume),
                max_clearance=float(max_clearance[new_id - 1]),
                mean_clearance=float(mean_clearance[new_id - 1]),
                dimensionality=rank,
                percolation_vectors=vectors,
            )
        )
    return labels, pores
