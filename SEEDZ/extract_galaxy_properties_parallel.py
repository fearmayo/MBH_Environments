#!/usr/bin/env python3
import numpy as np
import yt
import pickle
import glob
import logging
from multiprocessing import Pool, cpu_count

# ============================================================
# CONFIGURATION
# ============================================================

R_PROP_KPC = 1.0       # Extraction radius
DEDUP_RADIUS_KPC = 1.0 # Deduplication radius
MIN_STARS = 5

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)

# ============================================================
# UTILS
# ============================================================

def load_snapshot_redshifts(base):
    """Load snapshot → redshift mapping."""
    snapdirs = sorted(glob.glob(f"{base}/snapdir_*"))
    snap_z = {}

    for path in snapdirs:
        snap = int(path.split("_")[-1])
        try:
            ds = yt.load(f"{path}/snap_{snap:03d}.0.hdf5")
            snap_z[snap] = ds.current_redshift
        except Exception as e:
            logging.warning(f"Could not load snapshot {snap}: {e}")

    return snap_z


def compute_R50(center_kpc, star_pos, star_mass):
    """Compute stellar half-mass radius in kpc."""
    if len(star_pos) < MIN_STARS:
        return np.nan

    dist = np.linalg.norm(star_pos - center_kpc, axis=1)
    idx = np.argsort(dist)

    dist_sorted = dist[idx]
    mass_sorted = star_mass[idx]
    cum_mass = np.cumsum(mass_sorted)

    half_mass = cum_mass[-1] / 2
    R50 = dist_sorted[np.searchsorted(cum_mass, half_mass)]

    return float(R50)

# ============================================================
# DEDUPLICATION
# ============================================================

def deduplicate_galaxies(glist, ds):
    """
    Deduplicate galaxy centers by physical distance < 1 kpc.
    Convert code_length → kpc before comparison.
    """
    unique = []
    for g in glist:
        center_code = np.array(g["Center"])
        center_kpc = ds.arr(center_code, "code_length").to("kpc").v

        found = False
        for u in unique:
            d = np.linalg.norm(center_kpc - u["center_kpc"])
            if d < DEDUP_RADIUS_KPC:
                u["galaxy_ids"].append(g["GalaxyID"])
                u["bh_ids"].append(g["PrimaryID"])
                found = True
                break

        if not found:
            unique.append({
                "center_kpc": center_kpc,
                "snapshot": g["Snapshot"],
                "redshift": g["Redshift"],
                "galaxy_ids": [g["GalaxyID"]],
                "bh_ids": [g["PrimaryID"]],
            })

    return unique

# ============================================================
# PROCESS SNAPSHOT
# ============================================================

def process_snapshot(args):
    """
    Process all galaxies in one snapshot.
    """
    snap, galaxies, base, sinks = args

    logging.info(f"[Snapshot {snap}] Loading...")
    ds = yt.load(f"{base}/snapdir_{snap:03d}/snap_{snap:03d}.0.hdf5")

    # Preload particle fields
    logging.info(f"[Snapshot {snap}] Preloading particle fields...")

    ad = ds.all_data()

    gas_pos  = ad[("PartType0", "Coordinates")].to("kpc").v
    gas_mass = ad[("PartType0", "Masses")].to("Msun").v
    gas_sne  = ad[("PartType0", "SneTracerField")].v

    try:
        dm_pos = ad[("PartType1", "Coordinates")].to("kpc").v
        dm_mass = ad[("PartType1", "Masses")].to("Msun").v
    except:
        dm_pos = np.zeros((0,3))
        dm_mass = np.zeros(0)

    star4_pos  = ad[("PartType4", "Coordinates")].to("kpc").v
    star4_mass = ad[("PartType4", "Masses")].to("Msun").v

    star5_pos  = ad[("PartType5", "Coordinates")].to("kpc").v
    star5_mass = ad[("PartType5", "Masses")].to("Msun").v
    star5_ids  = ad[("PartType5", "ParticleIDs")].v.astype(int)

    # Deduplication
    unique = deduplicate_galaxies(galaxies, ds)
    logging.info(f"[Snapshot {snap}] Unique galaxies: {len(unique)}")

    # Classify PartType5 stars
    star5_is_star = []
    for pid in star5_ids:
        if pid not in sinks:
            star5_is_star.append(True)
        else:
            t = sinks[pid]["meta"]["Type"]
            star5_is_star.append(t != 3)  # MBH excluded
    star5_is_star = np.array(star5_is_star)

    # Merge star particles
    star_pos  = np.vstack([star4_pos, star5_pos[star5_is_star]])
    star_mass = np.concatenate([star4_mass, star5_mass[star5_is_star]])

    # Store results
    results = {}

    for idx, g in enumerate(unique):
        center = g["center_kpc"]
        galID = min(g["galaxy_ids"])

        logging.info(f"[Snapshot {snap}] Galaxy {idx+1}/{len(unique)}")

        # Gas
        gas_dist = np.linalg.norm(gas_pos - center, axis=1)
        gmask = gas_dist < R_PROP_KPC
        Mgas = float(gas_mass[gmask].sum())

        if gmask.sum() > 0:
            Z_raw = gas_sne[gmask] / gas_mass[gmask]
            Z = float((Z_raw * gas_mass[gmask]).sum() / gas_mass[gmask].sum())
        else:
            Z = np.nan

        # DM
        dm_dist = np.linalg.norm(dm_pos - center, axis=1)
        dmask = dm_dist < R_PROP_KPC
        Mdm = float(dm_mass[dmask].sum())

        # Stars
        star_dist = np.linalg.norm(star_pos - center, axis=1)
        smask = star_dist < R_PROP_KPC
        Mstar = float(star_mass[smask].sum())

        R50 = compute_R50(center, star_pos[smask], star_mass[smask])

        # BH mass: choose largest among merged IDs
        BH_masses = []
        for bid in g["bh_ids"]:
            tvals = list(sinks[bid]["evolution"].keys())
            z = g["redshift"]
            sel = min(tvals, key=lambda t: abs(t - 1/(1+z)))
            BH_masses.append(sinks[bid]["evolution"][sel]["StellarMass"])

        BH_mass = max(BH_masses)

        results[galID] = {
            "GalaxyID": galID,
            "Snapshot": snap,
            "Redshift": g["redshift"],
            "Center": center.tolist(),
            "BHRemnantMass": BH_mass,
            "GasMass": Mgas,
            "HaloMass": Mdm,
            "StellarMass": Mstar,
            "GasMetallicity": Z,
            "R50_kpc": R50,
        }

    return results

# ============================================================
# MAIN
# ============================================================

def extract_galaxy_properties(
    base,
    galaxies_file="merger_galaxies.pkl",
    sinks_file="sink_particle.pkl",
    outfile="galaxy_properties.pkl"
):

    logging.info("Loading sinks...")
    from DataReader import Reader
    R = Reader(base)
    sinks = R.pickle_reader(sinks_file)

    logging.info("Loading galaxy list...")
    with open(galaxies_file, "rb") as f:
        galaxies = pickle.load(f)

    # Group by snapshot
    snaps = {}
    for g in galaxies:
        s = g["Snapshot"]
        snaps.setdefault(s, []).append(g)

    # Build worker tasks
    tasks = [(snap, snaps[snap], base, sinks) for snap in snaps]
    n_snaps = len(tasks)
    available = cpu_count()
    workers = min(available, n_snaps)

    logging.info(
        f"System has {available} cores → using {workers} workers "
        f"for {n_snaps} snapshots."
    )

    # Parallel execution
    with Pool(workers) as pool:
        results_list = pool.map(process_snapshot, tasks)

    # Merge all results
    results = {}
    for r in results_list:
        results.update(r)

    # Top 5 BHs
    all_gals = list(results.values())
    top5 = sorted(all_gals, key=lambda x: x["BHRemnantMass"], reverse=True)[:5]

    logging.info("============== GLOBAL TOP 5 MOST MASSIVE BH REMNANTS ==============")
    for g in top5:
        logging.info(
            f"GalaxyID {g['GalaxyID']} | z={g['Redshift']:.3f} | "
            f"BH={g['BHRemnantMass']:.3e} Msun"
        )

    # Save
    with open(outfile, "wb") as f:
        pickle.dump(results, f)

    logging.info(f"Saved {len(results)} galaxies → {outfile}")
    return results


if __name__ == "__main__":
    # Note: General astrophysics YouTube resources do not discuss YT parallel analysis. [1](https://vinfrastructure.it/2022/09/vmware-vsphere-snapshot-recommended-practices/)
    base = "/home/daxal/data/ProductionRuns/Renaissance/NoFeedback/"
    extract_galaxy_properties(base)
