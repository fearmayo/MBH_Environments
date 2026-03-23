import numpy as np
import yt
import pickle
import glob

R_PROP_KPC = 1.0
MIN_STARS = 5

# -------------------------------------------------------------------
# Metallicity using your SneTracerField / mass recipe
# -------------------------------------------------------------------
def compute_metallicity(region):
    sne = region[("PartType0","SneTracerField")]
    gas_mass = region[("PartType0","Masses")].v  # raw float array
    Z = np.divide(sne, gas_mass, out=np.zeros_like(sne), where=gas_mass > 0)
    Z = Z / 0.02  # convert to Z/Zsun
    Z[Z < 1e-20] = 1e-20
    return Z, gas_mass

# -------------------------------------------------------------------
# Compute stellar half-mass radius
# -------------------------------------------------------------------
def compute_R50(center, star_pos, star_mass, ds):
    if len(star_pos) < MIN_STARS:
        return np.nan

    star_pos = np.array(star_pos)
    star_mass = np.array(star_mass)

    dist = np.linalg.norm(star_pos - center, axis=1)
    idx = np.argsort(dist)
    dist_sorted = dist[idx]
    mass_sorted = star_mass[idx]

    cum_mass = np.cumsum(mass_sorted)
    half_mass = cum_mass[-1] / 2.0

    R50_code = dist_sorted[np.searchsorted(cum_mass, half_mass)]
    return float(ds.quan(R50_code, "code_length").to("kpc"))

# -------------------------------------------------------------------
# Load snapshot redshifts
# -------------------------------------------------------------------
def load_snapshot_redshifts(base):
    snapdirs = sorted(glob.glob(f"{base}/snapdir_*"))
    snap_info = {}

    for path in snapdirs:
        snap = int(path.split("_")[-1])
        try:
            ds = yt.load(f"{path}/snap_{snap:03d}.0.hdf5")
            snap_info[snap] = ds.current_redshift
        except Exception as e:
            print(f"[WARN] Cannot load snapshot {snap}: {e}")

    return snap_info

# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def extract_galaxy_properties(
    base,
    galaxies_file="merger_galaxies.pkl",
    sinks_file="sink_particle.pkl",
    outfile="galaxy_properties.pkl"
):

    # Load sinks
    from DataReader import Reader
    R = Reader(base)
    sinks = R.pickle_reader("sink_particle.pkl")

    # Load galaxy centers from Step 2
    with open(galaxies_file, "rb") as f:
        galaxies = pickle.load(f)

    # Load snapshot redshifts
    snap_z = load_snapshot_redshifts(base)

    # Cache snapshots
    snap_cache = {}
    results = {}

    # -----------------------------------------
    # MAIN LOOP OVER GALAXIES
    # -----------------------------------------
    for G in galaxies:

        galID = G["GalaxyID"]
        snap = G["Snapshot"]
        z_snap = G["Redshift"]
        center = np.array(G["Center"])
        BH_id = G["PrimaryID"]

        print(f"\n=== GalaxyID {galID} | Snapshot {snap} | z = {z_snap:.2f} ===")

        # Load or reuse snapshot
        if snap not in snap_cache:
            print(f"Loading snapshot {snap}...")
            ds = yt.load(f"{base}/snapdir_{snap:03d}/snap_{snap:03d}.0.hdf5")
            snap_cache[snap] = ds
        else:
            ds = snap_cache[snap]

        center_yt = ds.arr(center, "code_length")
        region = ds.sphere(center_yt, (R_PROP_KPC, "kpc"))

        # -----------------------------------------
        # BH remnant mass (from sink evolution)
        # -----------------------------------------
        BH_mass = sinks[BH_id]["evolution"][min(
            sinks[BH_id]["evolution"].keys(),
            key=lambda t: abs(t - 1/(1+z_snap))
        )]["StellarMass"]

        print(f" BH remnant pos  : {center.tolist()} (code units)")
        print(f" BH remnant mass : {BH_mass:.3e} Msun (from sinks)")

        # -----------------------------------------
        # Gas properties
        # -----------------------------------------
        gas_mass = region[("PartType0","Masses")].to("Msun").v
        Mgas = float(gas_mass.sum())

        Z_raw, gas_mass_raw = compute_metallicity(region)
        if gas_mass_raw.sum() > 0:
            Z_gas = float(np.sum(Z_raw * gas_mass_raw) / np.sum(gas_mass_raw))
        else:
            Z_gas = np.nan

        print(f" Gas mass        : {Mgas:.3e} Msun")
        print(f" Gas metallicity : {Z_gas:.3e} Zsun")

        # -----------------------------------------
        # Dark matter
        # -----------------------------------------
        try:
            dm_mass = region[("PartType1", "Masses")].to("Msun").v
            Mdm = float(dm_mass.sum())
        except:
            Mdm = np.nan

        print(f" Halo mass (DM)  : {Mdm:.3e} Msun")

        # -----------------------------------------
        # Stellar component
        # -----------------------------------------
        star_pos4 = region[("PartType4","Coordinates")].to("code_length").v
        star_mass4 = region[("PartType4","Masses")].to("Msun").v

        ids5 = region[("PartType5","ParticleIDs")].v.astype(int)
        pos5 = region[("PartType5","Coordinates")].to("code_length").v
        mass5 = region[("PartType5","Masses")].to("Msun").v

        star_pos5 = []
        star_mass5 = []

        for pid, p5, m5 in zip(ids5, pos5, mass5):
            if pid in sinks:
                ptype = sinks[pid]["meta"]["Type"]
                if ptype == 3:   # MBH → skip
                    continue
                # else, it's a star
                star_pos5.append(p5)
                star_mass5.append(m5)

        # Merge stars
        star_pos = np.vstack(
            [star_pos4] + ([np.array(star_pos5)] if len(star_pos5) > 0 else [])
        )
        star_mass = np.concatenate([star_mass4, np.array(star_mass5)])

        Mstar = float(star_mass.sum())
        print(f" Stellar mass    : {Mstar:.3e} Msun")

        # -----------------------------------------
        # R50
        # -----------------------------------------
        R50 = compute_R50(center, star_pos, star_mass, ds)
        print(f" R50 (half-mass) : {R50:.3f} kpc")

        # -----------------------------------------
        # Store result
        # -----------------------------------------
        results[galID] = {
            "GalaxyID": galID,
            "Snapshot": snap,
            "Redshift": z_snap,
            "Center": center.tolist(),
            "BHRemnantMass": BH_mass,
            "GasMass": Mgas,
            "HaloMass": Mdm,
            "StellarMass": Mstar,
            "GasMetallicity": Z_gas,
            "R50_kpc": R50,
        }

    # ============================================================
    # GLOBAL TOP 5 MOST MASSIVE BH REMNANTS (AFTER LOOP)
    # ============================================================
    sorted_gals = sorted(
        results.values(),
        key=lambda x: x["BHRemnantMass"],
        reverse=True
    )

    top5 = sorted_gals[:5]

    print("\n================ GLOBAL TOP 5 MOST MASSIVE BH REMNANTS ================\n")
    for g in top5:
        print(f"GalaxyID        : {g['GalaxyID']}")
        print(f"Snapshot        : {g['Snapshot']}")
        print(f"Redshift        : {g['Redshift']:.3f}")
        print(f"BH Mass         : {g['BHRemnantMass']:.3e} Msun")
        print(f"Gas Mass        : {g['GasMass']:.3e} Msun")
        print(f"Halo Mass       : {g['HaloMass']:.3e} Msun")
        print(f"Stellar Mass    : {g['StellarMass']:.3e} Msun")
        print(f"Gas Metallicity : {g['GasMetallicity']:.3e} Zsun")
        print(f"R50 (kpc)       : {g['R50_kpc']:.3f}")
        print(f"Center (code)   : {g['Center']}")
        print("--------------------------------------------------------------------")

    # ------------------------------------------------------------
    # Save file
    # ------------------------------------------------------------
    with open(outfile, "wb") as f:
        pickle.dump(results, f)

    print(f"\nSaved galaxy properties for {len(results)} galaxies → {outfile}")

    return results


if __name__ == "__main__":
    base = "/home/daxal/data/ProductionRuns/Renaissance/NoFeedback/"
    extract_galaxy_properties(base)
