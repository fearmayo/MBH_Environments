#!/usr/bin/env python3
import h5py
import numpy as np
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

# ---------------------------------------------------------
# Publication-quality Matplotlib styling
# ---------------------------------------------------------
plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 20,
    "axes.titlesize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 16,
    "figure.titlesize": 24,
    "lines.linewidth": 2,
    "axes.linewidth": 1.8,
    "xtick.major.width": 1.6,
    "ytick.major.width": 1.6,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# CATALOG DEFINITIONS
# ============================================================
# Add/remove entries here to change which catalogs are plotted.
CATALOGS = [
    {
        "name": "SEEDZ_Full",
        "file": "/home/regan/Dropbox/LISA_GW/MBHEnvironments/Catalogues/SEEDZ/Normal1_FullFeedback/MBH_Environment_Catalog_SEEDZ.hdf5",
        "color": "red",
        "marker": "o",
        "alpha": 0.85, 
    },
    {
        "name": "SEEDZ",
        "file": "/home/regan/Dropbox/LISA_GW/MBHEnvironments/Catalogues/SEEDZ/Normal1_WeakFeedback/MBH_Environment_Catalog_SEEDZ.hdf5",
        "color": "black",
        "marker": "o",
        "alpha": 0.85, 
    },
    {
        "name": "L-Galaxies (MRII)",
        "file": "/home/regan/Dropbox/LISA_GW/MBHEnvironments/Catalogues/LGalaxies/MBH_Environment_Catalog_LGalaxiesBH_MRII.hdf5",
        "color": "tab:orange",
        "marker": "^",
         "alpha": 0.05, 
    },
    {
        "name": "Romulus25",
        "file": "/home/regan/Dropbox/LISA_GW/MBHEnvironments/Catalogues/Romulus/MBH_Environment_Catalog_Romulus25.hdf5",
        "color": "blue",
        "marker": "s",
        "alpha": 0.35, 
    },
]

# Scatter appearance tuning — small/transparent points work best when
# catalogs have very different N, so no single one visually dominates.
MARKER_SIZE = 12
ALPHA = 0.35


# ============================================================
# LOAD + CLEAN A SINGLE CATALOG
# ============================================================

def load_catalog(filename):
    with h5py.File(filename, "r") as f:
        BH_Primary = np.array(f["Binaries"]["PrimaryMass"])
        BH_Secondary = np.array(f["Binaries"]["SecondaryMass"])
        Z = np.array(f["HostGalaxy"]["HostGalaxyMetallicity"])
        MStellar = np.array(f["HostGalaxy"]["HostGalaxyStellarMass"])
        Redshift = np.array(f["HostGalaxy"]["HostGalaxyRedshift"])
    return BH_Primary, BH_Secondary, Z, MStellar, Redshift


def load_and_clean(catalog):
    name, filename = catalog["name"], catalog["file"]
    logging.info(f"Reading catalog: {name} ({filename})")

    BH_Primary, BH_Secondary, Z, MStellar, Redshift = load_catalog(filename)
    logging.info(f"  {name}: {len(BH_Primary)} merger systems before masking")

    mask = (
        (BH_Primary > 0) & (BH_Secondary > 0) & (Z > 0) & (MStellar > 0) &
        np.isfinite(BH_Primary) & np.isfinite(BH_Secondary) &
        np.isfinite(Z) & np.isfinite(MStellar)
    )

    BH_Primary = BH_Primary[mask]
    BH_Secondary = BH_Secondary[mask]
    RemnantBHMass = BH_Primary + BH_Secondary
    Z = Z[mask]
    MStellar = MStellar[mask]
    Redshift = Redshift[mask]

    logging.info(f"  {name}: {len(BH_Primary)} merger systems after masking")

    return {
        "name": name,
        "color": catalog["color"],
        "marker": catalog["marker"],
        "alpha": catalog["alpha"],
        "RemnantBHMass": RemnantBHMass,
        "Z": Z,
        "MStellar": MStellar,
        "Redshift": Redshift,
    }


# ============================================================
# PLOTTING
# ============================================================

def plot_comparison(datasets, xkey, ykey, xlabel, ylabel, outname,
                     xscale="log", yscale="log", xlim=None):
    plt.figure(figsize=(10, 6))

    for d in datasets:
        plt.scatter(
            d[xkey], d[ykey],
            s=MARKER_SIZE, alpha=d["alpha"],
            color=d["color"], marker=d["marker"],
            label=f"{d['name']} (N={len(d[xkey])})",
            rasterized=True,  # keeps file size sane with many points
        )

    if xscale:
        plt.xscale(xscale)
    if yscale:
        plt.yscale(yscale)
    if xlim:
        plt.xlim(*xlim)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(markerscale=3, frameon=False)
    plt.tight_layout()
    plt.savefig(outname)
    plt.close()
    logging.info(f"Saved {outname}")


def make_plots(datasets):
    plot_comparison(
        datasets, "Z", "RemnantBHMass",
        xlabel=r"Mass-Weighted Gas Metallicity [$\rm{Z/Z_\odot}$]",
        ylabel=r"BH Remnant Mass [$\rm{M_\odot}$]",
        outname="Comparison_BHMass_Metallicity.png",
    )

    plot_comparison(
        datasets, "MStellar", "RemnantBHMass",
        xlabel=r"Stellar Mass [$\rm{M_\odot}$]",
        ylabel=r"BH Remnant Mass [$\rm{M_\odot}$]",
        outname="Comparison_BHMass_StellarMass.png",
    )

    plot_comparison(
        datasets, "Redshift", "RemnantBHMass",
        xlabel=r"Redshift",
        ylabel=r"BH Remnant Mass [$\rm{M_\odot}$]",
        outname="Comparison_BHMass_Redshift.png",
        xscale=None, xlim=(20, 0),
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    datasets = [load_and_clean(cat) for cat in CATALOGS]
    make_plots(datasets)
