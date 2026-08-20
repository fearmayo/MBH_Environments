#!/usr/bin/env python3
import h5py
import numpy as np
import matplotlib.pyplot as plt
import logging
from scipy.stats import spearmanr
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
    {
        "name": "KETJU",
        "file": "/home/regan/Dropbox/LISA_GW/MBHEnvironments/Catalogues/KETJU/MBH_Environment_Catalog_KETJU.hdf5",
        "color": "green",
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
        NumberDensity = np.array(f["Binaries"]["NumberDensity"])
        Separation = np.array(f["Binaries"]["Separation"])

        Z = np.array(f["HostGalaxy"]["HostGalaxyMetallicity"])
        MStellar = np.array(f["HostGalaxy"]["HostGalaxyStellarMass"])
        Redshift = np.array(f["HostGalaxy"]["HostGalaxyRedshift"])
        HaloMass = np.array(f["HostGalaxy"]["HostGalaxyHaloMass"])
        R50 = np.array(f["HostGalaxy"]["HostGalaxyR50"])
        Position = np.array(f["HostGalaxy"]["HostGalaxyPosition"]).astype(str)  # bytes -> str
    return BH_Primary, BH_Secondary, Z, MStellar, Redshift, NumberDensity, Separation, HaloMass, R50, Position


def load_and_clean(catalog):
    name, filename = catalog["name"], catalog["file"]
    logging.info(f"Reading catalog: {name} ({filename})")

    BH_Primary, BH_Secondary, Z, MStellar, Redshift, NumberDensity, Separation, HaloMass, R50, Position = load_catalog(filename)
    logging.info(f"  {name}: {len(BH_Primary)} merger systems before masking")

    mask = (
        (BH_Primary > 0) & (BH_Secondary > 0) & (Z > 0) & (MStellar > 0) & (HaloMass > 0) & 
        np.isfinite(BH_Primary) & np.isfinite(BH_Secondary) &
        np.isfinite(Z) & np.isfinite(MStellar) & np.isfinite(HaloMass)
    )

    BH_Primary = BH_Primary[mask]
    BH_Secondary = BH_Secondary[mask]
    RemnantBHMass = BH_Primary + BH_Secondary
    # Ensure primary is always the more massive of the two (guard against ordering issues)
    M1 = np.maximum(BH_Primary, BH_Secondary)
    M2 = np.minimum(BH_Primary, BH_Secondary)
    MassRatio = M2 / M1   # q in (0, 1]
    # Chirp mass
    ChirpMass = (M1 * M2)**(3/5) / (M1 + M2)**(1/5)
    Z = Z[mask]
    MStellar = MStellar[mask]
    Redshift = Redshift[mask]

    NumberDensity = NumberDensity[mask]
    Separation = Separation[mask]
    HaloMass = HaloMass[mask]
    R50 = R50[mask]
    Position = Position[mask]

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
        "NumberDensity": NumberDensity,
        "Separation": Separation,
        "HaloMass": HaloMass,
        "R50": R50,
        "Position" : Position,
        "MassRatio": MassRatio,
        "ChirpMass": ChirpMass
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


    #------------------------------------------------------------------
# Plot: BH Remnant Mass vs Halo Mass, split by Central/Satellite
# Why: HostGalaxyPosition (central vs satellite) is a genuine
# environment-type proxy - satellites sit in denser group/cluster
# potentials and are subject to stripping/harassment, centrals are
# more isolated. Splitting the BH-halo relation by this flag tests
# whether MBHM environment type (not just halo mass) shapes the
# relation - e.g. do satellite-hosted BHs undershoot the central
# relation at fixed halo mass?
# ------------------------------------------------------------------
def plot_central_satellite(datasets, xkey, ykey, xlabel, ylabel, outname):
    plt.figure(figsize=(10, 6))

    # Pool all catalogs together, split only by Position
    all_x = np.concatenate([d[xkey] for d in datasets])
    all_y = np.concatenate([d[ykey] for d in datasets])
    all_pos = np.concatenate([d["Position"] for d in datasets])

    for label, color, marker in [("central", "tab:blue", "o"),
                                  ("satellite", "tab:red", "^")]:
        sel = all_pos == label
        plt.scatter(all_x[sel], all_y[sel], s=12, alpha=0.4,
                    color=color, marker=marker,
                    label=f"{label} (N={sel.sum()})", rasterized=True)

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(markerscale=3, frameon=False)
    plt.tight_layout()
    plt.savefig(outname)
    plt.close()
    logging.info(f"Saved {outname}")



#Correlation coefficient summary (Spearman) per catalog

#For each catalog, compute the Spearman rank correlation between BH remnant mass and
#each environment variable (HaloMass, StellarMass, Metallicity, R50) — then plot those
#correlation coefficients side-by-side across catalogs. If SEEDZ, Romulus, L-Galaxies,
#and KETJU all show a strong positive correlation between HaloMass and BH mass, that's
#evidence the relationship is a genuine physical signal, independent of subgrid model.
#If they diverge — some show strong correlation, others weak or none — that tells you
#the relation is likely an artifact of a specific model's implementation rather than a
#robust prediction.

def compute_correlations(datasets, ykey="RemnantBHMass",
                          xkeys=("HaloMass", "MStellar", "Z", "R50")):
    """
    For each catalog, compute Spearman rank correlation between ykey
    and each environment variable in xkeys. Returns a dict:
      {catalog_name: {xkey: (rho, pvalue)}}
    """
    results = {}
    for d in datasets:
        results[d["name"]] = {}
        for xkey in xkeys:
            rho, pval = spearmanr(d[xkey], d[ykey])
            results[d["name"]][xkey] = (rho, pval)
    return results


def plot_correlation_summary(datasets, ykey="RemnantBHMass",
                              xkeys=("HaloMass", "MStellar", "Z", "R50"),
                              xlabels=("Halo Mass", "Stellar Mass", "Metallicity", "R50"),
                              outname="Correlation_Summary.png"):
    corr = compute_correlations(datasets, ykey, xkeys)

    n_cats = len(datasets)
    n_vars = len(xkeys)
    x = np.arange(n_vars)
    width = 0.8 / n_cats

    plt.figure(figsize=(10, 6))
    for i, d in enumerate(datasets):
        rhos = [corr[d["name"]][k][0] for k in xkeys]
        N = len(d[xkeys[0]])   # number of merger systems in this catalog
        plt.bar(x + i*width, rhos, width,
                label=f"{d['name']} (N={N})", 
                color=d["color"])
        
    plt.axhline(0, color="black", linewidth=1)
    plt.xticks(x + width*(n_cats-1)/2, xlabels)
    plt.ylabel(rf"Spearman $\rho$ (vs {ykey})") 
    plt.legend(frameon=False)
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

    plot_comparison(
        datasets, "HaloMass", "RemnantBHMass",
        xlabel=r"HaloMass [$\rm{M_\odot}$]",
        ylabel=r"BH Remnant Mass [$\rm{M_\odot}$]",
        outname="Comparison_BHMass_HaloMass.png",
    )

    # ------------------------------------------------------------------
    # Plot: Galaxy Size (R50) vs Halo Mass
    # Tests galaxy compactness as a function of the halo mass
    # environment rather than stellar mass. Divergence from the classic
    # size-stellar-mass relation would flag environment-driven structural
    # effects (e.g. tidal compaction in dense halos, or extended low-
    # surface-brightness growth in less massive/isolated halos).
    # ------------------------------------------------------------------
    plot_comparison(
        datasets, "HaloMass", "R50",
        xlabel=r"Host Halo Mass [$\rm{M_\odot}$]",
        ylabel=r"Stellar Half Mass Radius R50 [kpc]",
        outname="Comparison_R50_HaloMass.png",
    )


    # ------------------------------------------------------------------
    # Plot: Mass Ratio (q = M2/M1) vs Redshift
    # Tests whether the model predict that MBHMs are dominated by major mergers (q -> 1)
    # or minor mergers (q << 1), and whether that balance shifts across
    # cosmic time. Directly relevant to LISA detectability, since
    # waveform amplitude and SNR depend strongly on mass ratio.
    # ------------------------------------------------------------------
    plot_comparison(
        datasets, "Redshift", "MassRatio",
        xlabel=r"Redshift",
        ylabel=r"Mass Ratio $q = M_2/M_1$",
        outname="Comparison_MassRatio_Redshift.png",
        xscale=None, yscale=None, xlim=(20, 0),
    )

  
    plot_comparison(
        datasets, "HaloMass", "MassRatio",
        xlabel=r"Host Halo Mass [$\rm{M_\odot}$]",
        ylabel=r"Mass Ratio $q = M_2/M_1$",
        outname="Comparison_MassRatio_HaloMass.png",
        xscale="log", yscale=None,
    )

    plot_comparison(
        datasets, "MStellar", "MassRatio",
        xlabel=r"Stellar Mass [$\rm{M_\odot}$]",
        ylabel=r"Mass Ratio $q = M_2/M_1$",
        outname="Comparison_MassRatio_StellarMass.png",
        xscale="log", yscale=None,
    )


    plot_comparison(
        datasets, "R50", "MassRatio",
        xlabel=r"Stellar Half Mass Radius R50 [kpc]",
        ylabel=r"Mass Ratio $q = M_2/M_1$",
        outname="Comparison_MassRatio_R50.png",
        xscale="log", yscale=None,
    )


    plot_correlation_summary(
    datasets,
    ykey="RemnantBHMass",
    xkeys=("HaloMass", "MStellar", "Z", "R50"),
    xlabels=("Halo Mass", "Stellar Mass", "Metallicity", "R50"),
    outname="Correlation_Summary_RemnantMass.png",
    )


    # ------------------------------------------------------------------
    # Correlation summary: environment vs Mass Ratio (q = M2/M1)
    # Why: RemnantMass is dominated by M1 (the pre-existing primary),
    # so correlating environment against it mostly re-tests known
    # M_BH-M_halo/M* scaling relations. Mass ratio is a genuine property
    # of the MERGER EVENT itself - testing whether environment predicts
    # major vs minor mergers is a more direct environment-drives-merger
    # test than remnant mass.
    # ------------------------------------------------------------------
    plot_correlation_summary(
        datasets,
        ykey="MassRatio",
        xkeys=("HaloMass", "MStellar", "Z", "R50"),
        xlabels=("Halo Mass", "Stellar Mass", "Metallicity", "R50"),
        outname="Correlation_Summary_MassRatio.png",
    )

    # ------------------------------------------------------------------
    # Correlation summary: environment vs Chirp Mass
    # Why: Chirp mass is the LISA-relevant combined mass and, like mass
    # ratio, is less dominated by M1 alone than the simple remnant sum -
    # directly ties the environment correlation test back to GW/LISA
    # science relevance.
    # ------------------------------------------------------------------
    plot_correlation_summary(
        datasets,
        ykey="ChirpMass",
        xkeys=("HaloMass", "MStellar", "Z", "R50"),
        xlabels=("Halo Mass", "Stellar Mass", "Metallicity", "R50"),
        outname="Correlation_Summary_ChirpMass.png",
    )
    #To be updated
    plot_central_satellite(
    datasets, "HaloMass", "RemnantBHMass",
    xlabel=r"Host Halo Mass [$\rm{M_\odot}$]",
    ylabel=r"BH Remnant Mass [$\rm{M_\odot}$]",
    outname="Comparison_BHMass_HaloMass_CentralSatellite.png",
)



    

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    datasets = [load_and_clean(cat) for cat in CATALOGS]
    make_plots(datasets)
