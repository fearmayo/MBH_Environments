"""LISA AstroWG - MBH Environments - Catalog generation script - v2.0


#V2.0 introduced 6 new fields
#Stellar densities at R50 (half mass radius)
#V/sigma at R50
#Gas density at R50
#SFH
#Baryon Fraction
#Sersic index 

This script used the script written by Matteo Bonetti & Luke Zoltan Kelley as a blueprint. 

Catalog v1.0 is primarily meant to collect data from datasets on
(i) the environments surrounding MBH (numerical) mergers


Authors
-------
- Matteo Bonetti : matteo.bonetti@unimib.it
- Luke Zoltan Kelley : lzkelley@berkeley.edu
- John Regan: john.regan@mu.ie

- Structure
-------
This script has four main parts:
    - part A [MODIFY]       : functions that need input from users to collect information about the specific models.
                            Please carefully read the docstring of the function 'input_data()' that you find below.
    - part B [DO NOT MODIFY]: data structure for the hdf5 file
    - part C [DO NOT MODIFY]: functions to produce and validate hdf5 files. 
    - part D [DO NOT MODIFY]: main routine.

----
- Validation
    - Make sure numbers of elements all match
    - Even when not doing units checks, make sure values are sane (e.g. all positive, nonzero; etc) - DONE
    - Make sure HostGalaxyPosition only contains 'central' or 'satellite'  (Optional)
    - New (as of V2.0) host-galaxy fields (densities at R50, V/sigma, Sersic index, baryon fraction,
      stellar mass history): if present they are checked for shape, dtype and sane ranges.
      NaN is allowed in these fields and means "not computed for this binary", as was the standard in V1.0 and before.

---
- Simple usage:

python3 MBHEnvCatalogGenerator_<YourSimulationDataset>.py [-o <output_filename.hdf5>]  (or whatever works for your datasets)

If no output filename is given the catalog is written to
MBH_Environment_Catalog_<SIMULATION>.hdf5 in the current directory.

"""
############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################

# relevant packages, DO NOT REMOVE THEM
import argparse
import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import h5py
import numpy as np
import sys
import pickle
import glob

__VERSION__ = '2.0' # catalog version
DEBUG = False
np.random.seed(5)

############################################################################################
############################################################################################
############################################################################################
#### PART A: CHANGES NEEDED FROM USERS #####################################################
############################################################################################
############################################################################################
############################################################################################

# You'll need to change this to the name of your simulation / model
SIMULATION = "example-sim"

def input_data():
    '''
    This function collects the necessary information to produce the MBH Env catalog.

    Users should perform the following actions:
    1) edit the 'metadata' dictionary in this function to provide the specific information concerning a certain model;
    2) edit the 'get_binary_information' function in order to populate the numpy arrays with information from your datasets

    IMPORTANT: If a field does not apply to your specific model (e.g. spatial resolution for EPS SAMs), set it to 'np.nan'.

    NOTE: we require the number density of events. This is a meaningful quantity for EPS SAMs, for cosmological simulations
    and SAMs based on dark matter merger trees this is just 1/V_box 

    All parts that need to be modified by the user are enclosed into starred blocks like this:
    #************************************************#
    #************************************************#
    ...
    ...
    ...
    ...
    #************************************************#
    #************************************************#
        
    All the other functions should be left unchanged.

    Returns: 2 dictionaries -> metadata, mbhenv
    '''

    ###########################################################
    ####### CHANGE THE CODE BELOW #############################
    ###########################################################
    #*********************************************************#
    #*********************************************************#

    metadata = {
        # ---- Header data - identifying information for the dataset
        # REQUIRED
        'SimulationName': SIMULATION,
        'SimulationVersion': 'v1',
        'ModelType': 'Hydro',
        'Version': __VERSION__,
        'Date': str(datetime.datetime.now()),
        'Contributor': ["Your Name"],
        'Email': ["your.name@institution.edu"],
        'Principal': ["Your Name"],
        'Reference': ["DOI1", "DOI2"],
        # OPTIONAL:
        'Website': ["None"],

        # ---- Model parameters - metadata specification for simulation(s) used to construct catalog
        # REQUIRED
        'HubbleConstant': 70, #km s^-1 Mpc^-1
        'OmegaMatter': 0.3,
        'OmegaLambda': 0.7,
        'BoxSize': 1e0, # cMpc h^-1 (comoving)
        'MinBHSeedMass': 5e3, #M_sun
        'MinRedshift': 10,
        'MaxRedshift': 20,
        'StellarMassResolution': 1e3, # M_sun
        'DarkMatterMassResolution': 1e5, # M_sun
        'SpatialResolution': 5, # pc h^-1 (comoving)
        # OPTIONAL:
        'MinimumDarkMatterHaloMass': 1e6, # M_sun
        'GasMassResolution': 1e2, # M_sun

        # ---- Definitions behind the optional host-galaxy fields.
        # Only needed if you provide the densities at R50 / the gas number density.
        # The agreed defaults are shown; if you deviate, change the value here so the
        # catalog records what was actually done.
        'DensityShellInnerR50': 0.8,    # inner edge of the shell used for densities at R50 [units of R50]
        'DensityShellOuterR50': 1.2,    # outer edge of that shell [units of R50]
        'GasMeanMolecularWeight': 1.22, # mu in n = rho / (mu m_p) for the gas number density
    }
    #*********************************************************#
    #*********************************************************#

    ###########################################################
    #### DO NOT CHANGE FUNCTION CALLS AND RETURN ##############
    ###########################################################


    mbhenv = get_binary_information(metadata)


    return metadata, mbhenv

############################################################################################

def get_binary_information(metadata):

    '''
    Function to collect properties of binaries assuming no-delay models.
    In principle we can include delays as the MBHCatalogs work has done. 
    However, doing so would cause highly non-linear impacts to our SFR plots after merger
    and to the stellar distribution plots post-merger. For now, we therefore
    use only no-delay models and note that realism can only be improved if 
    the underlying MBH merger algorithm improves. 

    THE CODE BELOW PRODUCES FAKE BINARIES PROPERTIES, 
    PLEASE REPLACE IT WITH CUSTOMARY CODE TO COLLECT BINARIES FROM YOUR MODEL.

    Every field below must be a 1D array with one entry per binary (length N_binaries),
    except the stellar mass history, which is 2D (see below). A single scalar
    (e.g. galpos = "central") is NOT valid and will be rejected when the file is written.

    # Required fields
        N_binaries: number of binaries 
        
        #Black Hole Dictionary
        galid: Host Galaxy ID [None]
        m1: primary mass [M_sun]
        m2: secondary mass [M_sun]
        z: redshift of binary merger [None]
        sepa: separation at merger [proper kpc]
        W: number density [cMpc^-3] i.e. per comoving cubic Mpc 

        #Host Galaxy Dictionary
        galid: Host Galaxy ID [None] (sanity check. must match above ID)
        mstar: host galaxy stellar mass at merger or just after it [M_sun]
        zgal: redshift [None]
        R50: half-mass radius or effective radius [proper kpc]
        mhalo: Host halo mass at merger or just after [M_sun]
        galpos: Must be exactly 'central' or 'satellite' for every binary [string]
        metallicity: Mass-weighted gas metallicity of host galaxy at merger or just after [Z/Z_sun]

    # New fields as of V2.0
        Use np.nan for any binary where the quantity could not be computed (e.g. too few
        particles). Leave a field out entirely if you cannot provide it for any binary.
        All quantities are measured at the same epoch as mstar (merger or just after).

        baryon_fraction: (M_star + M_gas) / mhalo, with mhalo as above [None]
        stellar_density_r50: stellar mass density in a spherical shell from
            DensityShellInnerR50*R50 to DensityShellOuterR50*R50 (see metadata) [M_sun kpc^-3, proper]
        gas_density_r50: gas number density in the same shell, n = rho / (mu m_p) with
            mu = GasMeanMolecularWeight (see metadata) [cm^-3, proper]
        v_over_sigma_r50: stars within R50. v_rot is the mass-weighted mean rotational
            velocity about the angular-momentum axis of those stars; sigma is the 1D velocity
            dispersion of the residual velocities (3D dispersion / sqrt(3)). [None]
        sersic_index: Sersic index n from a fit to the stellar surface density profile;
            np.nan if the fit fails or there are too few particles [None]

        sfr{}: stellar mass history of the host galaxy (its main progenitor), as far back
            as you can follow it. Two 2D arrays of shape (N_binaries, N_snap):
              'Redshift'    : redshift of each snapshot [None]
              'StellarMass' : host stellar mass at that redshift [M_sun]

            Column 0 is the epoch of mstar/zgal (merger or just after); later columns go
            back in time, so redshift never decreases along a row. Rows with fewer than
            N_snap snapshots are padded with np.nan at the END of the row.
            Leave sfr = {} if you cannot provide this.

         e.g. sfr['Redshift']            sfr['StellarMass']  (M_sun)
            12.0  12.5  13.0  13.5     3e8   2e8   1e8   5e7     <- galaxy 0
            11.0  11.5  12.0  12.5     5e8   4e8   2e8   1e8     <- galaxy 1
            14.0  14.5   NaN   NaN     1e8   6e7   NaN   NaN     <- galaxy 2
    

    Returns: dictionary 
    '''
    ID = 0
    ###########################################################
    ####### CHANGE THE CODE BELOW #############################
    ###########################################################
    #*********************************************************#
    #*********************************************************#

    # generate fake binary properties
    N_binaries = 1000

    #Black Hole properties
    galid = np.arange(N_binaries)
    m1 = np.random.normal(loc=1e7, scale=1e6, size=N_binaries)
    m2 = np.random.normal(loc=1e7, scale=1e6, size=N_binaries)
    m1, m2 = np.max([m1, m2], axis=0), np.min([m1, m2], axis=0)
    z = 0.01 + np.random.uniform(0, 10, size=N_binaries)
    sepa = 10.0 ** np.random.uniform(2.0, 4.0, size=N_binaries)
    # number density
    W = 1/metadata["BoxSize"]**3*np.ones(len(m1))
    
    
    #Host galaxy properties

    # generate fake binary-host galaxy-remnant properties
    galid = galid
    mstar = np.random.normal(loc=1e10, scale=1e8, size=N_binaries)
    mhalo = mstar * np.maximum(1.0, np.random.normal(loc=10.0, scale=1.0, size=N_binaries))
    zgal = z - np.random.uniform(0, 0.001, size=N_binaries)
    R50 = np.random.normal(loc=3, scale=0.1, size=N_binaries)
    metallicity = 10.0 ** np.random.uniform(-4.0, 0.0, size=N_binaries)   # Z/Z_sun
    # one entry per binary, each exactly "central" or "satellite"
    galpos = np.random.choice(["central", "satellite"], size=N_binaries, p=[0.4, 0.6])

    # Optional host galaxy properties (FAKE values - replace with your own measurements)
    baryon_fraction = np.random.uniform(0.02, 0.157, size=N_binaries)
    stellar_density_r50 = 10.0 ** np.random.uniform(6.0, 9.0, size=N_binaries)     # M_sun kpc^-3
    gas_density_r50 = 10.0 ** np.random.uniform(-1.0, 3.0, size=N_binaries)        # cm^-3
    v_over_sigma_r50 = 10.0 ** np.random.normal(0.0, 0.3, size=N_binaries)
    sersic_index = np.clip(np.random.lognormal(np.log(1.5), 0.4, size=N_binaries), 0.5, 8.0)
    # example of the NaN convention: pretend 10% of the Sersic fits failed
    sersic_index[np.random.uniform(size=N_binaries) < 0.1] = np.nan

    # Stellar mass history (FAKE): N_snap snapshots spaced by delta_z = 0.25 going back in time,
    # with a random number of valid snapshots per binary and NaN padding at the end of each row
    N_snap = 20
    k = np.arange(N_snap)
    n_valid = np.random.randint(5, N_snap + 1, size=N_binaries)
    sfh_z = zgal[:, None] + 0.25 * k[None, :]
    sfh_mstar = mstar[:, None] * np.exp(-0.4 * k[None, :])
    pad = k[None, :] >= n_valid[:, None]
    sfh_z[pad] = np.nan
    sfh_mstar[pad] = np.nan
    sfr = {
        'Redshift': sfh_z,
        'StellarMass': sfh_mstar,
    }
   
    # FILL METADATA INFO
    # total number of merged binaries assuming no delays
    metadata['NumberBinaries'] = N_binaries

    # provide an explanation of the merger criterion, modify the string
    metadata['MergerCriteria'] = (
        "mergers occur when two MBH particles come within a gravitational softening length of "
        "eachother, and the kinetic energy of the pair is less than the gravitational "
        "potential energy between them.")
    
    metadata['Comments'] = (
        "Any additional information you consider relevant for any clarification, i.e."
        "model special features, recipe to deal with MBH evolution etc.")

    # describe how the optional quantities were actually measured (modify these strings)
    metadata['VoverSigmaDefinition'] = (
        "Fake example: stars within R50, v_rot about the stellar angular momentum axis, "
        "sigma = 3D dispersion / sqrt(3).")
    metadata['SersicFitDescription'] = (
        "Fake example: Sersic fit to the projected stellar surface density profile, "
        "NaN where the fit failed.")

    #**********************************************************#
    #**********************************************************#
    ############################################################
    # DO NOT CHANGE DICTIONARY AND RETURN ######################
    ############################################################

    # collect data in a dictionary
    mbhenv = {
        "BlackHoles": {
            'GalaxyID': galid,
            'PrimaryMass': m1,
            'SecondaryMass': m2,
            'Redshift': z,
            'Separation': sepa,
            'NumberDensity': W,
        },
        "HostGalaxy": {
            'GalaxyID': galid,
            'SFR': sfr,
            'HostGalaxyStellarMass': mstar,
            'HostGalaxyHaloMass': mhalo,
            'HostGalaxyRedshift': zgal,
            'HostGalaxyR50': R50,
            'HostGalaxyMetallicity': metallicity,
            'HostGalaxyPosition': galpos,
             # V2,0 fields (remove a line if you cannot provide that field at all)
            'HostGalaxyBaryonFraction': baryon_fraction,
            'HostGalaxyStellarDensityR50': stellar_density_r50,
            'HostGalaxyGasDensityR50': gas_density_r50,
            'HostGalaxyVoverSigmaR50': v_over_sigma_r50,
            'HostGalaxySersicIndex': sersic_index,
        }
    }

    return mbhenv

############################################################################################
#### PART B/C: HDF5 WRITING AND VALIDATION - DO NOT MODIFY #################################
############################################################################################

# V2.0 host-galaxy fields and the range of finite values each is allowed to take.
# field: (lower bound, upper bound, is the lower bound strict?)
OPTIONAL_GAL_FIELDS = {
    "HostGalaxyBaryonFraction":    (0.0, 1.0,    False),
    "HostGalaxyStellarDensityR50": (0.0, np.inf, True),
    "HostGalaxyGasDensityR50":     (0.0, np.inf, True),
    "HostGalaxyVoverSigmaR50":     (0.0, np.inf, False),
    "HostGalaxySersicIndex":       (0.0, np.inf, True),
}


def _prepare_for_hdf5(key, arr):
    """
    Convert an input array into something h5py can store safely.

      - scalars are rejected (every field must have one entry per binary)
      - None entries become np.nan
      - string arrays become fixed-width byte strings whose width is computed from the
        longest entry, so no string is ever truncated
    """
    arr_np = np.array(arr)

    if arr_np.ndim == 0:
        raise ValueError(
            f"Field '{key}' is a scalar. Every field must be an array with one entry per binary."
        )

    # ---------- FIX 1: Replace None with np.nan ----------
    if arr_np.dtype == object:
        arr_np = np.array([np.nan if x is None else x for x in arr_np])

    # ---------- FIX 2: Convert remaining object strings ----------
    if arr_np.dtype == object:
        if all(isinstance(x, str) for x in arr_np):
            max_len = max(len(x) for x in arr_np)
            arr_np = arr_np.astype(f'S{max_len}')
        else:
            raise TypeError(f"Cannot store field '{key}' with mixed dtype object.")

    # ---------- FIX 3: Convert Unicode strings ----------
    if arr_np.dtype.kind == 'U':  # Unicode dtype <U...
        arr_np = arr_np.astype('S')

    return arr_np


def write_catalog_hdf5(filename, metadata, mbhenv):
    """
    Write the MBH environment catalog to an HDF5 file.

    Parameters
    ----------
    filename : str
    metadata : dict
    mbhenv : dict with structure:
        mbhenv["BlackHoles"][field] = array
        mbhenv["HostGalaxy"][field] = array
        mbhenv["HostGalaxy"]["SFR"] = dict of 2D arrays (or an empty dict)
    """

    with h5py.File(filename, "w") as f:

        # ----------------------------------------------------
        # Metadata group
        # ----------------------------------------------------
        gmeta = f.create_group("Metadata")
        for key, value in metadata.items():
            # store strings as fixed-length UTF-8
            if isinstance(value, str):
                gmeta.attrs[key] = np.bytes_(value)
            elif isinstance(value, list):
                if all(isinstance(v, str) for v in value):
                    # convert lists-of-strings to variable-length strings
                    gmeta.attrs[key] = [np.bytes_(v) for v in value]
                else:
                    gmeta.attrs[key] = np.array(value)
            else:
                gmeta.attrs[key] = value

        # ----------------------------------------------------
        # Black hole binary information
        # ----------------------------------------------------
        gbh = f.create_group("Binaries")
        for key, arr in mbhenv["BlackHoles"].items():
            gbh.create_dataset(key, data=_prepare_for_hdf5(key, arr),
                               compression="gzip")

        # ----------------------------------------------------
        # Host galaxy information (robust string handling)
        # ----------------------------------------------------
        ggal = f.create_group("HostGalaxy")
        for key, arr in mbhenv["HostGalaxy"].items():

            # SFR is a subgroup holding the stellar mass history
            if key == "SFR":
                gsfr = ggal.create_group("SFR")
                for k2, arr2 in arr.items():
                    gsfr.create_dataset(k2, data=_prepare_for_hdf5("SFR/" + k2, arr2),
                                        compression="gzip")
                continue

            ggal.create_dataset(key, data=_prepare_for_hdf5(key, arr),
                                compression="gzip")

    print(f"[✓] Wrote catalog to {filename}")


def _check_numeric_dataset(name, arr, lo=0.0, hi=np.inf, strict=False):
    """Shared checks for optional numeric datasets. NaN is allowed, inf is not."""
    if arr.dtype.kind not in "fiu":
        raise TypeError(f"[Validator] {name} must be numeric, found dtype {arr.dtype}")
    if np.isinf(arr).any():
        raise ValueError(f"[Validator] {name} contains inf")
    finite = arr[np.isfinite(arr)]
    too_low = (finite <= lo) if strict else (finite < lo)
    outside = too_low | (finite > hi)
    if outside.any():
        raise ValueError(
            f"[Validator] {name} has {int(outside.sum())} values outside the allowed range "
            f"({'>' if strict else '>='} {lo}" + (f", <= {hi}" if np.isfinite(hi) else "") + ")"
        )
    return finite.size


def _validate_sfh(gal, N):
    """Validate the optional stellar mass history stored in HostGalaxy/SFR."""
    if "SFR" not in gal:
        print("[Validator] Optional stellar mass history (HostGalaxy/SFR) not provided")
        return

    sfr = gal["SFR"]
    if not isinstance(sfr, h5py.Group):
        raise TypeError("[Validator] HostGalaxy/SFR must be a group of datasets")
    if len(sfr) == 0:
        print("[Validator] Optional stellar mass history (HostGalaxy/SFR) not provided")
        return

    for key in ("Redshift", "StellarMass"):
        if key not in sfr:
            raise ValueError(f"[Validator] HostGalaxy/SFR is provided but SFR/{key} is missing")

    z = sfr["Redshift"][:]
    if z.ndim != 2 or z.shape[0] != N:
        raise ValueError(
            f"[Validator] SFR/Redshift must have shape (N_binaries={N}, N_snap), found {z.shape}"
        )

    # every provided dataset has the same shape, is numeric, non-negative and finite-or-NaN
    for key in ("Redshift", "StellarMass"):
        if key not in sfr:
            continue
        arr = sfr[key][:]
        if arr.shape != z.shape:
            raise ValueError(f"[Validator] SFR/{key} has shape {arr.shape}, expected {z.shape}")
        _check_numeric_dataset(f"SFR/{key}", arr, lo=0.0)

    # Redshift and StellarMass must be valid in exactly the same places
    m = sfr["StellarMass"][:]
    if not np.array_equal(np.isnan(z), np.isnan(m)):
        raise ValueError("[Validator] SFR/Redshift and SFR/StellarMass must have NaNs in the same places")

    # NaN padding only at the END of each row, and redshift never decreases along a row
    finite = np.isfinite(z)
    n_valid = finite.sum(axis=1)
    expected = np.arange(z.shape[1])[None, :] < n_valid[:, None]
    if not np.array_equal(finite, expected):
        raise ValueError("[Validator] SFR arrays must be padded with NaN only at the end of each row")
    if np.any(np.diff(np.where(finite, z, np.nan), axis=1) < 0):
        raise ValueError(
            "[Validator] SFR/Redshift must not decrease along a row (column 0 = merger epoch, "
            "later columns go back in time)"
        )

    print("[Validator] Stellar mass history OK (shape %s, %d/%d binaries have a history)"
          % (z.shape, int((n_valid > 0).sum()), N))


def validate_catalog(filename):
    """
    Validate the HDF5 catalog written by write_catalog_hdf5().
    Ensures:
      - Required groups exist
      - Required datasets exist
      - Arrays have consistent lengths
      - No invalid dtypes (e.g. object)
      - HostGalaxyPosition only contains 'central' or 'satellite'
      - No NaNs in required fields
      - Optional fields, when present, have the right shape, dtype and sane values
    """

    required_groups = ["Metadata", "Binaries", "HostGalaxy"]

    required_bh_fields = [
        "GalaxyID",
        "PrimaryMass",
        "SecondaryMass",
        "Redshift",
        "Separation",
        "NumberDensity",
    ]

    required_gal_fields = [
        "GalaxyID",
        "HostGalaxyStellarMass",
        "HostGalaxyHaloMass",
        "HostGalaxyMetallicity",
        "HostGalaxyR50",
        "HostGalaxyRedshift",
        "HostGalaxyPosition",
    ]

    print("\n[Validator] Validating catalog:", filename)

    with h5py.File(filename, "r") as f:

        # ---- Check groups exist ----
        for g in required_groups:
            if g not in f:
                raise ValueError(f"[Validator] Missing group: {g}")
        print("[Validator] Groups OK")

        # ---- Check binary fields ----
        bh = f["Binaries"]
        for key in required_bh_fields:
            if key not in bh:
                raise ValueError(f"[Validator] Missing Binaries/{key}")
        print("[Validator] Binaries fields OK")

        # ---- Check host galaxy fields ----
        gal = f["HostGalaxy"]
        for key in required_gal_fields:
            if key not in gal:
                raise ValueError(f"[Validator] Missing HostGalaxy/{key}")
        print("[Validator] HostGalaxy fields OK")

        # ---- Check array lengths match ----
        N = len(bh["GalaxyID"])
        for key in required_bh_fields:
            if len(bh[key]) != N:
                raise ValueError(f"[Validator] Binaries/{key} wrong length")

        for key in required_gal_fields:
            if len(gal[key]) != N:
                raise ValueError(f"[Validator] HostGalaxy/{key} wrong length")
        print("[Validator] Field lengths OK (N = %d)" % N)

        # ---- Check no object dtypes ----
        for group, keys in [("Binaries", required_bh_fields),
                            ("HostGalaxy", required_gal_fields)]:
            for key in keys:
                arr = f[group][key][:]
                if arr.dtype == object:
                    raise TypeError(f"[Validator] {group}/{key} has object dtype")

        print("[Validator] Datatypes OK")

        # ---- Check HostGalaxyPosition labels ----
        labels = {p.decode("utf-8") if isinstance(p, bytes) else str(p)
                  for p in gal["HostGalaxyPosition"][:]}
        bad = sorted(labels - {"central", "satellite"})
        if bad:
            raise ValueError(
                f"[Validator] HostGalaxyPosition must be 'central' or 'satellite', found: {bad}"
            )
        print("[Validator] HostGalaxyPosition labels OK")

        # ---- Check no NaNs in required numeric fields ----
        numeric_bh_fields = ["PrimaryMass", "Redshift", "NumberDensity"]
        numeric_gal_fields = ["HostGalaxyStellarMass", "HostGalaxyHaloMass",
                              "HostGalaxyMetallicity", "HostGalaxyR50"]

        for key in numeric_bh_fields:
            if np.isnan(bh[key][:]).any():
                raise ValueError(f"[Validator] NaNs in Binaries/{key}")

        for key in numeric_gal_fields:
            if np.isnan(gal[key][:]).any():
                raise ValueError(f"[Validator] NaNs in HostGalaxy/{key}")

        print("[Validator] No NaNs in required fields")

        # ---- V2.0 host galaxy fields (NaN allowed = not computed) ----
        for key, (lo, hi, strict) in OPTIONAL_GAL_FIELDS.items():
            if key not in gal:
                print(f"[Validator] Optional HostGalaxy/{key} not provided")
                continue
            arr = gal[key][:]
            if arr.ndim != 1 or len(arr) != N:
                raise ValueError(f"[Validator] HostGalaxy/{key} must be 1D with length {N}")
            n_finite = _check_numeric_dataset(f"HostGalaxy/{key}", arr, lo=lo, hi=hi, strict=strict)
            print(f"[Validator] Optional HostGalaxy/{key} OK ({n_finite}/{N} finite)")

        # ---- Optional stellar mass history ----
        _validate_sfh(gal, N)

    print("[Validator] ✓ Catalog validation PASSED\n")

############################################################################################
#### PART D: MAIN ROUTINE - DO NOT MODIFY ##################################################
############################################################################################

def main():
    parser = argparse.ArgumentParser(description="Generate the MBH Environments catalog.")
    parser.add_argument(
        "-o", "--output",
        default="MBH_Environment_Catalog_%s.hdf5" % (SIMULATION),
        help="Output HDF5 filename (default: MBH_Environment_Catalog_<SIMULATION>.hdf5)",
    )
    args = parser.parse_args()

    print("Generating MBH Environment Catalog for %s..." % (SIMULATION))

    metadata, mbhenv = input_data()

    write_catalog_hdf5(args.output, metadata, mbhenv)
    validate_catalog(args.output)

if __name__ == "__main__":
    main()
