"""LISA AstroWG - MBH Environments - Catalog generation script - v1.0

This script used the script written by Matteo Bonetti & Luke Zoltan Kelley as a blueprint. 

Catalog v1.0 is primarily meant to collect data from datasets on
(i) the environments surrounding MBH (numerical) mergers


(i) information on binary MBHs for “no delay” cases that will be used to create a uniform “delay” case for all models, 
(ii) information on the full MBH population that will be used to build AGN luminosity functions and MBH-Mgal relations to compare to observations. 
A group can also provide a “delay” case that will be used for the paper, but this can also be provided at later time. 
There are some additional data, such as dark matter halo information, that can be optionally provided for v2.1 if desired.  

Links
-----
- Catalog v2.1 spreadsheet:
  https://docs.google.com/spreadsheets/d/1Zz-A5QcKHdPabpSmkKcs9ERzeAA0sqM_/edit?usp=sharing&ouid=113068423500393235248&rtpof=true&sd=true
- MBHCatalogs github for code:
  https://github.com/mbonetti90/MBHCatalogs
- MBHCatalogs google drive for simulation data:
  https://drive.google.com/drive/folders/1rW-cdOrMglfp2w72X0jmdu1G5COb7RDk?usp=sharing
- MBHCatalogs general folder
  https://drive.google.com/drive/folders/1YQYw-Km-N5b5jjeHYQ4Ls6LTbXxKh9VC?usp=sharing
Authors
-------
- Matteo Bonetti : matteo.bonetti@unimib.it
- Luke Zoltan Kelley : lzkelley@berkeley.edu

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

---
- Simple usage:

python3 hdf5_catalog_v2.1.py -W -F the_name_of_the_file_to_be_produced

type -h for immediate help


"""
############################################################################################
############################################################################################
############################################################################################
############################################################################################
############################################################################################

# relevant packages, DO NOT REMOVE THEM
import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import h5py
import numpy as np
import argparse
import sys

__VERSION__ = '1.0' # catalog version
DEBUG = False
DEF_FILENAME = "example-sim"
np.random.seed(5)

############################################################################################
############################################################################################
############################################################################################
#### PART A: CHANGES NEEDED FROM USERS #####################################################
############################################################################################
############################################################################################
############################################################################################

def input_data():
    '''
    This function collects the necessary information to produce the MBH Env catalog.

    Users should perform the following actions:
    1) edit the 'metadata' dictionary in this function to provide with the specific information concerning a certain model;
    2) edit the 'MBHB_no_delay' function to provide with the no-delay binary catalog;
    3) [optional] edit the 'MBHB_delay' function to provide with the delay binary catalog. Leave an empty dict if the delay model is not present;
    4) edit the MBH_population function to provide with the complete MBH population at selected redshifts.

    IMPORTANT: If a field does not apply to your specific model (e.g. spatial resolution for EPS SAMs), set it to 'np.nan'.

    NOTE: we require the number density of events. This is a meaninful quantity for EPS SAMs, for cosmological simulations
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

    Returns: 4 dictionaries -> metadata, allbhs, binaries_nodelay, binaries_delay
    '''

    ###########################################################
    ####### CHANGE THE CODE BELOW #############################
    ###########################################################
    #*********************************************************#
    #*********************************************************#

    metadata = {
        # ---- Header data - identifying information for the dataset
        # REQUIRED
        'SimulationName': 'SEEDZ',
        'SimulationVersion': 'LowRes',
        'ModelType': 'Hydro',
        'Version': __VERSION__,
        'Date': str(datetime.datetime.now()),
        'Contributor': ["John Regan", "Daxal Mehta", "Lewis Prole"],
        'Email': ["john.regan@mu.ie"],
        'Principal': ["John Regan"],
        'Reference': ["DOI1", "DOI2"],
        # OPTIONAL:
        'Website': ["https://www.github.com/mbonetti90/MBHCatalogs"],

        # ---- Model parameters - metadata specification for simulation(s) used to construct catalog
        # REQUIRED
        'HubbleConstant': 70, #km s^-1 Mpc^-1
        'OmegaMatter': 0.3,
        'OmegaLambda': 0.7,
        'BoxSize': 1e0, # cMpc
        'MinBHSeedMass': 5e3, #M_sun
        'MinRedshift': 10,
        'MaxRedshift': 20,
        'StellarMassResolution': 1e3, # M_sun
        'DarkMatterMassResolution': 1e5, # M_sun
        'SpatialResolution': np.nan, # kpc
        # OPTIONAL:
        'MinimumDarkMatterHaloMass': 1e6, # M_sun
        'GasMassResolution': 1e2, # M_sun
    }
    #*********************************************************#
    #*********************************************************#

    ###########################################################
    #### DO NOT CHANGE FUNCTION CALLS AND RETURN ##############
    ###########################################################


    mbhenv = get_binary_information(metadata)


    
    # REQUIRED
    # ---- Binary mergers at time of galaxy merger (no-delay) ----

    binaries_nodelay = MBHB_no_delay(metadata)


    # OPTIONAL (can be empty dictionary):
    # ---- Binary mergers including a model of delay times ----

    binaries_delay = MBHB_delay(metadata)


    # REQUIRED:
    # ---- All black holes and host galaxies at target redshifts

    allbhs = MBH_population(metadata)

    #########################

    return metadata, allbhs, binaries_nodelay, binaries_delay

############################################################################################

def MBHB_no_delay(metadata):

    '''
    Function to collect properties of binaries assuming no-delay models.

    THE CODE BELOW PRODUCES FAKE BINARIES PROPERTIES, 
    PLEASE REPLACE IT WITH CUSTOMARY CODE TO COLLECT BINARIES FROM YOUR MODEL.

    # Required fields
        N_binaries: number of binaries 
        m1: primary mass [M_sun]
        m2: secondary mass [M_sun]
        z: redshift of binary merger [None]
        sepa: separation at merger [proper kpc]
        W: number density [Mpc^-3]
        mstar: host galaxy stellar mass at merger or just after it [M_sun]
        zgal: redshift of galaxy stellar mass [None]
        R50: half-mass radius or effective radius [proper kpc]
    
    # optional fields
        mdm: dark matter mass at merger or just after it [M_sun]

    Returns: dictionary 
    '''

    ###########################################################
    ####### CHANGE THE CODE BELOW #############################
    ###########################################################
    #*********************************************************#
    #*********************************************************#

    # generate fake binary properties
    N_binaries = 1000
    m1 = np.random.normal(loc=1e7, scale=1e6, size=N_binaries)
    m2 = np.random.normal(loc=1e7, scale=1e6, size=N_binaries)
    m1, m2 = np.max([m1, m2], axis=0), np.min([m1, m2], axis=0)
    z = 0.01 + np.random.uniform(0, 10, size=N_binaries)
    sepa = 10.0 ** np.random.uniform(2.0, 4.0, size=N_binaries)

    # generate fake binary-host galaxy-remnant properties
    mstar = np.random.normal(loc=1e10, scale=1e8, size=N_binaries)
    mdm = mstar * np.maximum(1.0, np.random.normal(loc=10.0, scale=1.0, size=N_binaries))
    zgal = z - np.random.uniform(0, 0.001, size=N_binaries)
    R50 = np.random.normal(loc=3, scale=0.1, size=N_binaries)

    # number density
    W = 1/metadata["BoxSize"]**3*np.ones(len(m1))

    # FILL METADATA INFO
    # total number of merged binaries assuming no delays
    metadata['NumberBinaries'] = N_binaries

    # provide an explanation of the merger criterion, modify the string
    metadata['NoDelay'] = (
        "mergers occur when two MBH particles come within a gravitational softening length of "
        "eachother, and the kinetic energy of the pair is less than the gravitational "
        "potential energy between them.")
    
    metadata['CommentsNoDelay'] = (
        "Any additional information you consider relevant for any clarification, i.e."
        "model special features, recipe to deal with MBH evolution etc.")

    #**********************************************************#
    #**********************************************************#
    ############################################################
    # DO NOT CHANGE DICTIONARY AND RETURN ######################
    ############################################################

    # collect data in a dictionary
    binaries_nodelay = {
        "BlackHoles": {
            'PrimaryMass': m1,
            'SecondaryMass': m2,
            'Redshift': z,
            'Separation': sepa,
            'NumberDensity': W,
        },
        "Galaxies": {
            'RemnantStellarMass': mstar,
            'RemnantRedshift': zgal,
            'RemnantR50': R50,
            # OPTIONAL:
            'RemnantDarkMatterMass': mdm,
        }
    }

    return binaries_nodelay

#########################
def MBHB_delay(metadata):

    '''
    Function to collect properties of binaries assuming delay models.

    THE CODE BELOW PRODUCES FAKE BINARIES PROPERTIES, 
    PLEASE REPLACE IT WITH CUSTOMARY CODE TO COLLECT BINARIES FROM YOUR MODEL.
    IF YOU DO NOT PROVIDE ANY DELAY MODEL JUST SET 'delay_model' TO 'False' AND DELETE THE CODE, AN EMPTY DICT WILL BE SET.

    Returns:  
    # Required fields
        N_binaries_delay: number of binaries 
        m1_delay: primary mass [M_sun]
        m2_delay: secondary mass [M_sun]
        z_delay: redshift of binary merger [None]
        sepa_delay: separation at merger [proper kpc]
        W: number density [Mpc^-3]
        mstar_delay: host galaxy stellar mass at merger or just after it [M_sun]
        zgal_delay: redshift of galaxy stellar mass [None]
        R50: half-mass radius or effective radius [proper kpc]
    
    # optional fields
        mdm_delay: dark matter mass at merger or just after it [M_sun]
    '''

    ###########################################################
    ####### CHANGE THE CODE BELOW #############################
    ###########################################################
    #*********************************************************#
    #*********************************************************#

    # set to true if data for a model with delays are available
    delay_model = True

    # generate fake binary properties
    N_binaries = 1000
    m1 = np.random.normal(loc=1e7, scale=1e6, size=N_binaries)
    m2 = np.random.normal(loc=1e7, scale=1e6, size=N_binaries)
    m1, m2 = np.max([m1, m2], axis=0), np.min([m1, m2], axis=0)
    z = 0.01 + np.random.uniform(0, 10, size=N_binaries)
    sepa = 10.0 ** np.random.uniform(2.0, 4.0, size=N_binaries)

    # generate fake binary-host galaxy-remnant properties
    mstar = np.random.normal(loc=1e10, scale=1e8, size=N_binaries)
    mdm = mstar * np.maximum(1.0, np.random.normal(loc=10.0, scale=1.0, size=N_binaries))
    zgal = z - np.random.uniform(0, 0.001, size=N_binaries)
    R50 = np.random.normal(loc=3, scale=1, size=N_binaries)

    # choose some amount of delay in redshift
    delta_redz = np.random.uniform(0.01, 1.0, N_binaries)
    z_delay = z - delta_redz
    zgal_delay = zgal - delta_redz
    # valid binaries must coalesce before redshift zero
    sel_valid = (z_delay > 0.0) & (zgal_delay > 0.0)
    m1_delay = m1[sel_valid]
    m2_delay = m2[sel_valid]
    z_delay = z_delay[sel_valid]
    zgal_delay = zgal_delay[sel_valid]
    sepa_delay = sepa[sel_valid] * (10.0 ** np.random.normal(loc=0.02, scale=0.05))
    # make sure delayed separation is smaller than no-delay separation
    sepa_delay = np.minimum(sepa_delay, sepa[sel_valid]*0.98)

    # find number of coalesced binaries in the delay model
    N_binaries_delay = int(m1_delay.size)

    # define new arrays with delayed binaries
    m1_delay = m1_delay * (1.0 + np.random.normal(loc=0.1, scale=0.1, size=N_binaries_delay))
    m2_delay = m2_delay * (1.0 + np.random.normal(loc=0.2, scale=0.1, size=N_binaries_delay))
    mstar_delay = mstar[sel_valid] * (1.0 + np.random.normal(loc=0.15, scale=0.1, size=N_binaries_delay))
    mdm_delay = mstar[sel_valid] * (1.0 + np.random.normal(loc=0.15, scale=0.1, size=N_binaries_delay))
    # primary and secondary may have switched
    m1_delay, m2_delay = np.max([m1_delay, m2_delay], axis=0), np.min([m1_delay, m2_delay], axis=0)

    # number density
    W = 1/metadata["BoxSize"]**3*np.ones(len(m1_delay))

    # FILL METADATA INFO
    # total number of merged binaries assuming delays
    metadata["NumberBinariesDelay"] = N_binaries_delay

    # provide an explanation of the merger criterion, modify the string
    metadata["Delay"] = (
        "Binaries are evolved in post-processing using a semi-analytic model for dynamical "
        "friction.  Binaries are considered to have merged once they become gravitationally bound."
    )

    metadata['CommentsDelay'] = (
        "Any additional information you consider relevant for any clarification, i.e."
        "model special features, recipe to deal with MBH evolution etc.")

    #**********************************************************#
    #**********************************************************#
    ############################################################
    # DO NOT CHANGE DICTIONARY AND RETURN ######################
    ############################################################

    if delay_model:
        # collect data in a dictionary
        binaries_delay = {
            "BlackHoles": {
                'PrimaryMass': m1_delay,
                'SecondaryMass': m2_delay,
                'Redshift': z_delay,
                'Separation': sepa_delay,
                'NumberDensity': W,
            },
            "Galaxies": {
                'RemnantStellarMass': mstar_delay,
                'RemnantRedshift': zgal_delay,
                'RemnantR50': np.ones(len(R50))*np.nan,
                # OPTIONAL:
                'RemnantDarkMatterMass': mdm_delay,
            }
        }

    # overwrite with an emtpy dict if no model with delays is present
    else:
        binaries_delay = {}

    return binaries_delay

#########################
def MBH_population(metadata):

    '''
    Function to collect properties of all BHs at selected redshift [TARGET_REDSHIFTS array].
    See also https://docs.google.com/spreadsheets/d/1Zz-A5QcKHdPabpSmkKcs9ERzeAA0sqM_/edit#gid=1167697008

    THE CODE BELOW PRODUCES FAKE BHs PROPERTIES, 
    PLEASE REPLACE IT WITH CUSTOMARY CODE TO COLLECT BHs FROM YOUR MODEL.

    # Required fields at selected target redshifts:
        mass: BH mass [M_sun]
        mdot: accretion rate [M_sun/yr]
        radeff: radiative efficiency [-]
        W: number density [Mpc^-3]
        mstar: host galaxy stellar mass [M_sun]
        mdm: dark matter mass at merger [M_sun]


    Returns: dictionary 
    '''

    # data structure to contain collected data
    allbhs = {}
    allbh_redshifts = []
    num_allbhs = 0

    ###########################################################
    ####### CHANGE THE CODE BELOW #############################
    ###########################################################
    #*********************************************************#
    #*********************************************************#
    
    min_redshift = np.inf
    for tarz in TARGET_REDSHIFTS:
        # simulation snapshots may not exactly match target redshifts
        myz = tarz + np.random.uniform(-0.01, +0.01)
        # store actual simulation redshifts
        allbh_redshifts.append(myz)
        myz = np.maximum(myz, 1.0e-4)
        # generate a key for this redshift (i.e. string specification in consistent format)
        zkey = f"Z{myz:011.8f}"
        # number of BHs in this snapshot
        numbhs = np.random.randint(100, 200)
        # store the number of BHs in the lowest-redshift snapshot being stored
        if myz < min_redshift:
            min_redshift = myz
            num_allbhs = numbhs

        # generate fake BH data
        mass = 10.0 ** np.random.uniform(6, 10, numbhs)
        mdot = 10.0 ** np.random.normal(-3.0, 1.0, numbhs)
        radEff = 0.1*np.ones(len(mdot)) # example with fixed rad efficiency for all BHs
        # generate fake galaxy data
        mstar = mass * np.random.normal(3e2, 10.0, numbhs)
        mdm = mstar * np.random.normal(8.0, 1.0, numbhs)

        # number density
        W = 1/metadata["BoxSize"]**3*np.ones(len(mass))

        # put data for this redshift/snapshot into nested dictionaries
        temp_data = {
            "BlackHoles": {
                "Mass": mass,
                "Mdot": mdot,
                "RadEff": radEff,
                "NumberDensity": W,
            },
            "Galaxies": {
                "StellarMass": mstar,
                "DarkMatterMass": mdm,
            },
        }
        # add to combined allbhs data
        allbhs[zkey] = temp_data

    metadata['CommentsAllBHs'] = (
        "Any additional information you consider relevant for any clarification, i.e."
        "model special features, recipe to deal with MBH evolution etc.")

    #*********************************************************#
    #*********************************************************#
    ###########################################################
    # DO NOT CHANGE METADATA ASSIGNMENT AND RETURN ############
    ###########################################################

    metadata['NumberAllBHs'] = num_allbhs
    metadata['RedshiftsAllBHs'] = allbh_redshifts

    return allbhs


#########################
def get_catalog_information(metadata):

    '''
    Function to collect properties of all BHs at selected redshift [TARGET_REDSHIFTS array].
    See also https://docs.google.com/spreadsheets/d/1Zz-A5QcKHdPabpSmkKcs9ERzeAA0sqM_/edit#gid=1167697008

    THE CODE BELOW PRODUCES FAKE BHs PROPERTIES, 
    PLEASE REPLACE IT WITH CUSTOMARY CODE TO COLLECT BHs FROM YOUR MODEL.

    # Required fields at selected target redshifts:
        mass: BH mass [M_sun]
        mdot: accretion rate [M_sun/yr]
        radeff: radiative efficiency [-]
        W: number density [Mpc^-3]
        mstar: host galaxy stellar mass [M_sun]
        mdm: dark matter mass at merger [M_sun]


    Returns: dictionary 
    '''

    # data structure to contain collected data
    allbhs = {}
    allbh_redshifts = []
    num_allbhs = 0

    ###########################################################
    ####### CHANGE THE CODE BELOW #############################
    ###########################################################
    #*********************************************************#
    #*********************************************************#
    
    min_redshift = np.inf
    for tarz in TARGET_REDSHIFTS:
        # simulation snapshots may not exactly match target redshifts
        myz = tarz + np.random.uniform(-0.01, +0.01)
        # store actual simulation redshifts
        allbh_redshifts.append(myz)
        myz = np.maximum(myz, 1.0e-4)
        # generate a key for this redshift (i.e. string specification in consistent format)
        zkey = f"Z{myz:011.8f}"
        # number of BHs in this snapshot
        numbhs = np.random.randint(100, 200)
        # store the number of BHs in the lowest-redshift snapshot being stored
        if myz < min_redshift:
            min_redshift = myz
            num_allbhs = numbhs

        # generate fake BH data
        mass = 10.0 ** np.random.uniform(6, 10, numbhs)
        mdot = 10.0 ** np.random.normal(-3.0, 1.0, numbhs)
        radEff = 0.1*np.ones(len(mdot)) # example with fixed rad efficiency for all BHs
        # generate fake galaxy data
        mstar = mass * np.random.normal(3e2, 10.0, numbhs)
        mdm = mstar * np.random.normal(8.0, 1.0, numbhs)

        # number density
        W = 1/metadata["BoxSize"]**3*np.ones(len(mass))

        # put data for this redshift/snapshot into nested dictionaries
        temp_data = {
            "BlackHoles": {
                "Mass": mass,
                "Mdot": mdot,
                "RadEff": radEff,
                "NumberDensity": W,
            },
            "Galaxies": {
                "StellarMass": mstar,
                "DarkMatterMass": mdm,
            },
        }
        # add to combined allbhs data
        allbhs[zkey] = temp_data

    metadata['CommentsAllBHs'] = (
        "Any additional information you consider relevant for any clarification, i.e."
        "model special features, recipe to deal with MBH evolution etc.")

    #*********************************************************#
    #*********************************************************#
    ###########################################################
    # DO NOT CHANGE METADATA ASSIGNMENT AND RETURN ############
    ###########################################################

    metadata['NumberAllBHs'] = num_allbhs
    metadata['RedshiftsAllBHs'] = allbh_redshifts

    return allbhs
