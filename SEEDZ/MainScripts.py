import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.image as mpimg
import yt
from mpi4py import MPI
import pickle
import os
from scipy.spatial import KDTree
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=67.7, Om0=0.2592, Tcmb0=2.725)


import YT
import SinkParticles
import Utilities
import Plotter
import DataReader
import GroupsReader

CS = Utilities.Constants()
DU = Utilities.DataUtilities()

color_schemes = {
    "Okabe–Ito": ['#0072B2', '#E69F00', '#009E73', '#D55E00'],
    "Tol Muted": ['#88CCEE', '#DDCC77', '#117733', '#CC6677'],
    "Material": ['#673AB7', '#FFC107', '#009688', '#E53935'],
    "Categorical": ['#1F77B4', '#FF7F0E', '#2CA02C', '#D62728'],
}


class Scripts:
    def __init__(self, base, output_dir, feedback=False, array_length=1):
        self.base = base
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.start_snap, self.end_snap = DU.file_range(base)
        self.feedback = feedback
        self.array_length = array_length
        self.BinaryReader = DataReader.BinaryReader(self.base, feedback=self.feedback, array_length=self.array_length)
        self.PickleReader = DataReader.Reader(self.base)

        print(f"We are loading data from {self.base}, for which accretion feedback is {self.feedback}. The outputs of the functions would be stored at {self.output_dir}.")
  
    def GasProjectionCenter(self, snap):
        print("Entering GasProjectionCenter function.")
        #sfile = "/home/lewis/data/SEEDZ_data/production/Rarepeak_thermal/snapdir_035/snap_035.0.hdf5"
        sfile = f"{self.base}snapdir_{snap:03d}/snap_{snap:03d}.0.hdf5"
        
        ds = yt.load(sfile)
        ad = ds.all_data()

        projector = YT.ProjectionPlot(ds, ad)
        projector.add_field(("gas", "temperature"))
        
        #projector.set_center([20221.03268839, 19792.76559769, 19740.91547032], 6000, units="code_length")
        projector.set_center(width=1000, units="code_length")
        projector.plot()
        projector.customize(cmap="magma", beauty=True)
        projector.save_plots(base=self.output_dir, prefix="PlotCenter")
    
    def GasProjectionSink(self, snap, Pos, Rad=1):
        print("Entering GasProjectionSink function")

        self.BinaryReader.read_sink_snap(snap)
        id_binary = self.BinaryReader.extract_data("ID")
        type_binary = self.BinaryReader.extract_data("Type")

        sfile = f"{self.base}snapdir_{snap:03d}/snap_{snap:03d}.0.hdf5"
        ds = yt.load(sfile)
        ad = ds.all_data()
        Groups = GroupsReader.Reader(self.base, snap)
        Subhalos = Groups.subhalos
        SubhaloPos = Subhalos["SubhaloCM"]
        SubhaloRad = Subhalos["SubhaloHalfmassRad"]

        projector = YT.ProjectionPlot(ds, ad)
        projector.add_field(("gas", "number_density"))
            
        projector.set_center(Pos, Rad * 500, units="code_length")
        projector.plot()
        projector.customize(cmap="viridis")
        projector.plot_sink_particles(id_binary, type_binary)
        for i in range(len(SubhaloRad)):
            projector.plot_halo(SubhaloPos[i], SubhaloRad[i], units="code_length")
        projector.save_plots(base=self.output_dir, prefix=f"SinkCenter_{np.random.randint(1, 1000)}")

    def GasProjectionID(self, snap, ID=None):
        print("Entering GasProjectionID.")
        ID, x, y, z, HeavySeedMass, StellarMass = np.genfromtxt("HeavySeedLoc.txt", usecols=(0, 1, 2, 3, 4, 5), skip_header=1, unpack=True)
        ind = HeavySeedMass == np.max(HeavySeedMass)
        ID = ID[ind]
        ID = 96918313
        sfile = f"{self.base}snapdir_{snap:03d}/snap_{snap:03d}.0.hdf5"
        ds = yt.load(sfile)
        ad = ds.all_data()

        self.BinaryReader.read_sink_snap(snap)
        id_binary = self.BinaryReader.extract_data("ID")
        if ID not in id_binary: return
        type_binary = self.BinaryReader.extract_data("Type")
        pos_binary = self.BinaryReader.extract_data("Pos")
        mass_binary = self.BinaryReader.extract_data("StellarMass") * 1e10 / CS.hubble_parameter
        ind = ID == id_binary

        HeavySeedPos = pos_binary[ind][0].flatten()
        print(HeavySeedPos)
        HeavySeedMass = DU.unwrap(mass_binary[ind])


        projector = YT.ProjectionPlot(ds, ad)
        projector.add_field(("gas", "number_density"))
        projector.set_center(HeavySeedPos, 1 * 5, units="code_length")
        projector.plot()
        projector.customize(cmap="viridis")
        projector.plot_sink_particles(id_binary, type_binary)
        projector.annotate_text(0.1, 0.9, "$\mathrm{M}_{BH} = %.2e$" %HeavySeedMass)
        projector.annotate_text(0.1, 0.85, f"Redshift = {ds.current_redshift}")
        projector.save_plots(prefix=f"IDCenter_{snap}")

        phase = YT.PhasePlot(ds, ad)
        phase.set_center(HeavySeedPos, 1, units="code_length")
        phase.plot()
        phase.save_plots(base=self.output_dir, prefix=f"IDPhase_{snap}")
        
    def GasProjectionHalo(self, snap, field="number_density", beauty=False):
        print("Entering GasProjectionHalo function.")

        cmap, lowlim, highlim, beauty = "viridis", 1e-2, 1e5, beauty
        if field == "temperature":
            cmap, lowlim, highlim, beauty = "magma", 1e2, 5e5, beauty

        Groups = GroupsReader.Reader(self.base, snap)
        Halos = Groups.halos
        HaloPos = Halos["GroupPos"]
        HaloMass = Halos["GroupMass"] * 1e10 / CS.hubble_parameter
        HaloRad = Halos["Group_R_TopHat200"]
        HaloMassType = Halos["GroupMassType"] * 1e10 / CS.hubble_parameter

        Subhalos = Groups.subhalos
        SubhaloPos = Subhalos["SubhaloCM"]
        SubhaloRad = Subhalos["SubhaloHalfmassRad"]
        
        self.BinaryReader = DataReader.BinaryReader(self.base, feedback=self.feedback, array_length=self.array_length)
        self.BinaryReader.read_sink_snap(snap)
        id_binary = self.BinaryReader.extract_data("ID")
        type_binary = self.BinaryReader.extract_data("Type")
        
        
        sfile = f"{self.base}snapdir_{snap:03d}/snap_{snap:03d}.0.hdf5"
        ds = yt.load(sfile)
        ad = ds.all_data()

        #Pos = [20007.16000627, 19964.24154942, 20132.233911]
        #dist = np.linalg.norm(HaloPos[0] - Pos) / (1 + ds.current_redshift) / CS.hubble_parameter
        #Rad = HaloRad[0] / (1 + ds.current_redshift) / CS.hubble_parameter
        #print(snap, HaloPos[0], HaloRad[0], dist + 1.5 * Rad, flush=True)
        
        for i in range(1):
            projector = YT.ProjectionPlot(ds, ad, normal="x")
            projector.add_field(("gas", field))
            
            #projector.set_center(Pos, 2 * (dist + Rad), units="kpc")
            #print(HaloRad[i] * 1e3 / CS.hubble_parameter / (1 + ds.current_redshift))
            projector.set_center(SubhaloPos[i], SubhaloRad[i] * 2.5, units="code_length")
            projector.plot()
            projector.customize(cmap=cmap, lowlim=lowlim, highlim=highlim, beauty=beauty)
            #projector.annotate_text(0.75, 0.25, f"Redshift = {ds.current_redshift:.2f}")
            #projector.annotate_text(0.75, 0.2, "M$_{Halo}$ = %.2e" % HaloMass[i])
            #projector.annotate_text(0.75, 0.15, "M$_{Gas}$ = %.2e" % HaloMassType[i, 0])
            #BHMass, StellarMass = projector.FindBHandStellarProperty(id_binary, type_binary)
            #projector.CheckContamination()
            #projector.annotate_text(0.75, 0.10, "M$_{BH}$ = %.2e" % BHMass)
            #projector.annotate_text(0.75, 0.05, "M$_{*}$ = %.2e" % StellarMass)
            #projector.plot_sink_particles(id_binary, type_binary, heavy_only=False, stars=False)
            projector.plot_halo(SubhaloPos[i], SubhaloRad[i], units="code_length")
            projector.save_plots(base=self.output_dir, prefix=f"HaloCenter_{snap}")

    def StellarProjectionPlot(self, snap):
        print("Entering StellarProjectionPlot() function.")

        Groups = GroupsReader.Reader(self.base, snap)
        Halos = Groups.halos
        HaloPos = Halos["GroupPos"]
        HaloMass = Halos["GroupMass"] * 1e10 / CS.hubble_parameter
        HaloRad = Halos["Group_R_TopHat200"]
        HaloMassType = Halos["GroupMassType"] * 1e10 / CS.hubble_parameter

        sfile = f"{self.base}snapdir_{snap:03d}/snap_{snap:03d}.0.hdf5"
        ds = yt.load(sfile)
        ad = ds.all_data()

        projector = YT.StellarParticlePlot(ds, ad)
        projector.set_center(HaloPos[0], HaloRad[0] / 2, units="code_length")
        projector.plot(cmap="Reds")
        projector.customize(beauty=True, buff_size=512)
        projector.save_plots(base=self.output_dir, prefix=f"StellarProjection_{snap}")
    
    def DarkMatterProjection(self, snap):
        print("Entering DarkMatterProjection function.")
        sfile = f"{self.base}snapdir_{snap:03d}/snap_{snap:03d}.0.hdf5"
        #sfile = f"{self.base}ics_Rarepeak_40mpc_L8_L13.hdf5"
        ds = yt.load(sfile)
        ad = ds.all_data()
        #Groups = GroupsReader.Reader(self.base, snap)
        #Halos = Groups.halos
        #HaloPos = Halos["GroupPos"]
        #HaloMass = Halos["GroupMass"] * 1e10 / CS.hubble_parameter
        #HaloRad = Halos["Group_R_TopHat200"]
        #HaloMassType = Halos["GroupMassType"] * 1e10 / CS.hubble_parameter
        #print(snap, HaloPos[0], HaloRad[0])
        particle = YT.ParticlePlot(ds, ad)
        #particle = YT.ParticleProjectionPlot(ds, ad)
        particle.add_field(("PartType1", "particle_mass"), weight_field=None)
        #particle.set_center(HaloPos[0], 7.5, units="kpc")
        particle.set_center(None, 6000, units="code_length")
        particle.plot()
        particle.customize(buff_size=800, beauty=True)#, lowlim=5e5, highlim=2e6)
        
        #particle.annotate_text(0.1, 0.95, f"Redshift = {ds.current_redshift:.2f}")
        #particle.annotate_text(0.1, 0.9, "M$_{Halo}$ = %.2e" % HaloMass[0])
        #particle.annotate_text(0.1, 0.85, "M$_{Gas}$ = %.2e" % HaloMassType[0, 0])
        #particle.annotate_text(0.1, 0.80, "M$_{BH, *}$ = %.2e" % HaloMassType[0, 5])
        #particle.plot_halo(HaloPos[0], HaloRad[0], units="code_length")
        particle.save_plots(base=self.output_dir, prefix=f"DarkMatter_{snap}")

    def BHStellar(self, snap):
        print("Entering BHStellar function.")

        plot = Plotter.Plotter(figsize=(9, 6))
        
        sfile = f"{self.base}snapdir_{snap:03d}/snap_{snap:03d}.0.hdf5"
        ds = yt.load(sfile)
        ad = ds.all_data()

        YTU = YT.YTUtils(ds, ad)

        SinkData = SinkParticles.SinkData(self.base)
        SinkData.filter_by_type("MBH", kind="self")
        HeavySeedIDs = SinkData.ids
        HeavySeedMasses = SinkData.M_final * 1e10 / CS.hubble_parameter
        mask = HeavySeedMasses > 1e5
        HeavySeedIDs = HeavySeedIDs[mask]

        self.BinaryReader.read_sink_snap(snap)
        id_binary = self.BinaryReader.extract_data("ID")
        type_binary = self.BinaryReader.extract_data("Type")
        mass_binary = self.BinaryReader.extract_data("StellarMass") * 1e10 / CS.hubble_parameter
        pos_binary = self.BinaryReader.extract_data("Pos")

        MBHMass = []
        HostStellarMass =[]
        IDS = []
        f = open(f"IntermediateData/HeavySeedLoc_{snap}.txt", "w")
        f.write("ID x y z HeavySeedMass StellarMass\n")
        for i, ID in enumerate(HeavySeedIDs):
            if ID not in id_binary:
                print("Black hole not in this snapshot. Either it has mergered or was created after this snapshot.")
                continue
            StellarMass = 0.0
            ind = ID == id_binary
            HeavySeedMass = mass_binary[ind][0]
            #if HeavySeedMass < 1e5: continue
            HeavySeedPos = pos_binary[ind].flatten()
            SinkRad = 1
            print(f"HeavySeedPos: {HeavySeedPos}")
            region = YTU.SelectRegion(HeavySeedPos, 100, units="pc")
            id5 = region[("PartType5", "ParticleIDs")]
            mass5 = region[("PartType5", "Masses")].in_units("Msun").value
            if "PartType4" in ds.fields:
                mass4 = region[("PartType4", "Masses")].in_units("Msun").value
                StellarMass += np.sum(mass4)
                print(f"PartType4 mass: {StellarMass}")         
            for j in range(len(id5)):
                ind = id5[j] == id_binary
                type5 = type_binary[ind]
                if type5 == 2:
                    StellarMass += mass5[j]
            #if StellarMass < 1e5: continue
            IDS.append(ID)
            MBHMass.append(HeavySeedMass)
            HostStellarMass.append(StellarMass)
            print(f"HeavySeedMass: {HeavySeedMass}, StellarMass: {StellarMass}")
            f.write(f"{ID} {HeavySeedPos[0]} {HeavySeedPos[1]} {HeavySeedPos[2]} {HeavySeedMass} {StellarMass}\n")
        f.close()
        IDS = np.array(IDS)
        MBHMass = np.array(MBHMass)
        HostStellarMass = np.array(HostStellarMass)
        
        
        df = pd.DataFrame({
            "stellar": HostStellarMass,
            "bh": MBHMass
            })
        df_max = df.groupby("stellar", as_index=False)["bh"].max()
        plot.ax.scatter(np.log10(df_max["stellar"]), np.log10(df_max["bh"]), color="blue", marker="d")
        

        ObservationalData = Utilities.ObservationData()
        MaiolinoData = ObservationalData.maiolino
        JuodzbalisData = ObservationalData.juodzbalis

        MaiolinoMBH = np.array([MaiolinoData[ID]["MBH"][0] for ID in MaiolinoData])
        MaiolinoStellar = np.array([MaiolinoData[ID]["MStar"][0] for ID in MaiolinoData])

        JuodzbalisMBH = np.array([JuodzbalisData[ID]["MBH"][0] for ID in JuodzbalisData], dtype=float)
        JuodzbalisStellar = np.array([JuodzbalisData[ID]["MStar"][0] for ID in JuodzbalisData], dtype=float)

        plot.ax.scatter(MaiolinoStellar, MaiolinoMBH, color="purple", marker="s")
        plot.ax.scatter(JuodzbalisStellar, JuodzbalisMBH, color="red", marker="s")

        x = np.logspace(5, 11)
        ReinesData, err = ObservationalData.Reines(x)
        err = np.max(err) * np.ones_like(ReinesData)

        plot.ax.plot(np.log10(x), ReinesData, color="grey")
        plot.ax.fill_between(np.log10(x), ReinesData-err, ReinesData+err, alpha=0.4, color="grey")

        plot.ax.plot(np.log10(x), np.log10(x), linestyle="--", color="grey")
        plot.ax.text(8, 8.1, "M$_{\mathrm{BH}}$ = M$_*$", ha="center", rotation=40, rotation_mode="anchor", color="grey", fontsize=12)
        y = 0.1 * x 
        plot.ax.plot(np.log10(x), np.log10(y), linestyle="--", color="grey")
        plot.ax.text(9, 8.1, "M$_{\mathrm{BH}}$ = 0.1 * M$_*$", ha="center", rotation=40, rotation_mode="anchor", color="grey", fontsize=12)

        plot.add_details(plot.ax, xlabel="log$_{10}$(M$_*$)", ylabel="log$_{10}$(M$_{\mathrm{BH}}$)", xlim=(5, 11), ylim=(4, 9))

        plot.add_legends(color="gray", label="Reines & Volonteri 2015")
        plot.add_legends(type="scatter", color="purple", marker="s", label="Maiolino et al. 2023")
        plot.add_legends(type="scatter", color="red", marker="s", label="Juod\u017Ebalis et al. 2025")
        plot.add_legends(type="scatter", color="blue", marker="d", label="This Work")

        plot.save(f"{self.output_dir}BHStellar.png")

    def BHStellarHalo(self, snap):
        print("Entering BHStellarHalo function.")
        
        Groups = GroupsReader.Reader(self.base, snap)
        Subhalos = Groups.subhalos
        SubhaloPos = Subhalos["SubhaloCM"]
        SubhaloRad = Subhalos["SubhaloHalfmassRad"]
        SubhalosMassType = Subhalos["SubhaloMassType"] * 1e10 / CS.hubble_parameter
        N = len(SubhalosMassType)
        print(f"Number of subhalos in this snapshot is {N}")

        sfile = f"{self.base}snapdir_{snap:03d}/snap_{snap:03d}.0.hdf5"
        ds = yt.load(sfile)
        ad = ds.all_data()
        YTU = YT.YTUtils(ds, ad)
        self.BinaryReader.read_sink_snap(snap)
        id_binary = self.BinaryReader.extract_data("ID")
        type_binary = self.BinaryReader.extract_data("Type")
        mass_binary = self.BinaryReader.extract_data("StellarMass") * 1e10 / CS.hubble_parameter
        pos_binary = self.BinaryReader.extract_data("Pos")
        MBHID = []
        SubhaloMBH = []
        SubhaloStellar = []
        for i in range(N):
            if SubhalosMassType[i, 5] < 1e5: continue
            StellarMass = 0.0
            MBH = 0.0
            region = YTU.SelectRegion(SubhaloPos[i], SubhaloRad[i], units="code_length")
            mass2 = region[("PartType2", "Masses")].in_units("Msun").value
            if np.sum(mass2) > 0:
                print(f"Halo contanimated. {np.sum(mass2)} Msun of contanimation. Aborting for this subhalo.")
                continue
            mass5 = region[("PartType5", "Masses")].in_units("Msun").value
            id5 = region[("PartType5", "ParticleIDs")].value

            if "PartType4" in ds.fields:
                mass4 = region[("PartType4", "Masses")].in_units("Msun").value
                StellarMass += np.sum(mass4)

            
            for j in range(len(id5)):
                mask = id5[j] == id_binary
                typ = type_binary[mask]
                if typ == 3:
                    MBH = max(MBH, mass5[j])
                if typ == 2:
                    StellarMass += mass5[j]
            if MBH < 1e5: continue
            if StellarMass < 1e5: continue
            SubhaloMBH.append(MBH)
            SubhaloStellar.append(StellarMass)

            Ind = MBH == mass5
            MBHID.append(int(DU.unwrap(id5[Ind])))
        MBHID = np.array(MBHID)
        SubhaloMBH = np.array(SubhaloMBH)
        SubhaloStellar = np.array(SubhaloStellar)
        
        print(f"MBH ID: {MBHID}")
        print(f"MBH mass: {SubhaloMBH}")
        print(f"StellarMass: {SubhaloStellar}")
        plot = Plotter.Plotter(figsize=(9, 6))
        plot.ax.scatter(np.log10(SubhaloStellar), np.log10(SubhaloMBH), color="blue", marker="d")
        

        ObservationalData = Utilities.ObservationData()
        MaiolinoData = ObservationalData.maiolino
        JuodzbalisData = ObservationalData.juodzbalis

        MaiolinoMBH = np.array([MaiolinoData[ID]["MBH"][0] for ID in MaiolinoData])
        MaiolinoStellar = np.array([MaiolinoData[ID]["MStar"][0] for ID in MaiolinoData])

        JuodzbalisMBH = np.array([JuodzbalisData[ID]["MBH"][0] for ID in JuodzbalisData], dtype=float)
        JuodzbalisStellar = np.array([JuodzbalisData[ID]["MStar"][0] for ID in JuodzbalisData], dtype=float)

        plot.ax.scatter(MaiolinoStellar, MaiolinoMBH, color="purple", marker="s")
        plot.ax.scatter(JuodzbalisStellar, JuodzbalisMBH, color="red", marker="s")

        x = np.logspace(5, 11)
        ReinesData, err = ObservationalData.Reines(x)
        err = np.max(err) * np.ones_like(ReinesData)

        plot.ax.plot(np.log10(x), ReinesData, color="grey")
        plot.ax.fill_between(np.log10(x), ReinesData-err, ReinesData+err, alpha=0.4, color="grey")

        plot.ax.plot(np.log10(x), np.log10(x), linestyle="--", color="grey")
        plot.ax.text(8, 8.1, "M$_{\mathrm{BH}}$ = M$_*$", ha="center", rotation=40, rotation_mode="anchor", color="grey", fontsize=12)
        y = 0.1 * x 
        plot.ax.plot(np.log10(x), np.log10(y), linestyle="--", color="grey")
        plot.ax.text(9, 8.1, "M$_{\mathrm{BH}}$ = 0.1 * M$_*$", ha="center", rotation=40, rotation_mode="anchor", color="grey", fontsize=12)

        plot.add_details(plot.ax, xlabel="log$_{10}$(M$_*$)", ylabel="log$_{10}$(M$_{\mathrm{BH}}$)", xlim=(5, 11), ylim=(4, 9))

        plot.add_legends(color="gray", label="Reines & Volonteri 2015")
        plot.add_legends(type="scatter", color="purple", marker="s", label="Maiolino et al. 2023")
        plot.add_legends(type="scatter", color="red", marker="s", label="Juod\u017Ebalis et al. 2025")
        plot.add_legends(type="scatter", color="blue", marker="d", label="This Work")

        plot.save(f"{self.output_dir}BHStellarHalo.png")

        return MBHID

    def BHStellarCummulative(self):
        BHStellarData = self.PickleReader.pickle_reader(f"BHStellarData.pkl")
        plot = Plotter.Plotter(figsize=(9, 6))

        for SinkID in BHStellarData:
            Data = BHStellarData[SinkID]
            snaps = np.array(list(sorted(Data.keys())))
            MBHMass = np.array([Data[s]["MBHMass"] for s in snaps])
            StellarMass = np.array([Data[s]["StellarMass"] for s in snaps])
            if MBHMass[-1] < 1e5: continue
            plot.ax.scatter(np.log10(StellarMass[-1]), np.log10(MBHMass[-1]), color="blue", marker="d")
            if StellarMass[-1] != np.max(StellarMass): continue
            plot.ax.plot(np.log10(StellarMass), np.log10(MBHMass), color="blue", alpha=0.5)
            
        
        ObservationalData = Utilities.ObservationData()
        MaiolinoData = ObservationalData.maiolino
        JuodzbalisData = ObservationalData.juodzbalis

        MaiolinoMBH = np.array([MaiolinoData[ID]["MBH"][0] for ID in MaiolinoData])
        MaiolinoStellar = np.array([MaiolinoData[ID]["MStar"][0] for ID in MaiolinoData])

        JuodzbalisMBH = np.array([JuodzbalisData[ID]["MBH"][0] for ID in JuodzbalisData], dtype=float)
        JuodzbalisStellar = np.array([JuodzbalisData[ID]["MStar"][0] for ID in JuodzbalisData], dtype=float)

        plot.ax.scatter(MaiolinoStellar, MaiolinoMBH, color="purple", s=15, marker="s")
        plot.ax.scatter(JuodzbalisStellar, JuodzbalisMBH, color="red", s=15, marker="s")

        x = np.logspace(5, 11)
        ReinesData, err = ObservationalData.Reines(x)
        err = np.max(err) * np.ones_like(ReinesData)

        plot.ax.plot(np.log10(x), ReinesData, color="grey")
        plot.ax.fill_between(np.log10(x), ReinesData-err, ReinesData+err, alpha=0.4, color="grey")

        plot.ax.plot(np.log10(x), np.log10(x), linestyle="--", color="grey")
        plot.ax.text(8, 8.1, "M$_{\mathrm{BH}}$ = M$_*$", ha="center", rotation=41, rotation_mode="anchor", color="grey", fontsize=12)
        y = 0.1 * x 
        plot.ax.plot(np.log10(x), np.log10(y), linestyle="--", color="grey")
        plot.ax.text(9, 8.1, "M$_{\mathrm{BH}}$ = 0.1 * M$_*$", ha="center", rotation=41, rotation_mode="anchor", color="grey", fontsize=12)

        plot.add_details(plot.ax, xlabel="log$_{10}$(M$_*$)", ylabel="log$_{10}$(M$_{\mathrm{BH}}$)", xlim=(5, 11), ylim=(4, 9))

        plot.add_legends(color="gray", label="Reines & Volonteri 2015")
        plot.add_legends(type="scatter", color="purple", marker="s", label="Maiolino et al. 2023")
        plot.add_legends(type="scatter", color="red", marker="s", label="Juod\u017Ebalis et al. 2025")
        plot.add_legends(type="scatter", color="blue", marker="d", label="This Work")

        plot.save(f"{self.output_dir}BHStellarCummulative.png")
        
    def MergerPlots(self):
        SinkDataAll = SinkParticles.SinkDataAll(self.base)
        MergerMap = SinkDataAll.merger_information()
        plot = Plotter.Plotter(figsize=(9, 6))

        N = len(MergerMap.keys())
        MaxMergers = 34 + 1
        cmap = plt.get_cmap("magma_r")
        colors = [cmap(i) for i in np.linspace(0, 1, MaxMergers)]
        M1All = []
        M2All = []
        MrAll = []
        for i, ID in enumerate(MergerMap.keys()):
            M1 = MergerMap[ID]["M1"] * 1e10 / CS.hubble_parameter
            M2 = MergerMap[ID]["M2"] * 1e10 / CS.hubble_parameter
            Mr = MergerMap[ID]["Mr"] * 1e10 / CS.hubble_parameter
            times = MergerMap[ID]["Time"]
            Redshift = 1 / times - 1
            CosmicTime = cosmo.age(Redshift).to("Myr").value
            print(len(M1))
            ind = np.argsort(Redshift)
            plot.ax.scatter(Redshift[ind], np.log10(M1[ind]), color=colors[len(M1)], alpha=max(len(M1)/(MaxMergers + 1), 0.4), s=len(M1)*5, zorder=len(M1))
            plot.ax.plot(Redshift[ind], np.log10(M1[ind]), color=colors[len(M1)], alpha=max(len(M1)/(MaxMergers + 1), 0.4), zorder=len(M1))
            if len(M1) > 0:
                for j in range(len(M1)):
                    M1All.append(M1[j])
                    M2All.append(M2[j])
                    MrAll.append(Mr[j])
            else:
                M1All.append(M1)
                M2All.append(M2)
                MrAll.append(Mr)
        
        plot.ax.invert_xaxis()
        norm = mpl.colors.Normalize(vmin=1, vmax=50)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        plot.add_details(plot.ax, xlabel="Redshift", ylabel="log$_{10}$(M1)", h=sm, colorbar=True, cbarlabel="Number of Mergers", xlim=(20, 11), ylim=(4, 9.25))
        plot.save(f"{self.output_dir}MergerPlot.png")
        
        plot = Plotter.Plotter(figsize=(6, 6))
        x = np.linspace(0, 8, 100)
        plot.ax.plot(x, x, linestyle="--", color="grey")
        for i in range(len(M1All)):
            if M2All[i] > M1All[i]:
                temp = M1All[i]
                M1All[i] = M2All[i]
                M2All[i] = temp
        plot.ax.scatter(np.log10(M1All), np.log10(M2All), alpha=0.5, color="blue")
        plot.add_details(plot.ax, xlabel="log$_{10}$(M1)", ylabel="log$_{10}$(M2)", ylim=(0.1, 7.5), xlim=(0.1, 7.5))
        plot.save(f"{self.output_dir}M1M2.png")

        #################################################################################################################
        #################################################################################################################
        #################################################################################################################
        sink_particles = SinkDataAll.sink_particles

        plot = Plotter.Plotter(figsize=(9, 6))
        DataAccretion = {}
        DataMerger = {}
        DataTotal = {}
        for ID, sink in sink_particles.items():
            meta = sink["meta"]
            if meta["Type"] != "MBH":
                continue

            evolution = sink_particles[ID]["evolution"]
            times = np.array(list(sorted(evolution.keys())))
            Redshift = 1 / times - 1
            StellarMass = np.array([evolution[t]["StellarMass"] for t in times]) * 1e10 / CS.hubble_parameter
            MergerMass = np.array([evolution[t]["MergerMass"] for t in times]) * 1e10 / CS.hubble_parameter
            if StellarMass[-1] == StellarMass[0] or StellarMass[-1] < StellarMass[0]: continue
            mask = MergerMass > StellarMass
            if len(MergerMass[mask]) > 0:
                print(f"One of the MergerMass is Greater than StellarMass for ID {ID}")
                continue
            for i in range(len(Redshift)):
                z = Redshift[i]
                if StellarMass[i] - StellarMass[0] == 0: continue
                if StellarMass[i] <= MergerMass[i] : raise ValueError(f"MergerMass {MergerMass[i]} greater than StellarMass {StellarMass[i]}.")
                if z in DataMerger: 
                    DataAccretion[z] += StellarMass[i] - MergerMass[i] - StellarMass[0]
                    DataMerger[z] += MergerMass[i]
                    DataTotal[z] += StellarMass[i]
                    
                else:
                    DataAccretion[z] = StellarMass[i] - MergerMass[i] - StellarMass[0]
                    DataMerger[z] = MergerMass[i]
                    DataTotal[z] = StellarMass[i]
                    
            

        Redshift = np.array(list(sorted(DataMerger.keys())))
        MergerGrowth = np.array([DataMerger[z] for z in Redshift])
        AccretionGrowth = np.array([DataAccretion[z] for z in Redshift])
        TotalGrowth = np.array([DataTotal[z] for z in Redshift])
        mask = np.where((MergerGrowth > 0))
        plot.ax.plot(Redshift[mask], MergerGrowth[mask]/TotalGrowth[mask], color="red")
        mask = np.where(AccretionGrowth > 0)
        plot.ax.plot(Redshift[mask], AccretionGrowth[mask]/TotalGrowth[mask], color="blue")
        plot.ax.plot(Redshift, TotalGrowth/TotalGrowth, color="black")
        print(MergerGrowth[mask])
        print(AccretionGrowth[mask])

        plot.ax.invert_xaxis()
        plot.add_details(plot.ax, xlabel="Redshift", ylabel="Mass Fraction from Mergers", yscale="log", ylim=(1e-2, 2))
        plot.add_legends(color="red", label="Merger Fraction")
        plot.add_legends(color="blue", label="Accretion Fraction.")
        plot.add_legends(color="black", label="Total Mass")
        plot.save(f"{self.output_dir}MergerMassFrac.png", legend_loc="lower left")
 
    def SinkGrowth(self):
        plot = Plotter.Plotter(figsize=(9, 6))
        ax2 = plot.ax.twiny()
        plotPopIII = Plotter.Plotter(figsize=(9, 6))
        plotMBH = Plotter.Plotter(figsize=(9, 6))
        plotAR = Plotter.Plotter(figsize=(9, 6))
        ax2AR = plotAR.ax.twiny()
        plotER = Plotter.Plotter(figsize=(9, 6))
        ax2ER = plotER.ax.twiny()

        plotMerger = Plotter.Plotter(figsize=(9, 6))

        SnapshotRedshifts = np.genfromtxt("redshift.txt", usecols=(0), unpack=True)
        SnapshotRedshifts = SnapshotRedshifts[::-1]
        SinkDataAll = SinkParticles.SinkDataAll(self.base, feedback=self.feedback, array_length=self.array_length)
        sink_particles = SinkDataAll.sink_particles
        SinkIDs = sink_particles.keys()
        
        for i, ID in enumerate(SinkIDs):
            print(f"\rDoing for {i}/{len(SinkIDs)}.", end="")
            meta = sink_particles[ID]["meta"]
            evolution = sink_particles[ID]["evolution"]
            times = np.array(list(sorted(evolution.keys())))
            StellarMass = np.array([evolution[t]["StellarMass"] for t in times]) * 1e10 / CS.hubble_parameter
            MergerMass = np.array([evolution[t]["MergerMass"] for t in times]) * 1e10 / CS.hubble_parameter
            Redshift = 1 / times - 1
            

            if meta["Type"] == "PopII":
                plot.ax.plot(Redshift, StellarMass, color="orange", alpha=0.3, zorder=0)
            if meta["Type"] == "MBH":
                plot.ax.plot(Redshift, StellarMass, color="black", alpha=0.8, zorder=2)
                plotMBH.ax.plot(Redshift, StellarMass, color="black", alpha=0.7)
                #SnapshotStellar = np.interp(SnapshotRedshifts, Redshift[::-1], StellarMass[::-1])
                #plotMBH.ax.scatter(SnapshotRedshifts[::-1], SnapshotStellar[::-1], color="black", s=1)

                EvolutionTimes = cosmo.age(Redshift).to("Myr").value
                AccretionRate = np.diff(StellarMass) / np.diff(EvolutionTimes * 1e6)
                EddingtonLimit = DU.EddingtonRate(StellarMass[:-1])
                EddingtonRate = AccretionRate / EddingtonLimit
                maskAR = AccretionRate > 1e-5
                maskER = EddingtonRate > 1e-4
                plotAR.ax.plot(Redshift[:-1][maskAR], AccretionRate[maskAR], color="black", alpha=0.6)
                plotER.ax.plot(Redshift[:-1][maskER], EddingtonRate[maskER], color="black", alpha=0.6)

                mask = MergerMass != 0
                if len(MergerMass[mask]) > 1:
                    plotMerger.ax.plot(Redshift[mask], MergerMass[mask], alpha=0.5, color="black")


            if meta["Type"] == "PopIII":
                Type = np.array([evolution[t]["Type"] for t in times])
                PopIIIInd = np.where(Type == 0)[0]
                plot.ax.plot(Redshift[0:len(PopIIIInd)-1], StellarMass[0:len(PopIIIInd)-1], color="blue", zorder=1, alpha=0.5)
                plot.ax.plot(Redshift[len(PopIIIInd):], StellarMass[len(PopIIIInd):], color="black", zorder=2, alpha=0.8)
                plotPopIII.ax.plot(Redshift[0:len(PopIIIInd)-1], StellarMass[0:len(PopIIIInd)-1], color="blue", zorder=1, alpha=0.5)
                plotPopIII.ax.plot(Redshift[len(PopIIIInd):], StellarMass[len(PopIIIInd):], color="black", zorder=2, alpha=0.8)

                mask = MergerMass != 0
                if len(MergerMass[mask]) > 1:
                    if MergerMass[-1] < 1e3: continue
                    plotMerger.ax.plot(Redshift[mask], MergerMass[mask], alpha=0.5, color="black")
                    #plotMerger.ax.plot(Redshift[mask], StellarMass[mask], alpha=0.5, color="red")
        
        plot.ax.invert_xaxis()
        RedshiftTicks = np.linspace(plot.ax.get_xlim()[0], plot.ax.get_xlim()[1], 6)
        TimeTicks = cosmo.age(RedshiftTicks).to("Myr").value
        ax2.set_xlim(plot.ax.get_xlim())
        ax2.set_xticks(RedshiftTicks)
        ax2.set_xticklabels([f"{t:.1f}" for t in TimeTicks])
        ax2.set_xlabel("Time [Myr]", fontsize=15)
        plot.add_legends(color="blue", label="PopIII")
        plot.add_legends(color="orange", label="PopII")
        plot.add_legends(color="black", label="MBH")
        plot.add_details(plot.ax, xlabel="Redshift", ylabel="SinkMass [M$_\odot$]", yscale="log")
        plot.save(f"{self.output_dir}SinkGrowth.png")
        
        plotPopIII.ax.invert_xaxis()
        x = np.linspace(plotPopIII.ax.get_xlim()[0], plotPopIII.ax.get_xlim()[1], 100)
        plotPopIII.ax.fill_between(x, 11, 40, color="green", alpha=0.5)
        plotPopIII.ax.fill_between(x, 140, 260, color="purple", alpha=0.5)
        plotPopIII.add_legends(type="patch", color="purple", label="PISN")
        plotPopIII.add_legends(type="patch", color="green", label="Type-II SNe")
        plotPopIII.add_details(plotPopIII.ax, xlabel="Redshift", ylabel="PopIII Mass [M$_\odot$]", yscale="log")
        plotPopIII.save(f"{self.output_dir}PopIIIEvolution.png")

        plotMBH.ax.invert_xaxis()
        plotMBH.add_details(plotMBH.ax, xlabel="Redshift", ylabel="MBH Mass [M$_\odot$]", yscale="log", xlim=(15, 30))
        plotMBH.save(f"{self.output_dir}MBHEvolution.png")

        plotAR.ax.invert_xaxis()
        RedshiftTicks = np.linspace(plotAR.ax.get_xlim()[0], plotAR.ax.get_xlim()[1], 6)
        TimeTicks = cosmo.age(RedshiftTicks).to("Myr").value
        ax2AR.set_xlim(plotAR.ax.get_xlim())
        ax2AR.set_xticks(RedshiftTicks)
        ax2AR.set_xticklabels([f"{t:.1f}" for t in TimeTicks])
        ax2AR.set_xlabel("Time [Myr]", fontsize=15)
        plotAR.add_details(plotAR.ax, xlabel="Redshift", ylabel="Accretion Rate [M$_\odot / yr$]", yscale="log", ylim=(1e-5, 1e2))
        plotAR.save(f"{self.output_dir}AccretionRate.png")

        plotER.ax.invert_xaxis()
        RedshiftTicks = np.linspace(plotER.ax.get_xlim()[0], plotER.ax.get_xlim()[1], 6)
        TimeTicks = cosmo.age(RedshiftTicks).to("Myr").value
        ax2ER.set_xlim(plotER.ax.get_xlim())
        ax2ER.set_xticks(RedshiftTicks)
        ax2ER.set_xticklabels([f"{t:.1f}" for t in TimeTicks])
        ax2ER.set_xlabel("Time [Myr]", fontsize=15)
        plotER.add_details(plotER.ax, xlabel="Redshift", ylabel="f$_{\mathrm{edd}}$", yscale="log", ylim=(1e-4, 1e4))
        plotER.save(f"{self.output_dir}EddingtonRatio.png")
            
        plotMerger.ax.invert_xaxis()
        plotMerger.add_details(plotMerger.ax, xlabel="Redshift", ylabel="MergerMass [M$_\odot$]", yscale="log") 
        plotMerger.save(f"{self.output_dir}MergerMass.png")

    def SinkGrowthSnaps(self, feedback=False):
        print("Entering SinkGrowthSnaps() function.")

        SinkData = SinkParticles.SinkData(self.base)
        sink_particles = SinkData.sink_particles

        MBHID = [102573735, 154380948, 339351372, 291714062, 283304939, 204321441, 304809353, 170907394, 353631857, 306226867, 326692425, 232920095, 308660604, 287571337, 287719248, 346815834, 233552922, 377533351, 375503146, 312257629, 294538671, 319592445]

        plot = Plotter.Plotter(figsize=(9, 6))
        for ID in sink_particles.keys():
            if ID not in MBHID: continue
            evolution = sink_particles[ID]["evolution"]
            times = np.array(list(sorted(evolution.keys())))
            StellarMass = np.array([evolution[t]["StellarMass"] for t in times]) * 1e10 / CS.hubble_parameter
            Redshift = 1 / times - 1

            plot.ax.plot(Redshift, StellarMass)
        
        plot.ax.invert_xaxis()

        plot.add_details(xlabel="Redshift", ylabel="Mass", yscale="log")
        plot.save(f"{self.output_dir}SinkGrowthreduced.png")
        
    def BHMassFunction(self):
        plot = Plotter.Plotter()
        SinkDataAll = SinkParticles.SinkDataAll(self.base)
        dN_dM1, dN_dM2, dN_dM3, bin_centers = SinkDataAll.BlackHoleMassFunction()
        mask = dN_dM3 > 1e-6
        plot.ax.scatter(np.log10(bin_centers[mask]), dN_dM3[mask], color="blue", marker="d")
        plot.ax.plot(np.log10(bin_centers[mask]), dN_dM3[mask], color="blue")

        ObservationData = Utilities.ObservationData()
        Taylor = ObservationData.taylor

        bin_centers = np.array([Taylor[i]["MBH"] for i in Taylor])
        Phi = np.array([Taylor[i]["Phi"][0] for i in Taylor]) * 1e-6
        Phihigh = np.array([Taylor[i]["Phi"][1] for i in Taylor]) * 1e-6
        Philow = np.array([Taylor[i]["Phi"][2] for i in Taylor]) * 1e-6
        #plot.ax.scatter(bin_centers, Phi, color="grey")
        plot.ax.errorbar(bin_centers, Phi, yerr=[Philow, Phihigh], color="grey", capsize=2, fmt="o")
        
        plot.add_legends(type="scatter", color="grey", label="Taylor et al 2024")
        plot.add_legends(type="scatter", color="blue", label="This Work", marker="d")
        plot.add_details(plot.ax, xlabel="log$_{10}$(M$_{\mathrm{BH}}$)", ylabel="$\Phi$ [Mpc$^{-1}$ dex$^{-1}$]", yscale="log", ylim=(1e-6, 1e0))
        plot.save(f"{self.output_dir}BHMassFunction.png", legend_loc="upper right")
  
    def LuminosityFunction(self):
        plot = Plotter.Plotter(figsize=(9, 6))
        SinkDataAll = SinkParticles.SinkDataAll(self.base)
        MBHMass, Lbol = SinkDataAll.LuminosityFunction()
        plot.ax.scatter(np.log10(MBHMass), np.log10(Lbol), color="blue", marker="d")

        ObservationData = Utilities.ObservationData()
        MaiolinoData = ObservationData.maiolino
        JuodzbalisData = ObservationData.juodzbalis

        MBHMass = np.array([MaiolinoData[i]["MBH"][0] for i in MaiolinoData])
        Lbol = np.array([MaiolinoData[i]["Lbol"] for i in MaiolinoData])
        plot.ax.scatter(MBHMass, Lbol, color="purple", marker="s")

        MBHMass = np.array([JuodzbalisData[i]["MBH"][0] for i in JuodzbalisData])
        Lbol = np.array([JuodzbalisData[i]["Lbol"] for i in JuodzbalisData])
        plot.ax.scatter(MBHMass, Lbol, color="red", marker="s")

        mass = np.logspace(5, 9, 100)
        Lbol = DU.EddingtonLuminosity(mass)

        plot.ax.plot(np.log10(mass), np.log10(Lbol), color="grey", linestyle="--")
        x = 8.5
        y = DU.EddingtonLuminosity(10**x)
        plot.ax.text(x, 1.001*np.log10(y), "$\lambda_{\mathrm{Edd}}$", ha="center", rotation=35, rotation_mode="anchor", color="grey", fontsize=12)
        plot.ax.plot(np.log10(mass), np.log10(0.1 * Lbol), color="grey", linestyle="--")
        plot.ax.text(x, 1.001*np.log10(0.1 * y), "$0.1 * \lambda_{\mathrm{Edd}}$", ha="center", rotation=35, rotation_mode="anchor", color="grey", fontsize=12)
        plot.ax.plot(np.log10(mass), np.log10(0.01 * Lbol), color="grey", linestyle="--")
        plot.ax.text(x, 1.001*np.log10(0.01 * y), "$0.1 * \lambda_{\mathrm{Edd}}$", ha="center", rotation=35, rotation_mode="anchor", color="grey", fontsize=12)



        plot.add_legends(type="scatter", color="purple", label="Maiolino et al 2024", marker="s")
        plot.add_legends(type="scatter", color="red", label="Juod\u017Ebalis et al. 2025", marker="s")
        plot.add_legends(type="scatter", color="blue", label="This Work", marker="d")

        plot.add_details(plot.ax, xlabel="log$_{10}$(M$_{\mathrm{BH}}$) [M$_\odot$]", ylabel="log$_{10}$(L$_{\mathrm{bol}}$) [erg s$^{-1}$]")
        plot.save(f"{self.output_dir}LuminosityFunction.png")

    def PopIIMerger(self):
        for snap in range(self.end_snap-1, self.end_snap):
            self.BinaryReader.read_sink_snap(snap)
            if self.BinaryReader.num_sinks == 0: continue
            Type = self.BinaryReader.extract_data("Type")
            if len(Type[Type == 2]) == 0: continue
            time = self.BinaryReader.time
            Pos = self.BinaryReader.extract_data("Pos") * time / CS.hubble_parameter
            Vel = self.BinaryReader.extract_data("Vel") * pow(time, 0.5)
            Mass = self.BinaryReader.extract_data("StellarMass") * 1e10 / CS.hubble_parameter
            mask = Type == 2
            PopIILength = len(Type[mask])
            Pos = Pos[mask]
            Vel = Vel[mask]
            Mass = Mass[mask]
            bound = 0
            unbound = 0
            print(PopIILength)
            print(np.shape(Pos))
            tree = cKDTree(Pos * 1e3) # in pc
            pairs = tree.query_pairs(r=5)

            for i, j in pairs:
                p1, v1, m1 = Pos[i], Vel[i], Mass[i]
                p2, v2, m2 = Pos[j], Vel[j], Mass[j]

                r = np.linalg.norm(p1 - p2) * 1e3
                Egrav = - CS.G_in_cgs * m1 * CS.Msun_to_g * m2 * CS.Msun_to_g / (r * CS.pc_in_cgs)
                
                mu = m1 * m2 / (m1 + m2)
                v = np.linalg.norm(v1 - v2) * 1e5 # in cm/s
                Ekin = 0.5 * mu * CS.Msun_to_g * pow(v, 2)
                E = Egrav + Ekin
                if E < 0: bound += 1
                if E > 0: unbound += 1

            print(bound, unbound, PopIILength)
              
    def NumberDensityEvolution(self):
        plot = Plotter.Plotter()
        self.base = "/home/daxal/data/ProductionRuns/Rarepeak_zoom/"
        self.output_dir = "Random/"
        dire = ["Level13/", "Level14/", "Level15/", "Level15_feedback/"]
        Volume = [1**3, 1**3, 0.5**3, 0.5**3]

        colors = color_schemes["Material"]
        
        for i in range(len(dire)):
            feedback = False
            if dire[i] == "Level15_feedback/": feedback = True
            SinkData = SinkParticles.SinkData(self.base + dire[i])
            MDiff = SinkData.M_diff * 1e10 / CS.hubble_parameter
            MInit = SinkData.M_init * 1e10 / CS.hubble_parameter
            MFinal = SinkData.M_final * 1e10 / CS.hubble_parameter

            ind = np.where((MFinal > 1e3) & (MDiff > MInit))[0]
            ids = SinkData.ids[ind]
            print(ids)
            print(f"Number of heavy seeds in {dire[i]} is {len(ids)}")
            start_snap, end_snap = DU.file_range(f"{self.base}{dire[i]}")
            HeavySeedNumber = np.zeros(end_snap - start_snap, dtype=float)
            GalaxyNumber = np.zeros(end_snap - start_snap, dtype=float)
            Redshift = np.zeros(end_snap - start_snap, dtype=float)
            for j in range(start_snap, end_snap):
                if i == 0 and j == 32: continue
                self.BinaryReader = DataReader.BinaryReader(feedback=feedback)
                self.BinaryReader.read_sink_snap(j)
                if self.BinaryReader.time == 0: continue
                
                Redshift[j - start_snap] = 1 / self.BinaryReader.time - 1
                if self.BinaryReader.num_sinks == 0: continue
                Groups = GroupsReader.Reader(f"{self.base}{dire[i]}", j)
                Halos = Groups.halos
                HaloMass = Halos["GroupMass"] * 1e10 / CS.hubble_parameter
                SinkMass = Halos["GroupMassType"][:, 5] * 1e10 / CS.hubble_parameter

                if np.sum(SinkMass) < 1e3: continue

                mask = SinkMass > 0
                GalaxyNumber[j - start_snap] = len(HaloMass[mask])

                
                
                Type = self.BinaryReader.extract_data("Type")
                ID5 = self.BinaryReader.extract_data("ID")
                Mass = self.BinaryReader.extract_data("StellarMass") * 1e10 / CS.hubble_parameter
                mask = Mass > 1e3
                if len(Mass[mask]) == 0: continue
                for k, ID in enumerate(ids):
                    if ID not in ID5: continue
                    mask = ID5 == ID
                    if Type[mask] == 3 and Mass[mask] > 1e3:
                        HeavySeedNumber[j - start_snap] += 1
                
            
                

            norm = Groups.Normalization(Redshift[-1])

            HeavySeedNumber[:] *= norm
            GalaxyNumber[:] *= norm
            
            mask = HeavySeedNumber > 0
            plot.ax.plot(Redshift[mask], HeavySeedNumber[mask] / Volume[i], color=colors[i])
            mask = GalaxyNumber > 0
            plot.ax.plot(Redshift[mask], GalaxyNumber[mask] / Volume[i], color=colors[i], linestyle="--")
        
        plot.ax.invert_xaxis()
        plot.add_details(plot.ax, xlabel="Redshift", ylabel="dN/dV [cMpc$^{-3}$]", yscale="log", xlim=(34, 14))
        plot.add_legends(color=colors[0], label="L13")
        plot.add_legends(color=colors[1], label="L14")
        plot.add_legends(color=colors[2], label="L15")
        plot.add_legends(color=colors[3], label="L15_BHFB")
        plot.add_legends(color="black", label="MBH")
        plot.add_legends(color="black", label="Galaxy", linestyle="--")

        plot.save(f"{self.output_dir}NumberDensityEvolution.png", legend_loc="center right")        
    
    def CoolingTimeRatios(self):
        print("Entering cooling time ratios.")

        RadiationData = self.PickleReader.pickle_reader(f"RadiationPressure.pkl")

        TotalMasses = np.array([RadiationData[IDS]["TotalGasMass"] for IDS in RadiationData])
        Radius = 0.5 

        Density = TotalMasses / (4 / 3 * np.pi * Radius**3) # in Msun / pc^3
        Density *= CS.Msun_to_g / (CS.pc_in_cgs**3) # in g / cm^3
        print(Density)
        NumberDensity = Density / CS.m_p_in_cgs

        print(NumberDensity)

        Ratios = DU.CoolingTimeSoundCrossingTimeRation(NumberDensity, 1e5, 0.1)
        print(Ratios)

        plot = Plotter.Plotter()
        plot.ax.hist(Ratios, bins=100)
        plot.add_details(xlabel="tc/ts", ylabel="Num")
        plot.save(f"{self.output_dir}CoolingTimeRatios.png")

    def MetallicityChecks(self, snap):

        sfile = f"{self.base}snapdir_{snap:03d}/snap_{snap:03d}.0.hdf5"
        ds = yt.load(sfile)
        ad = ds.all_data()
        proj = YT.YTPlotter(ds, ad)
        proj.add_metallicity()
        GasMetallicity = proj.ad[("PartType0", "Metallicity")].value

        plot = Plotter.Plotter()
        bins = DU.histogram_bins(GasMetallicity, 100)
        plot.ax.hist(GasMetallicity, bins=bins)
        plot.add_details(plot.ax, xscale="log", yscale="log")
        plot.save(f"{self.output_dir}MetallicityRange.png")

    def AbouttheFiveLargestBH(self, snap):
        self.BinaryReader.read_sink_snap(snap)
        SinkMass = self.BinaryReader.extract_data("Mass") * 1e10 / CS.hubble_parameter
        SinkPos = self.BinaryReader.extract_data("Pos")
        ind = np.argsort(SinkMass)[-5:][::-1]

        SinkMass = SinkMass[ind]
        SinkPos = SinkPos[ind]

        Groups = GroupsReader.Reader(self.base, snap)
        for Pos, Mass in zip(SinkPos, SinkMass):
            print(Pos, Mass)
            print(Groups.HostGalaxyProperties(Pos))
            self.GasProjectionSink(snap, Pos)

    def AllPurposeFunction(self):
        
        #Video = Plotter.Video(f"VideoFolder/")
        #Video.make_video(0, 160, typ="gif")
        SinkDataAll = SinkParticles.SinkDataAll(self.base, feedback=self.feedback, array_length=self.array_length)
        #SinkDataAll.create_sink_particles_reduced()

        #DU.FindIMFUncerntainty()
        #print(DU.CoolingTimeSoundCrossingTimeRation(1e-21 / CS.m_p_in_cgs, 1e5, 0.1))

        #sfile = "/home/lewis/data/SEEDZ_data/production/Rarepeak_thermal/sink_snap_035"
        #self.BinaryReader = DataReader.BinaryReader(self.base, feedback=self.feedback, array_length=self.array_length)
        #self.BinaryReader.read_sink_snap(100)
        #time = self.BinaryReader.time
        #print(1/time - 1)
        #SinkPos = self.BinaryReader.extract_data("Pos")
        #print(SinkPos[:3])
        #Center = np.array([np.sum(SinkPos[:, 0]), np.sum(SinkPos[:, 1]), np.sum(SinkPos[:, 2])]) / len(SinkPos)
        #print(Center)
        #GasMass = ad[("PartType0", "Masses")].in_units("Msun").value
        #GasDensity = ad[("PartType0", "Density")].in_units("Msun/pc**3").value
        #GasVolume = GasMass / GasDensity
        #GasLength = pow(4 * GasVolume / 4 / np.pi, 1/3)

        #plot = Plotter.Plotter()
        #bins = DU.histogram_bins(GasMass, num=100)
        #plot.ax.hist(GasMass, bins=bins)
        #plot.add_details(plot.ax, xscale="log", yscale="log")
        #plot.save(f"{self.output_dir}CellMass.png")
        #snap = 128
        #sfile = f"{self.base}snapdir_{snap:03d}/snap_{snap:03d}.0.hdf5"
        #ds = yt.load(sfile)
        #ad = ds.all_data()

        #mass5 = ad[("PartType5", "Masses")].in_units("Msun").value

        #print(mass5)
        #plot = Plotter.Plotter()
        #bins = DU.histogram_bins(mass5, num=100)
        #plot.ax.hist(mass5, bins=bins)
        #plot.add_details(plot.ax, xscale="log", yscale="log")
        #plot.save(f"Random/SinkMassHistogram.png")

        #Groups = GroupsReader.Reader(self.base, self.end_snap-1)
        #Subhalos = Groups.subhalos
        #SubhaloMassType = Subhalos["SubhaloMassType"][:, 4] * 1e10 / CS.hubble_parameter
        #SubhaloMassType = SubhaloMassType[SubhaloMassType != 0]
        #print(f"{np.max(SubhaloMassType):.2e}")

        #ds = yt.load("Galaxy_Cutout_156_0.h5")
        
        #print(ds.current_redshift)
        #print(ds.field_list)

        #ds = yt.load(self.base+"snapdir_007/snap_007.0.hdf5")
        #print(ds.field_list)


        
class ParallelScripts(Scripts):
    def __init__(self, base, output_dir, feedback=False):
        self.base = base
        self.output_dir = output_dir
        self.start_snap, self.end_snap = DU.file_range(self.base)
        self.NSnap = self.end_snap - self.start_snap
        self.feedback = feedback
        self.BinaryReader = DataReader.BinaryReader(self.base, feedback=self.feedback)
        self.PickleReader = DataReader.Reader(self.base)
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def ParallelGasProjectionHalo(self):
        if self.rank == 0: print("Entering ParallelGasProjectionHalo function.", flush=True)
        #snaps = [96, 97, 98, 64]

        for i in range(self.rank, self.NSnap, self.size):
            if i <= 71: continue
            #if i not in snaps: continue
            self.GasProjectionHalo(i)

    def ParallelGasProjectionID(self):
        if self.rank == 0: print("Entering ParallelGasProjectionID function.", flush=True)

        N = self.end_snap - self.start_snap
        for i in range(self.rank, self.NSnap, self.size):
            if i < 85: continue
            self.GasProjectionID(i)
    
    def ParallelBHStellar(self):
        if self.rank == 0: print("Entering ParallelBHStellar function.", flush=True)
        BHStellarData = {}
        for i in range(self.rank, self.NSnap, self.size):
            if i <= 35: continue
            IDS, MBHMass, StellarMass = self.BHStellar(i)
        
    def ParallelDarkMatterProjection(self):
        if self.rank == 0: print("Entering ParallelDarkMatterProjectionHalo function.", flush=True)        

        for i in range(self.rank, self.NSnap, self.size):
            #if i <=61: continue
            self.DarkMatterProjection(i)

    def SurroundGasProperties(self, radius=5):
        if self.rank == 0: 
            print("Entering SurroundGasProperties function.", flush=True)
        
        SinkData = SinkParticles.SinkData(self.base)
        SinkData.filter_by_accretion(0.1, kind="self")
        IDS = SinkData.ids
        FormationSnap = SinkData.FormationSnap
        mask = FormationSnap != None
        FormationSnap = FormationSnap[mask]
        FirstSnap = np.min(FormationSnap)
        LocalGasProperties = {}
        
        for i in range(self.rank, self.NSnap, self.size):
            if i < FirstSnap: continue
            print(f"Rank {self.rank} working on snap {i}.", flush=True)
            
            sfile = f"{self.base}snapdir_{i:03d}/snap_{i:03d}.0.hdf5"
            ds = yt.load(sfile)
            ad = ds.all_data()

            SinkID = ad[("PartType5", "ParticleIDs")].value
            SinkPos = ad[("PartType5", "Coordinates")].value
            SinkVel = ad[("PartType5", "Velocities")].in_units("km/s").value
            SinkMass = ad[("PartType5", "Masses")].in_units("Msun").value

            proj = YT.YTPlotter(ds, ad)
            proj.add_metallicity()
            
            for j, ID in enumerate(IDS):
                if ID not in SinkID: continue
                if ID not in LocalGasProperties: LocalGasProperties[ID] = {}
                mask = SinkID == ID
                Pos = SinkPos[mask][0]
                Vel = SinkVel[mask][0]
                Mass = DU.unwrap(SinkMass[mask][0])

                region = proj.SelectRegion(Pos, radius, units="pc")
                GasID = region[("PartType0", "ParticleIDs")].value
                GasPos = region[("PartType0", "Coordinates")].value
                GasMass = region[("PartType0", "Masses")].in_units("Msun").value
                GasVel = region[("PartType0", "Velocities")].in_units("km/s").value
                GasMetallicity = region[("PartType0", "Metallicity")].value
                GasDensity = region[("PartType0", "Density")].in_units("g/cm**3").value
                GasTemperature = region[("PartType0", "Temperature")].in_units("K").value
            
                dPos = (GasPos - Pos)
                dVel = GasVel - Vel

                r = np.linalg.norm(dPos, axis=1)
                vrad = np.sum(dPos * dVel, axis=1) / r
                v = np.linalg.norm(dVel, axis=1)
                TotalGasMass = np.sum(GasMass)
                LocalGasProperties[ID][i] = {
                    "Mass" : Mass,
                    "Redshift": ds.current_redshift,
                    "NCells": len(GasID),
                    "GasMass": TotalGasMass,
                    "GasVelocity": np.sum(GasMass * v) / TotalGasMass,
                    "GasDensity": np.sum(GasDensity * GasMass) / TotalGasMass,
                    "GasTemperature": np.sum(GasTemperature * GasMass) / TotalGasMass,
                    "GasMetallicity": np.sum(GasMetallicity * GasMass) / TotalGasMass,
                    "GasInfalling": np.sum(GasMass[vrad < 0]),
                    "GasSoundSpeed": np.sqrt(1.4 * CS.k_B_in_cgs * np.sum(GasTemperature * GasMass) / (CS.m_p_in_cgs * TotalGasMass)),
                }
            
            print(f"Rank {self.rank} finished snap {i}.", flush=True)
        
        AllDicts = self.comm.gather(LocalGasProperties, root=0)
        if self.rank == 0:
            print("Gathered all the data on root 0.")
        if self.rank == 0:
            GlobalGasProperties = {}
            for d in AllDicts:
                for ID, snaps in d.items():
                    if ID not in GlobalGasProperties: GlobalGasProperties[ID] = {}
                    for snap, props in snaps.items():
                        GlobalGasProperties[ID][snap] = props
            
            for ID in GlobalGasProperties:
                for snap in GlobalGasProperties[ID]:
                    GlobalGasProperties[ID][snap]["Mdot"] = DU.BondiHoyleAccretion(GlobalGasProperties[ID][snap]["Mass"], GlobalGasProperties[ID][snap]["GasDensity"], GlobalGasProperties[ID][snap]["GasSoundSpeed"], GlobalGasProperties[ID][snap]["GasVelocity"])

            with open(f"{self.base}SurroundingGasProperties.pkl", "wb") as f:
                pickle.dump(GlobalGasProperties, f)

    def BHStellarCumulative(self):
        MBHID = [102573735, 154380948, 339351372, 291714062, 283304939, 204321441, 304809353, 170907394, 353631857, 306226867, 326692425, 232920095, 308660604, 287571337, 287719248, 346815834, 233552922, 377533351, 375503146, 312257629, 294538671, 319592445]

        #if self.rank == 0:
        #    MBHID = self.BHStellarHalo(self.end_snap)
        #else:
        #    MBHID = None
        #self.comm.barrier()
        #MBHID = self.comm.bcast(MBHID, root=0)
        
        BHStellarDataLocal = {}
        for i, ID in enumerate(MBHID):
            BHStellarDataLocal[ID] = {}

        for i in range(self.rank, self.NSnap, self.size):
            if i < 100: continue
            self.BinaryReader.read_sink_snap(i)
            id_binary = self.BinaryReader.extract_data("ID")
            pos_binary = self.BinaryReader.extract_data("Pos")
            type_binary = self.BinaryReader.extract_data("Type")
            mass_binary = self.BinaryReader.extract_data("StellarMass") * 1e10 / CS.hubble_parameter

            Groups = GroupsReader.Reader(self.base, i)
            Subhalos = Groups.subhalos
            SubhalosPos = Subhalos["SubhaloCM"]
            SubhaloRad = Subhalos["SubhaloHalfmassRad"]

            sfile = f"{self.base}snapdir_{i:03d}/snap_{i:03d}.0.hdf5"
            ds = yt.load(sfile)
            ad = ds.all_data()

            YTU = YT.YTUtils(ds, ad)

            Tree = KDTree(SubhalosPos)
            print("Tree made.")
            for j, ID in enumerate(MBHID):
                if ID not in id_binary: 
                    print("BH particle not in snapshot.")
                    continue
                StellarMass = 0.0
                mask = ID == id_binary
                MBHPos = pos_binary[mask][0]
                MBHMass = DU.unwrap(mass_binary[mask])
                dist, ind = Tree.query(MBHPos, k=1)
                if dist > SubhaloRad[ind]: 
                    print("BH seems to be orphan.")
                    continue
                region = YTU.SelectRegion(SubhalosPos[ind], SubhaloRad[ind], units="code_length")

                mass5 = region[("PartType5", "Masses")].in_units("Msun").value
                id5 = region[("PartType5", "ParticleIDs")].value

                if "PartType4" in ds.fields:
                    mass4 = region[("PartType4", "Masses")].in_units("Msun").value
                    StellarMass += np.sum(mass4)

                
                for k in range(len(id5)):
                    mask = id5[k] == id_binary
                    if np.any(type_binary[mask] == 2):
                        StellarMass += mass5[k]
                print(ID, i, MBHMass, StellarMass)
                BHStellarDataLocal[ID][i] = {
                    "MBHMass" : MBHMass,
                    "StellarMass": StellarMass,
                }

                if StellarMass == 0.0: StellarMass = 1e4

        AllData = self.comm.gather(BHStellarDataLocal, root=0)
        if self.rank == 0:
            print("Gathered all data on rank 0.")
        if self.rank == 0:
            BHStellarData = {}
            for data in AllData:
                for ID, snaps in data.items():
                    if ID not in BHStellarData: BHStellarData[ID] = {}
                    for snap, props in snaps.items():
                        BHStellarData[ID][snap] = props
            
            with open(f"{self.base}BHStellarData.pkl", "wb") as f:
                pickle.dump(BHStellarData, f)

