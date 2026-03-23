import os
import pickle
import numpy as np
from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0=67.7, Om0=0.2592, Tcmb0=2.725)


#import DataReader
#import Utilities
#CS = Utilities.Constants()
#DU = Utilities.DataUtilities()

#import SinkParticles



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
 


MergerPlots()
