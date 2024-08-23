import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./mist.mplstyle')

def plot_elias29():
    
    df = pd.read_csv("./data/observational/Elias29.dat", sep=r'\s+',
                     names=["lambda (um)", "Flux (Jy)",
                     "Sigma (Jy)", "AOT ident."], skiprows=6)
    # convert um to wavenumbers
    df['wavenumber'] = (10**4)/(df['lambda (um)'])
    wn_min = 500
    wn_max = 4000
    df = df[(df['wavenumber'] > wn_min) & \
            (df['wavenumber'] < wn_max)].copy(deep=True)
    
    # this file is in flux units, we need to get optical depth.
    # to do that we use the formula tau = -ln(fo/fc) where tau is the
    # optical depth, fo is the observed flux, and fc is the continuum
    
    # read the continuum, taken from LIDA
    cont = pd.read_csv("./data/all_SED/2.txt", sep=r'\s+',
                       names=['wavenumber (um)', 'Flux (Jy)'])
    # we need Fo and Fc to match in wavenumber space, so interpolate
    interp_cont = np.interp(x=df['lambda (um)'],
                            xp=cont['wavenumber (um)'],
                            fp=cont['Flux (Jy)'])
    # now we can calculate optical depth
    df['tau'] = [-np.log(fo/fc) for fo, fc in \
                     zip(df['Flux (Jy)'], interp_cont)]
    # also calculate its error
    df['error_tau'] = [np.abs(sigf/f) for f, sigf in \
                       zip(df['Flux (Jy)'], df['Sigma (Jy)'])]

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(8, 4.5)
    
    ax.plot(df['wavenumber'], df['tau'])
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel("Wavenumber (1/cm)")
    ax.set_ylabel("Optical Depth");
    ax.set_title("Elias 29");
    plt.close();

    return fig