Astrochemistry Summer School 2024
=====

|Pythonv| |License|

.. |Pythonv| image:: https://img.shields.io/badge/Python-3.10%2C%203.11-brightgreen.svg
            :target: https://github.com/adityamarabhavi/astrochemschool2024
.. |License| image:: https://img.shields.io/badge/license-MIT-blue.svg?style=flat
            :target: https://github.com/adityamarabhavi/astrochemschool2024/blob/master/LICENSE

This repository contains a tools used in the exercise sessions of the Astrochemistry summer school 2024.
These include James Webb Space Telescope spectra, synthetic molecular spectra, models to produce synthetic spectra, tools to analyse the data, notebook exercise.


Documentation
-------------
The notebook itself contains detailed instructions.


Installation and dependencies
-----------------------------
You would need the following three packages to use the exercise notebook: 1) python 2) conda 3) git. 

Below you find instructions on how to install each package. Please read through once even if you have the packages already installed.

1) Python:
----------
First step is to make sure python is installed on your system. To check try running ``python3 --version`` on a terminal (linux and mac OS) or command prompt/Windows powershell (on Windows). If it is not installed, Windows 11 will automatically redirect you to the microsoft store page to install the latest version of Python, proceed to install. 

For Mac OS and Windows, you can also install python by directly downloading it from the official website: https://www.python.org/downloads/

On Linux, you can run the following on a terminal: ``apt-get install python``. If administrative previleges are required run ``sudo apt-get install python``.

Exit the current terminal and restart a new terminal window, check again if python is installed by running ``python3 --version``.

2) Conda:
---------
Make sure that ``conda`` is installed on your system with the command line ``conda --version``. If not installed, please follow the instructions at the bottom of the miniconda installation page 'Quick command line install': https://docs.anaconda.com/miniconda/#quick-command-line-install

Following the installation of ``conda``, you might need to do the following for it to enter the conda environment:

.. code-block:: bash

  bash
  source ~/.bash_profile
  conda init
  conda activate

You might need to point to a different profile settings than ``.bash_profile``, e.g. ``.bashrc``, ``.zshrc``, etc.

The following steps are to be done in the conda environment.

Before installing the package, it is **highly recommended to create a dedicated conda environment** to not mess up with the package versions in your base environment. This can be done easily with (replace ``astrochem2024`` by the name you want for your environment):

.. code-block:: bash

  conda create -n astrochem2024 python=3.11

Then, to activate it (assuming you named it as above):

.. code-block:: bash

  conda activate astrochem2024

If ``jupyter`` in not installed, install it via:

.. code-block:: bash

  conda install jupyter
  
The notebook depends on one major package: ``prodimopy``, which comes with its own set of dependencies from the Python ecosystem, such as ``numpy``, ``scipy``, ``matplotlib``, ``pandas``, ``astropy``, ``spectres`` and others. 

3) Git:
--------
Git is a distributed version control system that tracks versions of files. We use this to deliver the files required for this exercise. To check if git is installed on your system try ``git --version`` in the terminal window. If it is not installed, it will redirect you to the installation page, or give you the instructions on how to install it. For MacOS, installing XCode from the App Store will install git. Most Linux distributions come pre-installed with git. 

If git is not installed you can also follow the installation instructions on the git webpage: https://git-scm.com/book/en/v2/Getting-Started-Installing-Git


Once git is installed, clone the Github repository first and pip install locally (within the conda environment that you created in the previous section):

.. code-block:: bash

  # make a directory in which you want the repository to be cloned, here we name it astrochemschool2024
  mkdir astrochemschool2024
  # change the directory to the one you just made
  cd astrochemschool2024
  # clone the files from the remote git repository to your local repository using
  git clone https://github.com/adityamarabhavi/astrochemschool2024.git .
  # the following command installs all dependencies/required packages 
  pip install -e .

Install python kernel to access via jupyter (replace ``astrochem2024`` by the name you want for your environment):

.. code-block:: bash

  python -m ipykernel install --user --name astrochem2024 --display-name "astrochem2024"

You can update to the latest changes made to the repository any time, with:

.. code-block:: bash

  git pull
  pip install -e .


To open the notebook, in the right conda environment, use ``jupyter notebook``. This should automatically open a jupyter session on your browser. If not then copy the link shown on the terminal via a browser. Then navigate to your notebook to open it.

If at a later stage, you would like to use the ``prodimopy`` package and do not require the other files in this repository, you can also simply install it via:

.. code-block:: bash

  pip install prodimopy


Attribution
-----------

If the tools are useful for your science, we kindly ask you to cite:

`Arabhavi et al. (2024), <https://ui.adsabs.harvard.edu/abs/2024Sci...384.1086A/abstract>`_ for the modeling tools

`Gordon et al. (2022), <https://ui.adsabs.harvard.edu/abs/2022JQSRT.27707949G/abstract>`_ for the HITRAN spectroscopic data
