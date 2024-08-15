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


TL;DR setup guide
-----------------
.. code-block:: bash

    pip install git+https://github.com/adityamarabhavi/astrochemschool2024.git

Then launch and run the exercise.ipynb notebook, after adapting the input path to your data.


Installation and dependencies
-----------------------------
The benefits of using a Python package manager (distribution), such as (ana)conda, are many. Mainly, it brings easy and robust package management and avoids messing up with your system's default python. 


Before installing the package, it is **highly recommended to create a dedicated conda environment** to not mess up with the package versions in your base environment. This can be done easily with (replace ``astrochem2024`` by the name you want for your environment):

.. code-block:: bash

  conda create -n astrochem2024 python=3.11

Then, to activate it (assuming you named it as above):

.. code-block:: bash

  conda activate astrochem2024


The pipeline depends on one major package: ``prodimopy``, which comes with its own set of dependencies from the Python ecosystem, such as ``numpy``, ``scipy``, ``matplotlib``, ``pandas``, ``astropy``, ``spectres`` and others. 

Clone the repository first and pip install locally:

.. code-block:: bash

  # cd where you want your local repository to be located
  git clone https://github.com/adityamarabhavi/astrochemschool2024.git
  # cd in your local repository
  pip install -e .

In the latter case, you can benefit from the latest changes made to the repository any time, with:

.. code-block:: bash

  git pull

If at a later stage, you would like to use the ``prodimopy`` package and do not require the other files in this repository, you can also simply install it via:

.. code-block:: bash

  pip install prodimopy


Attribution
-----------

If the tools are useful for your science, we kindly ask you to cite:

`Arabhavi et al. (2024), <https://ui.adsabs.harvard.edu/abs/2024Sci...384.1086A/abstract>`_
