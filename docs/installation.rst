Installation
============

Installing using mamba
----------------------

The recommended way to install ``openff-fragmenter`` is via the ``mamba`` package manger:

.. code-block:: bash

    mamba install -c conda-forge openff-fragmenter

If you do not have Conda installed, see the `OpenFF installation guide <openff.docs:install>`_.

The ``WBOFragmenter`` computes Wiberg bond orders with GFN2-xTB by calling the ``xtb`` program, which must be on
your ``PATH``. It is available from conda-forge:

.. code-block:: bash

    mamba install -c conda-forge "xtb>=6.7.1"

All other cheminformatics (conformer generation, stereochemistry perception, depiction) is done with the RDKit.
Neither AmberTools nor the OpenEye toolkits are used.

Installing from source
----------------------

To install ``openff-fragmenter`` from source begin by cloning the repository from `github
<https://github.com/openforcefield/openff-fragmenter>`_:

.. code-block:: bash

    git clone https://github.com/openforcefield/openff-fragmenter.git
    cd fragmenter

Create a custom conda environment which contains the required dependencies and activate it:

.. code-block:: bash

    mamba env create --name fragmenter --file devtools/conda-envs/test_env.yaml
    mamba activate fragmenter

Finally, install ``openff-fragmenter`` itself:

.. code-block:: bash

    python -m pip install -e .
