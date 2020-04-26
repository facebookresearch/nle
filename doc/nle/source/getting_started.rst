Getting Started
===============

Dependencies
************

NLE requires ``python>=3.7``, ``libzmq``, and ``flatbuffers`` to be installed and
available when building the package.

On **MacOS** you can use brew to install ``libzmq``:

.. code-block:: bash

   $ brew install zeromq
   $ sudo wget https://raw.githubusercontent.com/zeromq/cppzmq/v4.3.0/zmq.hpp -P \
       /usr/local/include


On **Ubuntu 18.04** instead:

.. code-block:: bash

   $ sudo apt-get install libzmq3-dev


For ``flatbuffers`` we instead advise to use conda, as that is the easiest way to
pull the dependencies:

.. code-block:: bash

   $ conda create -n nledev python=3.7
   $ conda activate nledev
   $ conda install flatbuffers


Installation
************

To then install nle, simply do:

.. code-block:: bash

   $ conda activate nledev
   $ pip install nle


Optionally, one can clone the repository and install the package manually.

.. code-block:: bash

   $ git clone https://github.com/facebookresearch/nle
   $ conda activate nledev
   $ pip install .


Trying it out
*************

NLE comes with a few scripts that allow to get some environment rollouts, and
play with the action space:

.. code-block:: bash

    $ python -m nle.scripts.play
    $ python -m nle.scripts.random_agent
    $ python -m nle.scripts.play_random_games


Additionally, a `TorchBeast <https://github.com/facebookresearch/torchbeast>`_
agent is bundled in ``nle.agent`` together with a simple model to provide a
starting point for experiments:

.. code-block:: bash

    $ pip install ".[agent]"
    $ python -m nle.agent.agent --help
