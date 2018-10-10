.. _installation:

******************
Installation Guide
******************

Installation of Ollin is straight-forward with a normal python configuration.
To install ``python`` and its package manager ``pip`` see:

:python: https://realpython.com/installing-python/

:pip: https://pip.pypa.io/en/stable/installing/

Installation with pip
=====================

To install Ollin with pip package manager just run the following command in the
terminal::

  pip install ollin

Installation from source code
=============================

An alternative to pip installation, is to download the source code from GitHub_
where the code is developed and maintained.

To download the source code either clone the repository::

  git clone https://github.com/mbsantiago/ollin

or download a compressed version::

  curl -L https://github.com/mbsantiago/ollin/tarball/master -o ollin.tar.gz

Inside the downloaded directory (once decompressed) install by running the
setup script::

  cd ollin
  python setup.py install


.. _GitHub: https://github.com/mbsantiago/ollin
