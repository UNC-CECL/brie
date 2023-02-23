============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/UNC-CECL/brie/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features. Anything tagged with "enhancement"
and "help wanted" is open to whoever wants to implement it.

Write Documentation
~~~~~~~~~~~~~~~~~~~

*brie* could always use more documentation, whether as part of the
official *brie* docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/UNC-CECL/brie/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up *brie* for local development.

1. Fork the *brie* repo on GitHub.
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/brie.git

3. Install your local copy into a conda environment. Assuming you have conda
   installed, this is how you set up your fork for local development::

    $ conda create -n brie python
    $ conda activate brie
    $ cd brie/
    $ pip install -e .

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. We use `nox`_ to automate routine maintenance tasks like running the tests,
   removing lint, etc. Install `nox`_ with *pip*::

    $ pip install nox

6. When you're done making changes, you can now run *nox* to check that the tests
   pass and that there isn't any lint::

    $ nox -s test  # run test unit tests
    $ nox -s test-notebooks  # test that the notebooks successfully
    $ nox -s test-bmi  # test the bmi
    $ nox -s lint  # find and, where possible, remove lint (black, flake8, etc.)

  To run all of the above in a single command::

    $ nox

7. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature

8. Submit a pull request through the GitHub website.

.. _nox: https://nox.thea.codes/

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in *README.rst*.
3. The pull request should work for Python 3.8 and higher. Check
   the tests pass for all supported Python versions.
4. Update *CHANGES.tst* with a brief description of what you pull request
   adds, fixes, etc.

Deploying
---------

A reminder for the maintainers on how to deploy.
Make sure all your changes are committed (including an entry in *CHANGES.rst*).
Then run::

    $ nox -s release
