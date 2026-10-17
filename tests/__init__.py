"""Test package marker.

Without this file ``tests`` is only a namespace portion, and a regular
``tests`` package shipped by any installed dependency takes precedence over
it -- which breaks ``python -m unittest tests.test_x`` on a machine with a
crowded site-packages, such as Colab.
"""
