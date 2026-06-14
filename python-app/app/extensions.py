"""
Shared Flask extension instances.
Only PyMongo — all auth extensions removed.
"""

from flask_pymongo import PyMongo

mongo = PyMongo()
