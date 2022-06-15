#!/usr/bin/env python3
import sys
import numpy as np

def main():
    print("Python verison: ", sys.version)
    print("Numpy version: ", np.__version__)

if __name__ == '__main__':
    try:
        main()
    except RuntimeError:
        pass