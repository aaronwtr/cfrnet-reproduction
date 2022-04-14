# CFRnet reimplementation
Deeplearning course CS4240 - TU Delft.

Original paper at [proceedings.mlr.press](http://proceedings.mlr.press/v70/shalit17a/shalit17a.pdf)

A blog describing the goal of this repository can be found [here](https://medium.com/@ivo.spam/machine-learning-for-causal-inference-reproducing-individual-treatment-effect-predictions-846c76a5edef).

# Contributors
This project has been created by
- A. Wenteler <a.wenteler@student.tudelft.nl>,
- W.S. Volkers <w.s.volkers@student.tudelft.nl>,
- C.C.I.A. Chen <c.c.i.a.chen@student.tudelft.nl>,
- L. Krudde <l.krudde@student.tudelft.nl>

# Usage
Install all requirements with

`pip install requirements.txt`

Training the CFRnet can then be done by running

`python cfrnet/main.py`

An output folder with the results will be created. This folder and all other configuration parameters can be specified in `cfrnet/config.py`. 