# mri
Tools for MRI I have used for research. Currently only have GRAPPA and dual polarity distortion correction ported from MATLAB. Due to the file size constraints, the multi-channel receive data for grappa which was used to test the implementation is a reduced spatial resolution. For the same reason, the data for dual polarity encoding has already been RSOS'd over coils. The distortion correction module uses joblib, which can cause issues when running in spyder. It is better to edit scripts in spyder, then run them from the terminal.

A more modern, updated walk-through of the dual polarity correction technique can be found here: https://colab.research.google.com/drive/1K0jXAovjfUPTIRD10b6mx2PAMwEDx6IT?usp=sharing
