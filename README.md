# openkim-INN
OpenKIM Model Driver for Implanted Neural Network Potentials.

## OpenKIM Binding
INN potentials are implemented using KIM API. The main code of the potential is the model driver and its routines in KIM API architecture. 
The model driver base to compile and integrate the routines in KIM API with version supported only <=v1.7. As of now, INN model driver can not yet be accessed through KIM ID and KIM archive.

INN_MD_XXXXXXXXXXXX_000
Link: http://www.openkim.org/

INN is developed using Torch7 which is a scientific computing framework for machine learning and is based on C, Lua/LuaJIT. The model driver code is based on KIM API's C/C++ interface. 

The core code of the INN model driver can be accessed using
git clone git@github.com/berkonat/openkim-INN

## Model Driver

## Models
The models are listed as follows:

Li-Si
KIM ID : LiSi_INN_MO_XXXXXXXXXXXX_000
Info: Developed for crystal phases of Li and Si, amorphous Si and amorphous Li-Si alloys.
License to use: If you would like to use this potential please cite the following paper as part of the license agreement. 

# Citation:

 - Berk Onat, Ekin D. Cubuk, Brad D. Malone, and Efthimios Kaxiras, "Implanted neural network potentials: Application to Li-Si alloys", Phys. Rev. B 97, 094106


# Installation
