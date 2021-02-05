/*                                                                            */
/* CDDL HEADER START                                                          */
/*                                                                            */
/* The contents of this file are subject to the terms of the Common           */
/* Development and Distribution License Version 1.0 (the "License").          */
/*                                                                            */
/* You can obtain a copy of the license at                                    */
/* http://www.opensource.org/licenses/CDDL-1.0.  See the License for the      */
/* specific language governing permissions and limitations under the License. */
/*                                                                            */
/* When distributing Covered Code, include this CDDL HEADER in each file and  */
/* include the License file in a prominent location with the name             */
/* LICENSE.CDDL.                                                              */
/*                                                                            */
/* CDDL HEADER END                                                            */
/*                                                                            */
/* Copyright (c) 2014, Harvard University.                                    */
/* All rights reserved.                                                       */
/* All rights of this application are governed by the policies of             */
/* Harvard University, including Harvard’s “Statement of Policy in Regard to  */
/* Intellectual Property” (the “IP Policy”)                                   */
/* http://www.otd.harvard.edu/resources/policies/IP/                          */
/*                                                                            */
/* Contributors:                                                              */
/*         Berk Onat                                                          */
/*         Ekin D. Cubuk                                                      */
/*         Brad Malone                                                        */
/*         Efthimios Kaxiras                                                  */
/*                                                                            */
/*                                                                            */


KIM_API_Version := 1.7.0

Unit_Handling    := fixed
Unit_length      := A
Unit_energy      := eV
Unit_charge      := e
Unit_temperature := K
Unit_time        := ps


#######################################################################################################
PARTICLE_SPECIES:
# Symbol/name               Type                    code

SPECIES_001_NAME_STR         spec                    0
SPECIES_002_NAME_STR         spec                    1
SPECIES_003_NAME_STR         spec                    2
SPECIES_004_NAME_STR         spec                    3
SPECIES_005_NAME_STR         spec                    4
SPECIES_006_NAME_STR         spec                    5
SPECIES_007_NAME_STR         spec                    6
SPECIES_008_NAME_STR         spec                    7
SPECIES_009_NAME_STR         spec                    8
SPECIES_010_NAME_STR         spec                    9
SPECIES_011_NAME_STR         spec                   10
SPECIES_012_NAME_STR         spec                   11
SPECIES_013_NAME_STR         spec                   12
SPECIES_014_NAME_STR         spec                   13
SPECIES_015_NAME_STR         spec                   14
SPECIES_016_NAME_STR         spec                   15
SPECIES_017_NAME_STR         spec                   16
SPECIES_018_NAME_STR         spec                   17
SPECIES_019_NAME_STR         spec                   18
SPECIES_020_NAME_STR         spec                   19


#######################################################################################################
CONVENTIONS:
# Name                      Type

ZeroBasedLists              flag
Neigh_IterAccess            flag
Neigh_LocaAccess            flag
NEIGH_RVEC_F                flag
NEIGH_PURE_F                flag
#NEIGH_RVEC_H                flag
#NEIGH_PURE_H                flag
MI_OPBC_F                   flag
#MI_OPBC_H                   flag
CLUSTER                     flag


#######################################################################################################
MODEL_INPUT:
# Name                      Type         Unit                Shape              Requirements

numberOfParticles           integer      none                []
numberContributingParticles integer      none                []                 optional
numberOfSpecies             integer      none                []
particleSpecies             integer      none                [numberOfParticles]
coordinates                 double       length              [numberOfParticles,3]
boxSideLengths              double       length              [3]                optional
get_neigh                   method       none                []                 optional
neighObject                 pointer      none                []                 optional


#######################################################################################################
MODEL_OUTPUT:
# Name                      Type         Unit                Shape                  Requirements

destroy                     method       none                []                     optional
compute                     method       none                []
reinit                      method       none                []                     optional
cutoff                      double       length              []
energy                      double       energy              []                     optional
forces                      double       force               [numberOfParticles,3]  optional
particleEnergy              double       energy              [numberOfParticles]    optional
virial                      double       energy              [6]                    optional
particleVirial              double       energy              [numberOfParticles,6]  optional


#######################################################################################################
MODEL_PARAMETERS:
# Name                      Type         Unit                Shape              Requirements

# --- No published parameters ---
