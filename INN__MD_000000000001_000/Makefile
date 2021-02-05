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


# load all basic KIM make configuration
include ../Makefile.KIM_Config


# set model driver specific details
MODEL_DRIVER_NAME := INN__MD_000000000001_000
MODEL_DRIVER_KIM_FILE_TEMPLATE := INN.kim.tpl
MODEL_DRIVER_INIT_FUNCTION_NAME := model_driver_init

TORCHDIR=${TORCH_HOME}/distro/install

# APPEND to compiler option flag lists
CFLAGS   += -g -O0 -Wall -Wl,-E -finline-functions -funit-at-a-time -Winline -I${TORCHDIR}/include/ -I${TORCHDIR}/include/TH/ -I${TORCHDIR}/include/THC/
CXXFLAGS += -g -O0 -Wall -Wl,-E -finline-functions -funit-at-a-time -Winline -I${TORCHDIR}/include/ -I${TORCHDIR}/include/TH/ -I${TORCHDIR}/include/THC/
LDFLAGS  += -g -Wl,-E -L${TORCHDIR}/lib/ -L${TORCHDIR}/lib/lua/5.1/ -lluajit -lm -ldl

INN_Torch_Driver.o :	INN_Torch_Driver.lua
	luajit -b INN_Torch_Driver.lua INN_Torch_Driver.o

LOCALOBJ = INN_Torch_Driver.o INN_Main.o

LOCALCLEAN = INN_Torch_Driver.o INN_Main.o

# load remaining KIM make configuration
include $(KIM_DIR)/$(builddir)/Makefile.ModelDriver
