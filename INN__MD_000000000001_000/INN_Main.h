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

#define __STDC_FORMAT_MACROS
#include <stdlib.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <inttypes.h>
#include <dlfcn.h>
#include "KIM_API_C.h"
#include "KIM_API_status.h"

extern "C" {
  /*  Lua headers */
  #include "lua.h"
  #include "lualib.h"
  #include "lauxlib.h"
  /* Definition of prototypes for KIM driver functions */
  int model_driver_init(void *km,char *paramfile_names, int* nmstrlen,int* numparamfiles);
  int reinit(void *km);
  int destroy(void *km);
  int compute(void *km);
}


/* Dimension of space */
#define DIM 3
#define VDIM 6

/* Definition of model_buffer structure */
struct model_buffer {
  /* indicies for KIM API object */
  int energy_ind;
  int forces_ind;
  int particleEnergy_ind;
  int numberOfParticles_ind;
  int numberOfSpecies_ind;
  int particleSpecies_ind;
  int coordinates_ind;
  int numberContribParticles_ind;
  int boxSideLengths_ind;
  int get_neigh_ind;
  int process_dEdr_ind;
  int cutoff_ind;
  int virial_ind;
  int particleVirial_ind;
  /* model driver indices */
  int ntypes;
  int NBC;
  int HalfOrFull;
  int IterOrLoca;
  int model_index_shift;
  /* ANN object and parameters */
  double cutsq;
  double Pcutoff;    /* Cutoff of ANN potential */
  int num_typ;      /* Number of Species in the model */
  int num_rad;      /* Number of Radial Symmetry Functions */
  int num_ang;      /* Number of Angular Symmetry Functions */
  double* rad_eta;  /* Parameters for radial sym. functions */
  double* ang_eta;  /* First parameters for angular symmetry functions */
  int* lambda;   /* Second parameters for angular sym. funcs. */
  int* zeta;     /* Third parameters for angular sym. funcs. */
  double* avgd;  
  double* stdd;  
  double* pcad;  
  double avge;  
  int avga;  
  double stde;  
  //double* THcomponent;  
  //char* THobject;
  lua_State* LuaL;
}; 



/* Definition pf prototypes for LuaKIM Library Routines */
static int LuaTHsize(lua_State* L);
static int LuaTHmyid(lua_State* L);
static int LuaTHglobal(lua_State* L);
static int LuaKIMinit_index(lua_State* L);
static int LuaKIMinput_index(lua_State* L);
static int LuaKIMpca_index(lua_State* L);
static int LuaKIMparams_index(lua_State* L);
static int LuaKIMinit_newindex(lua_State* L);
static int LuaKIMinput_newindex(lua_State* L);
static int LuaKIMpca_newindex(lua_State* L);
static int LuaKIMparams_newindex(lua_State* L);
static void create_LuaKIMinit_type(lua_State* L);
static void create_LuaKIMinput_type(lua_State* L);
static void create_LuaKIMpca_type(lua_State* L);
static void create_LuaKIMparams_type(lua_State* L);
static int expose_LuaKIMinit(lua_State* L, char LuaKIMinit[]);
static int expose_LuaKIMinput(lua_State* L, double LuaKIMinput[]);
static int expose_LuaKIMpca(lua_State* L, double LuaKIMpca[]);
static int expose_LuaKIMparams(lua_State* L, int LuaKIMparams[]);
static int getLuaKIMinit(lua_State* L);
static int getLuaKIMinput(lua_State* L);
static int getLuaKIMpca(lua_State* L);
static int getLuaKIMparams(lua_State* L);
int luaopen_LuaKIMinit(lua_State* L);
int luaopen_LuaKIMinput(lua_State* L);
int luaopen_LuaKIMpca(lua_State* L);
int luaopen_LuaKIMparams(lua_State* L);

/* Utility functions */
unsigned long fact(unsigned long f);
static inline double norm(double v[3]) __attribute__((always_inline));
static inline double dotp(double a[3], double b[3]) __attribute__((always_inline));
static inline double dist(double a1[3],double a2[3]) __attribute__((always_inline));

/* Definition of descriptor (Parrinello-Behler Symmetry) functions */
static inline double fcut(double* cutoff, double Rij[3]) __attribute__((always_inline));
static inline void fcut_dr_i(double* grad, double* cutoff, double Rij[3]) __attribute__((always_inline));
static inline void fcut_dr_j(double* grad, double* cutoff, double Rij[3]) __attribute__((always_inline));
static inline double costheta(double Rij[3], double Rik[3]) __attribute__((always_inline));
static inline void acostheta_dr(double* gradi, double* gradj, double* gradk, double Rij[3], double Rik[3]) __attribute__((always_inline));
static inline double rad_symf(double rad_eta, double* cutoff, double Rij[3]) __attribute__((always_inline));
static inline void rad_symf_dr(double* grad, double rad_eta, double* cutoff, double Rij[3]) __attribute__((always_inline));
static inline double ang_symf(double ang_eta,int lambda,int zeta,double* cutoff,double Rij[3],double Rik[3],double Rjk[3]) __attribute__((always_inline));
static inline void ang_symf_dr(double gradi[3], double gradj[3], double gradk[3], double ang_eta,int lambda,int zeta,double* cutoff,double Rij[3], double Rik[3], double Rjk[3]) __attribute__((always_inline));

