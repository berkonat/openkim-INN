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
/* Copyright (c) 2014-2016, Harvard University.                               */
/* All rights reserved.                                                       */
/* All rights of this application are governed by the policies of             */
/* Harvard University, including Harvard’s “Statement of Policy in Regard to  */
/* Intellectual Property” (the “IP Policy”)                                   */
/* For more details please see the link:                                      */
/*   http://www.otd.harvard.edu/resources/policies/IP/                        */
/*                                                                            */
/* Contributors:                                                              */
/*         Berk Onat                                                          */
/*         Ekin Dogus Cubuk                                                   */
/*         Brad Malone                                                        */
/*         Efthimios Kaxiras                                                  */
/*                                                                            */
/******************************************************************************/
/*                                                                            */
/* Version: 0.0.5  (15/11/2016) Modified to work with LAMMPS                  */
/*                                                                            */
/* KIM model driver for                                                       */
/*    Implanted Neural Network (INN) potentials.                              */
/*                                                                            */
/* ANN type is adapted from the potential defined by Behler and Parrinello    */
/* (See )                                                                     */
/*                                                                            */
/* Language: C, C++, Lua                                                       */
/* Libraries(Dependencies): Lua, LuaJIT, Torch7                               */
/*                                                                            */
/* Release: This file will be part of the kim-api.git repository.             */
/*                                                                            */
/******************************************************************************/

#include "INN_Main.h"

/* Torch7 LuaJIT and io objects*/
char* THobject;
int THsize;
int THmyid;
int THglobal=0;
int THcall=0;
double* THinput;
double* THcomponent;
//double* THd2out;
int* THparams;
size_t myid;
//FILE* f2out;
//FILE* f3out;
FILE* f4out;
 
/* Lua Interpreter*/  
lua_State* L;

/******* Routine to Call LuaJIT Object *******/
int initTHobject(lua_State* L){
  int status,ier;
  lua_register(L, "LuaTHsize", LuaTHsize);
  lua_register(L, "LuaTHmyid", LuaTHmyid);
  lua_register(L, "LuaTHglobal", LuaTHglobal);
  lua_getglobal(L, "require");
  lua_pushliteral(L, "INN_Torch_Driver");
  status = lua_pcall(L, 1, 1, 0);
  if (status) {
    fprintf(stderr, "LuaJIT Error: %s\n", lua_tostring(L, -1));
    return 1;
  }
  ier = (int)lua_tointeger(L, -1);
  if (ier<0) {
    fprintf(stderr, "LuaKIM Error code: %d\n", ier);
    return 1;
  }
  return 0;
  lua_pop(L,1);
}

int compTHobject(lua_State* L){
  int status,ier;
  luaopen_LuaKIMparams(L); 
  luaopen_LuaKIMinput(L); 
  fprintf(f4out,"Calling Lua comp_net\n");
  lua_getglobal(L, "comp_net");
  //lua_pushnumber(L, THcall);
  status = lua_pcall(L, 0, 1, 0);
  if (status) {
    fprintf(stderr, "LuaJIT Error: %s\n", lua_tostring(L, -1));
    return 1;
  }
  ier = (int)lua_tointeger(L, -1);
  if (ier<0) {
    fprintf(stderr, "LuaKIM Error code: %d\n", ier);
    return 1;
  }
  // Every compTHobject call triggers a lua_getglobal which adds a new item to Lua stack.
  lua_pop(L,1); // To prevent "LuaJIT Error: stack overflow", we remove the last item from the stack here
  return 0;
}

int freeTHobject(lua_State* L){
  int status,ier;
  fprintf(f4out,"Calling Lua trash_collect\n");
  lua_getglobal(L, "trash_collect");
  //lua_pushnumber(L, THcall);
  status = lua_pcall(L, 0, 1, 0);
  if (status) {
    fprintf(stderr, "LuaJIT Error: %s\n", lua_tostring(L, -1));
    return 1;
  }
  ier = (int)lua_tointeger(L, -1);
  if (ier<0) {
    fprintf(stderr, "LuaKIM Error code: %d\n", ier);
    return 1;
  }
  lua_pop(L,1);
  return 0;
}

int closeTH(lua_State* L){
  int status,ier;
  fprintf(f4out,"Calling Lua close_net\n");
  lua_getglobal(L, "close_net");
  //lua_pushnumber(L, THcall);
  status = lua_pcall(L, 0, 1, 0);
  if (status) {
    fprintf(stderr, "LuaJIT Error: %s\n", lua_tostring(L, -1));
    return 1;
  }
  ier = (int)lua_tointeger(L, -1);
  if (ier<0) {
    fprintf(stderr, "LuaKIM Error code: %d\n", ier);
    return 1;
  }
  lua_pop(L,1);
  return 0;
}



/******* LuaKIM Library Wrapper Functions *******/

/******* LuaKIMobject *******/
static int LuaTHsize(lua_State* L){
   lua_pushnumber(L, THsize);
   return 1;
}

static int LuaTHmyid(lua_State* L){
   lua_pushnumber(L, THmyid);
   return 1;
}

static int LuaTHglobal(lua_State* L){
   lua_pushnumber(L, THglobal);
   return 1;
}

/******* LuaKIMinit *******/

// metatable method for handling "LuaKIMinit[index]"
static int LuaKIMinit_index(lua_State* L) { 
   char** parray = (char**)luaL_checkudata(L, 1, "LuaKIMinit");
   int index = luaL_checkint(L, 2);
   lua_pushnumber(L, (*parray)[index-1]);
   return 1; 
}

// metatable method for handle "LuaKIMinit[index] = value"
static int LuaKIMinit_newindex(lua_State* L) { 
   char** parray = (char**)luaL_checkudata(L, 1, "LuaKIMinit");
   int index = luaL_checkint(L, 2);
   const char* value = luaL_checkstring(L, 3);
   (*parray)[index-1] = *value;
   return 0; 
}

// create a metatable for LuaKIMinit type
static void create_LuaKIMinit_type(lua_State* L) {
   static const struct luaL_reg LuaKIMinit[] = {
      { "__index",  LuaKIMinit_index  },
      { "__newindex",  LuaKIMinit_newindex  },
      {NULL, NULL}
   };
   luaL_newmetatable(L, "LuaKIMinit");
   luaL_openlib(L, NULL, LuaKIMinit, 0);
}

// expose the LuaKIMnit type to lua, by storing it in a userdata with the array metatable
static int expose_LuaKIMinit(lua_State* L, char LuaKIMinit[]) {
   char** parray = (char**)lua_newuserdata(L, sizeof(char**));
   *parray = LuaKIMinit;
   luaL_getmetatable(L, "LuaKIMinit");
   lua_setmetatable(L, -2);
   return 1;
}

// KIM API routine which exposes Torch7 object to Lua 
static int getLuaKIMinit(lua_State* L) { 
   return expose_LuaKIMinit( L, THobject ); 
}

int luaopen_LuaKIMinit(lua_State* L) {
   // create LuaKIMinit type
   create_LuaKIMinit_type(L);
   // make LuaKIMinit routine available to Lua
   lua_register(L, "LuaKIMinit", getLuaKIMinit);
   return 0;
}

/******* LuaKIM Input *******/

// metatable method for handling "LuaKIMinput[index]"
static int LuaKIMinput_index(lua_State* L) { 
   double** parray = (double**)luaL_checkudata(L, 1, "LuaKIMinput");
   int index = luaL_checkint(L, 2);
   lua_pushnumber(L, (*parray)[index-1]);
   return 1; 
}

// metatable method for handle "LuaKIMinput[index] = value"
static int LuaKIMinput_newindex(lua_State* L) { 
   double** parray = (double**)luaL_checkudata(L, 1, "LuaKIMinput");
   int index = luaL_checkint(L, 2);
   double value = luaL_checknumber(L, 3);
   (*parray)[index-1] = value;
   return 0; 
}

// create a metatable for LuaKIMinput type
static void create_LuaKIMinput_type(lua_State* L) {
   static const struct luaL_reg LuaKIMinput[] = {
      { "__index",  LuaKIMinput_index  },
      { "__newindex",  LuaKIMinput_newindex  },
      {NULL, NULL}
   };
   luaL_newmetatable(L, "LuaKIMinput");
   luaL_openlib(L, NULL, LuaKIMinput, 0);
}

// expose the LuaKIMinput type to lua, by storing it in a userdata with the array metatable
static int expose_LuaKIMinput(lua_State* L, double LuaKIMinput[]) {
   double** parray = (double**)lua_newuserdata(L, sizeof(double**));
   *parray = LuaKIMinput;
   luaL_getmetatable(L, "LuaKIMinput");
   lua_setmetatable(L, -2);
   return 1;
}

// KIM API routine which exposes Torch7 object to Lua 
static int getLuaKIMinput(lua_State* L) { 
   return expose_LuaKIMinput( L, THinput ); 
}

int luaopen_LuaKIMinput(lua_State* L) {
   // create LuaKIMinput type
   create_LuaKIMinput_type(L);
   // make LuaKIMinput routine available to Lua
   lua_register(L, "LuaKIMinput", getLuaKIMinput);
   return 0;
}

/******* LuaKIM PCA *******/

// metatable method for handling "LuaKIMpca[index]"
static int LuaKIMpca_index(lua_State* L) { 
   double** parray = (double**)luaL_checkudata(L, 1, "LuaKIMpca");
   int index = luaL_checkint(L, 2);
   lua_pushnumber(L, (*parray)[index-1]);
   return 1; 
}

// metatable method for handle "LuaKIMpca[index] = value"
static int LuaKIMpca_newindex(lua_State* L) { 
   double** parray = (double**)luaL_checkudata(L, 1, "LuaKIMpca");
   int index = luaL_checkint(L, 2);
   double value = luaL_checknumber(L, 3);
   (*parray)[index-1] = value;
   return 0; 
}

// create a metatable for LuaKIMpca type
static void create_LuaKIMpca_type(lua_State* L) {
   static const struct luaL_reg LuaKIMpca[] = {
      { "__index",  LuaKIMpca_index  },
      { "__newindex",  LuaKIMpca_newindex  },
      {NULL, NULL}
   };
   luaL_newmetatable(L, "LuaKIMpca");
   luaL_openlib(L, NULL, LuaKIMpca, 0);
}

// expose the LuaKIMpca type to lua, by storing it in a userdata with the array metatable
static int expose_LuaKIMpca(lua_State* L, double LuaKIMpca[]) {
   double** parray = (double**)lua_newuserdata(L, sizeof(double**));
   *parray = LuaKIMpca;
   luaL_getmetatable(L, "LuaKIMpca");
   lua_setmetatable(L, -2);
   return 1;
}

// KIM API routine which exposes Torch7 object to Lua 
static int getLuaKIMpca(lua_State* L) { 
   return expose_LuaKIMpca( L, THcomponent ); 
}

int luaopen_LuaKIMpca(lua_State* L) {
   // create LuaKIMpca type
   create_LuaKIMpca_type(L);
   // make LuaKIMpca routine available to Lua
   lua_register(L, "LuaKIMpca", getLuaKIMpca);
   return 0;
}



/******* LuaKIM Second Derivative Outputs *******/
/*
// metatable method for handling "LuaKIMd2eout[index]"
static int LuaKIMd2eout_index(lua_State* L) { 
   double** parray = luaL_checkudata(L, 1, "LuaKIMd2eout");
   int index = luaL_checkint(L, 2);
   lua_pushnumber(L, (*parray)[index-1]);
   return 1; 
}

// metatable method for handle "LuaKIMd2eout[index] = value"
static int LuaKIMd2eout_newindex(lua_State* L) { 
   double** parray = luaL_checkudata(L, 1, "LuaKIMd2eout");
   int index = luaL_checkint(L, 2);
   double value = luaL_checknumber(L, 3);
   (*parray)[index-1] = value;
   return 0; 
}

// create a metatable for LuaKIMdeout type
static void create_LuaKIMd2eout_type(lua_State* L) {
   static const struct luaL_reg LuaKIMd2eout[] = {
      { "__index",  LuaKIMd2eout_index  },
      { "__newindex",  LuaKIMd2eout_newindex  },
      {NULL, NULL}
   };
   luaL_newmetatable(L, "LuaKIMd2eout");
   luaL_openlib(L, NULL, LuaKIMd2eout, 0);
}

// expose the LuaKIMd2eout type to lua, by storing it in a userdata with the array metatable
static int expose_LuaKIMd2eout(lua_State* L, double LuaKIMd2eout[]) {
   double** parray = lua_newuserdata(L, sizeof(double**));
   *parray = LuaKIMd2eout;
   luaL_getmetatable(L, "LuaKIMd2eout");
   lua_setmetatable(L, -2);
   return 1;
}

// KIM API routine which exposes Torch7 object to Lua 
static int getLuaKIMd2eout(lua_State* L) { 
   return expose_LuaKIMd2eout( L, THd2eout ); 
}

int luaopen_LuaKIMd2eout(lua_State* L) {
   // create LuaKIMd2eout type
   create_LuaKIMd2eout_type(L);
   // make LuaKIMd2eout routine available to Lua
   lua_register(L, "LuaKIMd2eout", getLuaKIMd2eout);
   return 0;
}
*/

/******* LuaKIM Params *******/

// metatable method for handling "LuaKIMparams[index]"
static int LuaKIMparams_index(lua_State* L) { 
   int** parray = (int**)luaL_checkudata(L, 1, "LuaKIMparams");
   int index = luaL_checkint(L, 2);
   lua_pushnumber(L, (*parray)[index-1]);
   return 1; 
}

// metatable method for handle "LuaKIMparams[index] = value"
static int LuaKIMparams_newindex(lua_State* L) { 
   int** parray = (int**)luaL_checkudata(L, 1, "LuaKIMparams");
   int index = luaL_checkint(L, 2);
   int value = luaL_checkint(L, 3);
   (*parray)[index-1] = value;
   return 0; 
}

// create a metatable for LuaKIMparams type
static void create_LuaKIMparams_type(lua_State* L) {
   static const struct luaL_reg LuaKIMparams[] = {
      { "__index",  LuaKIMparams_index  },
      { "__newindex",  LuaKIMparams_newindex  },
      {NULL, NULL}
   };
   luaL_newmetatable(L, "LuaKIMparams");
   luaL_openlib(L, NULL, LuaKIMparams, 0);
}

// expose the LuaKIMparams type to lua, by storing it in a userdata with the array metatable
static int expose_LuaKIMparams(lua_State* L, int LuaKIMparams[]) {
   int** parray = (int**)lua_newuserdata(L, sizeof(int**));
   *parray = LuaKIMparams;
   luaL_getmetatable(L, "LuaKIMparams");
   lua_setmetatable(L, -2);
   return 1;
}

// KIM API routine which exposes Torch7 object to Lua 
static int getLuaKIMparams(lua_State* L) { 
   return expose_LuaKIMparams( L, THparams ); 
}

int luaopen_LuaKIMparams(lua_State* L) {
   // create LuaKIMparams type
   create_LuaKIMparams_type(L);
   // make LuaKIMparams routine available to Lua
   lua_register(L, "LuaKIMparams", getLuaKIMparams);
   return 0;
}

/******* Utility Functions ******/
unsigned long fact(unsigned long f){
  if (f<2) {
    return 1;
  } else {
    return f*fact(f-1);
  }
}

static inline double norm(double v[3]){
  return sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2]);
}

static inline double dotp(double a[3], double b[3]){
  return (a[0]*b[0]+a[1]*b[1]+a[2]*b[2]);
}

static inline double dist(double a1[3],double a2[3]){
  double value;
  value = sqrt((a2[0]-a1[0])*(a2[0]-a1[0])+(a2[1]-a1[1])*(a2[1]-a1[1])+(a2[2]-a1[2])*(a2[2]-a1[2]));
  return value;
}

/******* Descriptor Functions *******/

/* Calculate symmetry functions G(r) */
static inline double fcut(double* cutoff, double Rij[3]){
  double value=0.0,dij=0.0;
  double pi=3.1415926535897932384626433832795028841971694;
  dij=norm(Rij);
  if (dij < (*cutoff)) {
    value = 0.5*(cos(pi*dij/(*cutoff))+1.0); 
  } else {
    value = 0.0;
  }
  return value; 
}

static inline void fcut_dr_i(double grad[3], double* cutoff, double Rij[3]){
  int i;
  double value,dij;
  double pi=3.1415926535897932384626433832795028841971694;
  dij=norm(Rij);
  if (dij < *cutoff) {
    value = -(0.5*pi/( *cutoff ))*sin(pi*dij/( *cutoff )); 
  } else {
    value = 0.0; 
  }
  for(i=0;i<3;i++){
    grad[i]=value*Rij[i]/dij;
  } 
}

static inline void fcut_dr_j(double grad[3], double* cutoff, double Rij[3]){
  int i;
  double value,dij;
  double pi=3.1415926535897932384626433832795028841971694;
  dij=norm(Rij);
  if (dij < *cutoff) {
    value = -(0.5*pi/( *cutoff ))*sin(pi*dij/( *cutoff )); 
  } else {
    value = 0.0; 
  }
  for(i=0;i<3;i++){
    grad[i]=-value*Rij[i]/dij;
  } 
}

static inline double costheta(double Rij[3], double Rik[3]){
  double value=0.0,normval;
  double norm1,norm2;
  int i;
  norm1=norm(Rij);
  norm2=norm(Rik);
  if (norm1<=0.00000001 || norm2<=0.00000001) { fprintf(f4out,"ZERO zero norm1: %f  norm2: %f \n",norm1,norm2); exit(0); }
  normval=1.0/(norm1*norm2);
  for(i=0;i<3;i++){
     value+=Rij[i]*Rik[i];
  } 
  return value*normval;
}

static inline void acostheta_dr(double gradi[3], double gradj[3], double gradk[3], double Rij[3], double Rik[3]){
  double value=0.0,dij,dik,normval;
  double norm1,norm2;
  int i;
  dij=norm(Rij);
  dik=norm(Rik);
  norm1=1.0/(dij*dij);
  norm2=1.0/(dik*dik);
  normval=1.0/(dij*dik);
  for(i=0;i<3;i++){
     value+=Rij[i]*Rik[i];
  }
  for(i=0;i<3;i++){
     gradi[i] = ( ( Rij[i] + Rik[i] ) - value*( Rij[i]*norm1 + Rik[i]*norm2 ) )*normval;
     gradj[i] = ( value*Rij[i]*norm1 - Rik[i] ) * normval;
     gradk[i] = ( value*Rik[i]*norm2 - Rij[i] ) * normval;
  } 
}


static inline double rad_symf(double rad_eta, double* cutoff, double Rij[3]){
  double dij;
  dij=norm(Rij);
  return exp(-rad_eta*dij*dij)*fcut(cutoff,Rij); 
}

static inline void rad_symf_dr(double grad[3], double rad_eta, double* cutoff, double Rij[3]){
  int i;
  double dij,fcut_grad[3];
  dij=norm(Rij);
  fcut_dr_i(fcut_grad,cutoff,Rij); 
  for(i=0;i<3;i++){
     grad[i]=(-2.0*rad_eta*Rij[i]*exp(-rad_eta*dij*dij)*fcut(cutoff,Rij)) + (exp(-rad_eta*dij*dij)*fcut_grad[i]); 
  }
}

static inline double ang_symf(double ang_eta,int lambda,int zeta,double* cutoff,double Rij[3],double Rik[3],double Rjk[3]){
  double dij,djk,dik;
  double fcutall,power1,power2,value;
  dij=norm(Rij);
  djk=norm(Rjk);
  dik=norm(Rik);
  fcutall=fcut(cutoff,Rik)*fcut(cutoff,Rij)*fcut(cutoff,Rjk);
  power1=pow((1.0+lambda*costheta(Rij,Rik)),(double)zeta);
  power2=pow(2.0,(double)(1.0-zeta));
  value=exp(-ang_eta*(dik*dik+dij*dij+djk*djk))*fcutall*power1*power2;
  return value; 
}


static inline void ang_symf_dr(double gradi[3], double gradj[3], double gradk[3], double ang_eta,int lambda,int zeta,double* cutoff,double Rij[3], double Rik[3], double Rjk[3]){
  int i;
  double dij,djk,dik;
  double thetagradi[3],thetagradj[3],thetagradk[3];
  double expon,powksi,power2,powksi1;
  double fcut_gradi_ij[3],fcut_gradi_ik[3];
  double fcut_gradj_ij[3],fcut_gradj_jk[3];
  double fcut_gradk_ik[3],fcut_gradk_jk[3];
  double fc_ij,fc_ik,fc_jk;
  double costhetaval;
  dij=norm(Rij);
  djk=norm(Rjk);
  dik=norm(Rik);
  costhetaval=costheta(Rij,Rik); 
  acostheta_dr(thetagradi,thetagradj,thetagradk,Rij,Rik);
  expon=exp(-ang_eta*(dik*dik+dij*dij+djk*djk));
  powksi=pow(1.0+lambda*costhetaval,zeta);
  power2=pow(2.0,(double)(1.0-zeta));
  powksi1=pow(1.0+lambda*costhetaval,zeta-1.0);
  fcut_dr_i(fcut_gradi_ij,cutoff,Rij); 
  fcut_dr_i(fcut_gradi_ik,cutoff,Rik); 
  fcut_dr_i(fcut_gradj_jk,cutoff,Rjk); 
  fcut_dr_j(fcut_gradj_ij,cutoff,Rij); 
  fcut_dr_j(fcut_gradk_ik,cutoff,Rik); 
  fcut_dr_j(fcut_gradk_jk,cutoff,Rjk); 
  fc_ij=fcut(cutoff,Rij);
  fc_ik=fcut(cutoff,Rik);
  fc_jk=fcut(cutoff,Rjk);
  for(i=0;i<3;i++){
      gradi[i] = power2 * ( ( zeta*lambda*thetagradi[i]*powksi1 * expon*fc_ij*fc_ik*fc_jk ) +  
                           powksi * ( ( -ang_eta*2.0*( Rij[i] + Rik[i] )*expon * fc_ij*fc_ik*fc_jk ) + 
                                      ( expon * ( fcut_gradi_ij[i] * fc_ik*fc_jk + fc_ij * fcut_gradi_ik[i]*fc_jk  ) )
                                      //( expon * ( fcut_grad_ij[i] * fc_ik*fc_jk + fc_ij * ( fcut_grad_ik[i]*fc_jk +  fc_ik*fcut_grad_jk[i] ) ) )
                                    )
                         );
      gradj[i] = power2 * ( ( zeta*lambda*thetagradj[i]*powksi1 * expon*fc_ij*fc_ik*fc_jk ) +  
                           powksi * ( ( -ang_eta*2.0*( -Rij[i] + Rjk[i] )*expon * fc_ij*fc_ik*fc_jk ) + 
                                      ( expon * ( fcut_gradj_ij[i] * fc_ik*fc_jk + fc_ij * fc_ik*fcut_gradj_jk[i] ) )
                                    )
                         );
      gradk[i] = power2 * ( ( zeta*lambda*thetagradk[i]*powksi1 * expon*fc_ij*fc_ik*fc_jk ) +  
                           powksi * ( ( -ang_eta*2.0*( -Rik[i] - Rjk[i] )*expon * fc_ij*fc_ik*fc_jk ) + 
                                      ( expon * ( fc_ij * ( fcut_gradk_ik[i]*fc_jk +  fc_ik*fcut_gradk_jk[i] ) ) )
                                    )
                         );
  }
}


/******* KIM Model Driver Functions ******/

int reinit(void *km)
{
  intptr_t *pkim = *((intptr_t **) km);
  model_buffer *buffer;
  int ier;
  double cutoff;

  THglobal++;
  /* get model buffer from KIM object */
  buffer = (model_buffer *) KIM_API_get_model_buffer(pkim, &ier);
  if (KIM_STATUS_OK > ier) {
     ier = KIM_STATUS_FAIL;
     KIM_API_report_error(__LINE__, __FILE__, "KIM_API_get_model_buffer", ier);
     return ier;
  }
  cutoff = buffer->Pcutoff;
  buffer->Pcutoff = cutoff;

  return KIM_STATUS_OK;
}

int destroy(void *km)
{
  //intptr_t *pkim = *((intptr_t **) km);
  //model_buffer *buffer;
  //int ier;

  /* get model buffer from KIM object */
  //buffer = (model_buffer *) KIM_API_get_model_buffer(pkim, &ier);
  //if (KIM_STATUS_OK > ier) {
  //   ier = KIM_STATUS_FAIL;
  //   KIM_API_report_error(__LINE__, __FILE__, "KIM_API_get_model_buffer", ier);
  //   return ier;
  //}
  
  /* free Lua State (Calls corresponding garbage collection metamethods) */
  //L = buffer->LuaL;
//  closeTH(L);
//  lua_close(L);

  /* free THobject */
  //free(THobject);
  //free(THparams);
  //free(THinput);
  //free(THcomponent);

  /* free buffer */
  //free(buffer);

  return KIM_STATUS_OK;
}


int compute(void *km) {
  
  intptr_t *pkim = *((intptr_t **) km);
  
  /* pointer to model buffer */
  model_buffer *buffer;
  
  /* flags for available computes */
  int   comp_energy;
  int   comp_force;
  int   comp_particleEnergy;
  int   comp_virial;
  int   comp_particleVirial;
  
  /* pointers to objects in the KIM API */
  double* boxSideLengths;
  double* cutoff;
  double* coords;
  double* energy;
  double* forces;
  double* virial;
  double* particleVirial;
  int* numContrib;
  int* nParts;
  int* numberOfSpecies;
  double* particleEnergy;
  int* particleSpecies;
  
  /* KIM model parameters */
  int NBC;
  int HalfOrFull;
  int IterOrLoca;
  int model_index_shift;
  int zero = 0;
  int one = 1;
  int request;
  
  /* ANN potential parameters */
  double modcutoff;
  double* cutsq;
  int  num_check;
  int  THnum_check;
  int* num_atom_typs;
  int* all_atom_typs;
  int  num_typ;
  int  num_rad;
  int  num_ang;
  double* rad_eta;
  double* ang_eta;
  int* lambda;
  int* zeta;
  //double* avgdata;
  //double* stddata;
  //double* pcadata;
  double avgeng;
  double atomavgeng;
  int avgatom;
  double stdeng;
 
  /* Variables for neighbor and TH mapping calculations */ 
  double Rsqij;
  double Rsqik;
  double Rsqjk;
  //double d2Edr;
  double Rij[DIM];
  double Rik[DIM];
  double Rki[DIM];
  double Rjk[DIM];
  //double Rkj[DIM];
  double Rji[DIM];
  double Rgradi[DIM];
  double Rgradj[DIM];
  double Rgradk[DIM];
  int ier;
  int THier;
  int THindexShift;
  int elementflag;
  int i;
  int j;
  int jj;
  int kk;
  int k; // index of neighbor for particle j
  int d; // index of DIM
  int t; // type index
  int t1t; // type index for pair compare
  int t2t; // type index for pair compare
  int* specieCount; // Counter for specie index
  unsigned long* specieOffset; // Offset for specie index
  unsigned long* specieBackOffset; // Offset for specie index
  int num_gsum;  // number of rad sym. func.
  int num_gsumAng;  // number of ang sym. func.
  int num_syms;
  int symi; // index for sym. funcs.
  //int symj; // index for sym. funcs.
  unsigned long radoffset; // index for rad. sym. funcs. in the input array
  //unsigned long pcaradoffset; // index for rad. sym. funcs. in the input array
  unsigned long angoffseti; // index for ang. sym. funcs. in the input array
  //unsigned long pcaangoffseti; // index for ang. sym. funcs. in the input array
  unsigned long angoffsetj; // index for ang. sym. funcs. in the input array
  //unsigned long pcaangoffsetj; // index for ang. sym. funcs. in the input array
  //int pcashifti;
  //int pcaoffseti;
  //int pcashiftj;
  //int pcaoffsetj;
  unsigned long comb; // number of combinations for ang. sym. funcs.
  unsigned long* THPartListOffset;
  unsigned long* THPartBackOffset;
  int  currentPart;
  int  currentPartCount;
  int  partCounted;
  int* currentPartList;
  int* neighListOfCurrentPart;
  int* neighListOfParts;
  unsigned long* neighListOffset;
  unsigned long  neighCount;
  //unsigned long  neighCountI;
  unsigned long  neighCountJ;
  double* Rij_list;
  double* Rij_list_all;
  double  ff;
  double  ffi[DIM];
  double  ffj[DIM];
  double  ffk[DIM];
  double fmm;
  double  stress[DIM][DIM];
  int  numberContrib;
  int  THnumContrib;
  int  numOfPartNeigh;
  //int  numOfPartNeighI;
  int  numOfPartNeighJ;
  int* numOfPartNeighList;
  int  checkstate;

  /* Lua specific variables */
  double luatime;
  double luagarbage;
 
  /* Error message variable */
  char errortext[255];
 
  /* get buffer from KIM object */
  buffer = (model_buffer *) KIM_API_get_model_buffer(pkim, &ier);
  if (KIM_STATUS_OK > ier) {
     KIM_API_report_error(__LINE__, __FILE__, "KIM_API_get_model_buffer", ier);
     return ier;
  }
  
  /* check to see if we have been asked to compute the energy, forces, and particleEnergies */
  /* *INDENT-OFF* */
  KIM_API_getm_compute_by_index(pkim, &ier, 5 * 3,
                buffer->energy_ind,             &comp_energy,         1,
                buffer->forces_ind,             &comp_force,          1,
                buffer->particleEnergy_ind,     &comp_particleEnergy, 1,
                buffer->virial_ind,             &comp_virial,         1,
                buffer->particleVirial_ind,     &comp_particleVirial, 1);
  /* *INDENT-ON* */
  if (KIM_STATUS_OK > ier) {
    KIM_API_report_error(__LINE__, __FILE__, "KIM_API_getm_compute_by_index", ier);
    return ier;
  }

  /* unpack info from the buffer */
  modcutoff= buffer->Pcutoff; 
  cutoff = &modcutoff;
  NBC = buffer->NBC;
  HalfOrFull = buffer->HalfOrFull;
  IterOrLoca = buffer->IterOrLoca;
  model_index_shift = buffer->model_index_shift;
  cutsq = &(buffer->cutsq);
  num_typ = buffer->num_typ;
  num_rad = buffer->num_rad;
  rad_eta = buffer->rad_eta; 
  num_ang = buffer->num_ang; 
  ang_eta = buffer->ang_eta; 
  lambda = buffer->lambda; 
  zeta = buffer->zeta; 
  //avgdata = buffer->avgd; 
  //stddata = buffer->stdd; 
  //pcadata = buffer->pcad; 
  avgeng = buffer->avge; 
  avgatom = buffer->avga; 
  stdeng = buffer->stde; 
  //THobject = buffer->THobject; 
  L = buffer->LuaL;
  
  atomavgeng=avgeng/avgatom;

  KIM_API_getm_data_by_index(
      pkim, &ier, 11*3,
      buffer->boxSideLengths_ind,         &boxSideLengths,     (NBC==2),
      buffer->coordinates_ind,            &coords,             1,
      buffer->energy_ind,                 &energy,             comp_energy,
      buffer->forces_ind,                 &forces,             comp_force,
      buffer->virial_ind,                 &virial,             comp_virial,
      buffer->particleVirial_ind,         &particleVirial,     comp_particleVirial,
      buffer->numberContribParticles_ind, &numContrib,         (HalfOrFull==1),
      buffer->numberOfParticles_ind,      &nParts,             1,
      buffer->numberOfSpecies_ind,        &numberOfSpecies,    1,
      buffer->particleEnergy_ind,         &particleEnergy,     comp_particleEnergy,
      buffer->particleSpecies_ind,        &particleSpecies,    1);
  if (KIM_STATUS_OK > ier) {
    KIM_API_report_error(__LINE__, __FILE__, "KIM_API_getm_data_by_index", ier);
    return ier; 
  }
    

  if (HalfOrFull == 1) { /* Half == 1 */
    if (3 != NBC) {
      /* non-CLUSTER cases */
      numberContrib = *numContrib; 
    } else {
      /* CLUSTER cases */
      numberContrib = *nParts; 
    } 
  } else { /* Full == 2 */
    numberContrib = *nParts; 
  }
  fprintf(f4out,"numberContrib:%d\n",numberContrib);

  num_atom_typs = (int*) malloc(num_typ*sizeof(int));
  all_atom_typs = (int*) malloc(num_typ*sizeof(int));
  for(t = 0; t < num_typ; t++) {
      num_atom_typs[t]=0;
      all_atom_typs[t]=0;
  }
   
  /* Initialize neighbor handling for Iterator mode */

  if (1 == IterOrLoca) {
    ier = KIM_API_get_neigh(pkim, zero, zero, &currentPart, &numOfPartNeigh, &neighListOfCurrentPart, &Rij_list);
    // check for successful initialization
    if (KIM_STATUS_OK > ier) {
      KIM_API_report_error(__LINE__, __FILE__, "get_neigh error at iterator init.", ier);
      return ier; 
    } 
  }

  // Get neighbor list of all nParts atoms
  currentPartCount=0;
  i = 0;
  checkstate = 1;
  while( checkstate ) {
    // Set up neighbor list for next part for all NBC methods
    if (1 == IterOrLoca) {
      // ITERATOR mode
      ier = KIM_API_get_neigh(pkim, zero, one, &currentPart, &numOfPartNeigh, &neighListOfCurrentPart, &Rij_list);
      if (KIM_STATUS_NEIGH_ITER_PAST_END == ier) {
        // the end of the list, terminate loop
        checkstate = 0;
        break; 
      }
      if (KIM_STATUS_OK > ier) {
        // some sort of problem, exit
        KIM_API_report_error(__LINE__, __FILE__, "get_neigh error at iterator cycle.", ier);
        return ier; 
      }
      i = currentPart + model_index_shift; 
    } else {
      if (numberContrib <= i) {
        // incremented past end of list, terminate loop
        checkstate = 0;
        break; 
      }
      if (3 != NBC) {
        // LOCATOR mode
        request = i - model_index_shift;
        ier = KIM_API_get_neigh(pkim, one, request, &currentPart, &numOfPartNeigh, &neighListOfCurrentPart, &Rij_list);
        if (KIM_STATUS_OK > ier) {
          // some sort of problem, exit
          KIM_API_report_error(__LINE__, __FILE__, "get_neigh error at locator cycle.", ier);
          return ier; 
        } 
      } 
      i++;
    } // If IterOrLoca Ends
    currentPartCount++;
  } // infinite while loop (terminated by break statements above)
  THnumContrib=currentPartCount;
  fprintf(f4out,"THnumContrib:%d\n",THnumContrib);


  /* Calculated the number of atoms for each specie */
  ier = KIM_STATUS_FAIL; /* assume an error */
  for (i = 0; i < numberContrib; ++i) {
    elementflag=0;
    if ( i < THnumContrib ) {
     for(t = 0; t < num_typ; t++) {
       if(particleSpecies[i]==t){
          num_atom_typs[t]++;
          elementflag=1;
       }
     }
    }
    for(t = 0; t < num_typ; t++) {
       if(particleSpecies[i]==t){
          all_atom_typs[t]++;
          elementflag=1;
       }
    }
    /* Check to be sure that the species are correct */
    if ( elementflag != 1 ) {
      KIM_API_report_error(__LINE__, __FILE__, "Unexpected specie detected", ier);
      return ier; 
    }
  }
  num_check=0;
  THnum_check=0;
  for(t = 0; t < num_typ; t++) {
      num_check+=all_atom_typs[t];
      THnum_check+=num_atom_typs[t];
  }
  if ( ( numberContrib != num_check ) && ( THnum_check != THnumContrib ) ) {
    KIM_API_report_error(__LINE__, __FILE__, "Unexpected number of species detected", ier);
    return ier; 
  }
  ier = KIM_STATUS_OK;  
  
  /* Initialize Symmetry function arrays */
  /* Initialize Lua input & output arrays */
  if(num_typ<2) {
    num_gsum=num_typ*num_rad;
    num_gsumAng=num_typ*num_ang;
    num_syms=num_gsum+num_gsumAng;
  } else {
    comb=fact((unsigned long)num_typ)/(2*fact((unsigned long)(num_typ-2)));
    num_gsum=num_typ*num_rad;
    num_gsumAng=(num_typ+comb)*num_ang;
    num_syms=num_gsum+num_gsumAng;
  }
  THinput = (double*) realloc(THinput,(THnumContrib*(num_syms+1)+3)*sizeof(double));
  if (THinput==NULL) {
    ier = KIM_STATUS_FAIL;
    KIM_API_report_error(__LINE__, __FILE__, "Can not allocate memory for Torch input array", ier);
    return ier; 
  //} else {
  //  THinput = THptr;
  }
  specieCount = (int*) malloc(num_typ*sizeof(int));
  specieOffset = (unsigned long*) malloc((num_typ+1)*sizeof(unsigned long));
  specieBackOffset = (unsigned long*) malloc((num_typ+1)*sizeof(unsigned long));
  THparams[0]=num_typ;
  THparams[1]=num_syms;
  /* Index offset for first type of atoms is always zero.
     There is always an atom in the list */
  specieOffset[0]=0;
  specieBackOffset[0]=0;
  for (t=0;t<num_typ;t++){
        specieCount[t]=0;
        THparams[t+2]=num_atom_typs[t]; 
        if (t<1) { // Second specie's offset
          specieOffset[t+1]=num_syms*num_atom_typs[t];
          specieBackOffset[t+1]=num_atom_typs[t];
        } else { // Shift others accordingly
          specieOffset[t+1]=specieOffset[t]+num_syms*num_atom_typs[t];
          specieBackOffset[t+1]=specieBackOffset[t]+num_atom_typs[t];
        }
  }
  /* Write zeros only at the padding part of array */
  if (comp_energy || comp_force || comp_particleEnergy || comp_virial) {
    neighCount=0;
    for (t=0;t<THnumContrib;t++){
        for (symi=0;symi<num_syms+1;symi++){
             THinput[neighCount]=0.0;
             neighCount++;
        }
    }
    THinput[neighCount]=0.0;
    THinput[neighCount+1]=0.0;
    THinput[neighCount+2]=0.0;
  }
  /* initialize potential energies, forces, and virial term */
  if (comp_particleEnergy)
      for (i = 0; i <(*nParts); ++i) particleEnergy[i] = 0.0; 
  if (comp_energy) 
     *energy = 0.0; 

  if (comp_force) {
    //fforce = (double*) malloc(num_syms*sizeof(double)); 
    for (i = 0; i <(*nParts)*DIM; i++) forces[i] = 0.0; 
    //for (i = 0; i <num_syms; ++i) fforce[i] = 0.0; 
  }
  
  if (comp_virial) {
    for (i = 0; i <VDIM; ++i) virial[i] = 0.0; 
    for (i = 0; i <DIM; ++i) 
        for (j = 0; j <DIM; ++j) 
            stress[i][j] = 0.0; 
  }
  if (comp_particleVirial)
    for (i = 0; i < (*nParts)*VDIM; ++i) particleVirial[i] = 0.0;
  
  
  neighListOfParts = (int*) malloc(2*(*nParts)*(*nParts)*sizeof(int)); 
  if (0 == NBC) {
     Rij_list_all = (double*) malloc(2*(*nParts)*(*nParts)*DIM*sizeof(double)); 
  } else {
     Rij_list_all = NULL;
  }
  numOfPartNeighList = (int*) malloc((*nParts)*sizeof(int)); 
  currentPartList = (int*) malloc((*nParts)*sizeof(int)); 
  neighListOffset = (unsigned long*) malloc((*nParts)*sizeof(unsigned long)); 
  THPartListOffset = (unsigned long*) malloc(THnumContrib*sizeof(unsigned long)); 
  THPartBackOffset = (unsigned long*) malloc(THnumContrib*sizeof(unsigned long)); 
  /* Initialize neighbor handling for CLUSTER NBC */
  if (3 == NBC) {
     /* CLUSTER */
     neighListOfCurrentPart = (int*) malloc((*nParts)*sizeof(int)); 
     Rij_list=NULL;
  }
  
  /* Initialize neighbor handling for Iterator mode */

  if (1 == IterOrLoca) {
    ier = KIM_API_get_neigh(pkim, zero, zero, &currentPart, &numOfPartNeigh, &neighListOfCurrentPart, &Rij_list);
    /* check for successful initialization */
    if (KIM_STATUS_OK > ier) {
      KIM_API_report_error(__LINE__, __FILE__, "get_neigh error at iterator init.", ier);
      return ier; 
    } 
  }
  

  /* Get neighbor list of all nParts atoms */
  neighCount=0;
  currentPartCount=0;
  i = 0;
  checkstate = 1;
  while( checkstate ) {
    /* Set up neighbor list for next part for all NBC methods */
    if (1 == IterOrLoca) {
      /* ITERATOR mode */
      ier = KIM_API_get_neigh(pkim, zero, one, &currentPart, &numOfPartNeigh, &neighListOfCurrentPart, &Rij_list);
      if (KIM_STATUS_NEIGH_ITER_PAST_END == ier) {
        /* the end of the list, terminate loop */
        checkstate = 0;
        break; 
      }
      if (KIM_STATUS_OK > ier) {
        /* some sort of problem, exit */
        KIM_API_report_error(__LINE__, __FILE__, "get_neigh error at iterator cycle.", ier);
        return ier; 
      }
      i = currentPart + model_index_shift; 
    } else {
      if (numberContrib <= i) {
        /* incremented past end of list, terminate loop */
        checkstate = 0;
        break; 
      }
      if (3 == NBC) {
        /* CLUSTER NBC method */
        numOfPartNeigh = numberContrib - (i + 1);
        for (d = 0; d < numOfPartNeigh; ++d) {
          neighListOfCurrentPart[d] = i + d + 1 - model_index_shift; 
        }
        ier = KIM_STATUS_OK; 
      } else {
        /* LOCATOR mode */
        request = i - model_index_shift;
        ier = KIM_API_get_neigh(pkim, one, request, &currentPart, &numOfPartNeigh, &neighListOfCurrentPart, &Rij_list);
        if (KIM_STATUS_OK > ier) {
          /* some sort of problem, exit */
          KIM_API_report_error(__LINE__, __FILE__, "get_neigh error at locator cycle.", ier);
          return ier; 
        } 
      } 
      i++;
    } /* If IterOrLoca Ends */
  
    if (i<0 || i>=numberContrib) { fprintf(f4out,"Your index is out off number of particles. Do you know how you end up here? Because I don't."); exit(0); }
    /* Mapping of particle i to THinput array */
    //if ( i < THnumContrib ) 
    THPartListOffset[i]=specieOffset[particleSpecies[i]]+specieCount[particleSpecies[i]]*num_syms;
    THPartBackOffset[i]=specieBackOffset[particleSpecies[i]]+specieCount[particleSpecies[i]];
    specieCount[particleSpecies[i]]++;
    /* Put neighbor list and Rij of all atoms (numberContrib) to arrays */
    currentPartList[currentPartCount]=i;
    numOfPartNeighList[currentPartCount]=numOfPartNeigh;
    neighListOffset[currentPartCount]=neighCount;
    for (jj = 0; jj < numOfPartNeigh; ++jj) {
        neighListOfParts[neighCount]=neighListOfCurrentPart[jj]; 
        if (0 == NBC) {
           for (d = 0; d < DIM; ++d) {
              Rij_list_all[neighCount*DIM + d] = Rij_list[jj*DIM + d]; 
              //Rij[d] = Rij_list[jj*DIM + d];
           }
           //if (norm(Rij)<0.000001) printf("Rij ZERO! i:%d j:%d >>>> %25.16f\n",i,neighListOfCurrentPart[jj],norm(Rij)); 
        }
        neighCount++;
    }
    currentPartCount++;
  } /* infinite while loop (terminated by break statements above) */
  partCounted=currentPartCount-1;
  //printf("partCounted=%d\n",partCounted);
 
  for (t=0;t<num_typ;t++){
        specieCount[t]=0;
  }
  currentPartCount = 0;
  while( 1 ) {
    /* Loop over all atoms for all NBC methods */
    if (partCounted < currentPartCount) {
        /* incremented past end of list, terminate loop */
        break; 
    }
    numOfPartNeigh=numOfPartNeighList[currentPartCount];
    i=currentPartList[currentPartCount];
    specieCount[particleSpecies[i]]++;
    neighCount=neighListOffset[currentPartCount];
    for (jj = 0; jj < numOfPartNeigh; ++jj) {
      j = neighListOfParts[neighCount+jj];
      for (d = 0; d < DIM; ++d) {
        if (0 != NBC) {
          /* all methods except NEIGH_RVEC */
          Rij[d] = coords[i*DIM + d] - coords[j*DIM + d];
        } else {
          /* NEIGH_RVEC_* methods */
          Rij[d] = Rij_list_all[(neighCount+jj)*DIM + d];
        }
      } 
      //printf("Atom: %d Neighbor ID: %d  Dist: %f  Vec: %f %f %f\n",i,j,norm(Rij), Rij[0], Rij[1], Rij[2]);
      //if (norm(Rij)<=0.000000001) fprintf(f4out," Rij zero! i(t:%d)=%d j(t:%d)=%d >>> %f\n",particleSpecies[i],i,particleSpecies[j],j,norm(Rij));  
      if (norm(Rij)<=1.0) fprintf(f4out," Atomic distance is too close for i(t:%d)=%d j(t:%d)=%d , Rij= %f\n",particleSpecies[i],i,particleSpecies[j],j,norm(Rij));  
    }
    currentPartCount++;
  }  /* infinite while loop (terminated by break statements above) */

 
  /* We get the neighbors of all nParts atoms. 
     Now, we can calculate 2-body and 3-body functions
     and compute energy and forces */
  
  currentPartCount = 0;
  while( 1 ) {
    /* Loop over neighbor list for all NBC methods */
    if (partCounted < currentPartCount) {
        /* incremented past end of list, terminate loop */
        break; 
    }
    /* Get particle i from list */
    i = currentPartList[currentPartCount];
    /* Set number of neighbors of particle i */
    numOfPartNeigh=numOfPartNeighList[currentPartCount];
    neighCount=neighListOffset[currentPartCount];
    //printf("Neigh num: %d\n",numOfPartNeigh);
    /* loop over the neighbors of particle i */
    for (jj = 0; jj < numOfPartNeigh; ++jj) {
      j = neighListOfParts[neighCount+jj] + model_index_shift;  /* get neighbor ID */
      /* compute relative position vector and squared distance */
      Rsqij = 0.0;
      for (d = 0; d < DIM; ++d) {
        if (0 != NBC) {
          /* all methods except NEIGH_RVEC */
          Rij[d] = coords[i*DIM + d] - coords[j*DIM + d];
        } else {
          /* NEIGH_RVEC_* methods */
          Rij[d] = Rij_list_all[(neighCount+jj)*DIM + d]; 
        }

        /* apply periodic boundary conditions if required */
        if (2 == NBC) {
          if (abs(Rij[d]) > 0.5*boxSideLengths[d]) {
              Rij[d] -= (Rij[d]/fabs(Rij[d]))*boxSideLengths[d]; 
          } 
        }
        
        Rji[d] = -Rij[d];
        
        /* compute squared distance */
        Rsqij += Rij[d]*Rij[d]; 
      } /* End of Rij[k] loop */

      /* compute ANN input */
      if (Rsqij < (*cutsq) && sqrt(Rsqij > 0.1) ) {
        
        /* Calculate the contribution of Radial Symmetry Functions for each 
           neighbor of atom i. In calculation, the index of contributions in the 
           THinput array are shifted accordingly. */
        /* Contribution to particle i*/
        //if ( i < THnumContrib ) {
           //printf("<<<1>>> i:%d j:%d\n",i,j);
           radoffset=THPartListOffset[i]+particleSpecies[j]*num_rad;
           for (symi=0;symi<num_rad;symi++) { 
               THinput[radoffset+symi]+=rad_symf(rad_eta[symi], cutoff, Rij);
           }
        //}
        /* Contribution to particle j*/
        /* if half list add contribution to the other particle in the pair */
        if ((1 == HalfOrFull) && (j < numberContrib)) {
           radoffset=THPartListOffset[j]+particleSpecies[i]*num_rad;
           for (symi=0;symi<num_rad;symi++) { 
               THinput[radoffset+symi]+=rad_symf(rad_eta[symi], cutoff, Rji);
           }
        } 

        /* Calculate neighbors of particle j for angular contributions */ 

        /* Find particle j in currentPartList */
        /* Set number of neighbors of particle j */
        numOfPartNeighJ=numOfPartNeighList[currentPartCount];
        neighCountJ=neighListOffset[currentPartCount];
     
        /* loop over the neighbors of particle i */
        for (kk = 0; kk < numOfPartNeighJ; ++kk) {
        //for (kk = jj+1; kk < numOfPartNeighJ; ++kk) {
           k = neighListOfParts[neighCountJ+kk] + model_index_shift;  /* get neighbor ID */
           if (jj==kk) continue;
           //if (jj==numOfPartNeigh) break;

          /* compute relative position vector and squared distance */
          Rsqik = 0.0;
          Rsqjk = 0.0;
          for (d = 0; d < DIM; ++d) {
            if (0 != NBC) {
              /* all methods except NEIGH_RVEC */
              Rik[d] = coords[i*DIM + d] - coords[k*DIM + d]; 
              Rjk[d] = coords[j*DIM + d] - coords[k*DIM + d]; 
            } else {
              /* NEIGH_RVEC_* methods */
              Rik[d] = Rij_list_all[(neighCountJ+kk)*DIM + d]; 
              Rjk[d] = Rik[d] - Rij[d]; 
            }

            /* apply periodic boundary conditions if required */
            if (2 == NBC) {
              if (abs(Rik[d]) > 0.5*boxSideLengths[d]) {
                  Rik[d] -= (Rik[d]/fabs(Rik[d]))*boxSideLengths[d]; 
              } 
              if (abs(Rjk[d]) > 0.5*boxSideLengths[d]) {
                  Rjk[d] -= (Rjk[d]/fabs(Rjk[d]))*boxSideLengths[d]; 
              } 
            }

            /* compute squared distance */
            Rsqik += Rik[d]*Rik[d]; 
            Rsqjk += Rjk[d]*Rjk[d]; 
          } /* End of Rij[k] loop */
        
          /* compute ANN input */
          if (Rsqik < (*cutsq) && Rsqjk < (*cutsq) && Rsqik > 0.1 && Rsqjk > 0.1) {
            if (particleSpecies[j]==particleSpecies[k]) {
               //if ( i < THnumContrib ) {
                 angoffseti=THPartListOffset[i]+particleSpecies[j]*num_ang+num_typ*num_rad;
               //}
               if ((1 == HalfOrFull) && (j < numberContrib)) {
                 angoffsetj=THPartListOffset[j]+particleSpecies[i]*num_ang+num_typ*num_rad;
               }
            } 
            else {
               //if ( i < THnumContrib ) {
                  angoffseti=THPartListOffset[i]+num_typ*(num_ang+num_rad);
               //}
               if ((1 == HalfOrFull) && (j < numberContrib)) {
                  angoffsetj=THPartListOffset[j]+num_typ*(num_ang+num_rad);
               }
               for (t1t=0;t1t<num_typ;t1t++){
                   for (t2t=t1t+1;t2t<num_typ;t2t++){
                      if ( (j==t1t && k==t2t) || (j==t2t && k==t1t) ) {
                         t1t=num_typ;
                         t2t=num_typ;
                         break;
                      }
                      //printf("i: %d  j: %d  angoffseti: %d  angoffsetj: %d  t1: %d  t2: %d \n",i,j,angoffseti,angoffsetj,t1t,t2t);
//                      angoffseti=angoffseti+num_ang;
//                      angoffsetj=angoffsetj+num_ang;
                      //printf("i: %d  j: %d  angoffseti: %d  angoffsetj: %d  t1: %d  t2: %d \n",i,j,angoffseti,angoffsetj,t1t,t2t);
                   }
               }
            }
            //if ( i < THnumContrib ) {
               //printf("<<<2>>> i:%d j:%d k:%d\n",i,j,k);
               for (symi=0;symi<num_ang;symi++) { 
                   THinput[angoffseti+symi]+=ang_symf(ang_eta[symi],lambda[symi],zeta[symi],cutoff,Rij,Rik,Rjk);
               }
            //}
            if ((1 == HalfOrFull) && (j < numberContrib)) {
               for (symi=0;symi<num_ang;symi++) {
                   THinput[angoffsetj+symi]+=ang_symf(ang_eta[symi],lambda[symi],zeta[symi],cutoff,Rji,Rjk,Rik);
               }
            }
          } /* End of if inside cutoff, Rsqik < cutoff */
        }  /* loop on kk */
      } /* End of if inside cutoff, Rsqij < cutoff */
    }  /* loop on jj */
    currentPartCount++;
  }  /* infinite while loop (terminated by break statements above) */

  /* THobject call with Lua binding */
  THier=compTHobject(L);
  if (THier<1) {
     ier=KIM_STATUS_OK;
  } else {
     ier=KIM_STATUS_FAIL;
  }
  if (KIM_STATUS_OK != ier) {
    sprintf(errortext, "compTHobject Lua binding failed. Lua error code %d\n", THier);
    KIM_API_report_error(__LINE__, __FILE__, errortext, KIM_STATUS_FAIL);
    return ier; 
  }

  fprintf(f4out,"---Output from Lua Binding---\n");
  currentPartCount = 0;
  THindexShift=3+THnumContrib;
  while( 1 ) {
    /* Loop over neighbor list for all NBC methods */
    if (partCounted < currentPartCount) {
        /* incremented past end of list, terminate loop */
        break; 
    }
    i = currentPartList[currentPartCount];
    //printf(" %d %lf\n",currentPartCount,THoutput[currentPartCount+2]);
    
    if (comp_energy)
       *energy+=(THinput[THPartBackOffset[i]+3]*stdeng)+atomavgeng;
    if (comp_particleEnergy)
       particleEnergy[i]=(THinput[THPartBackOffset[i]+3]*stdeng)+atomavgeng;   
    
    currentPartCount++;
  }  /* infinite while loop (terminated by break statements above) */
  fprintf(f4out,"Total Energy: %f\n",*energy);
//  *energy=(THinput[1]*stdeng)+avgeng;   
//  printf("ANN total energy: %f\n",*energy);
  //fprintf(f2out," Number of atoms %d %d %d\n",*nParts,num_atom_typs[0],num_atom_typs[1]);
//  if (num_typ<2){ 
     //fprintf(f2out," %d %d %d %lf %lf %lf %lf\n",Giter,(*nParts),num_atom_typs[0],*energy,Getotal,Getotal-*energy,((Getotal-(*energy))*1000.0/(*nParts)));
//     fprintf(f2out," %d %d %f\n",THnumContrib,num_atom_typs[0],*energy);
//  } else {
     //fprintf(f2out," %d %d %d %d %lf %lf %lf %lf\n",Giter,(*nParts),num_atom_typs[0],num_atom_typs[1],*energy,Getotal,Getotal-*energy,((Getotal-*energy)/(*nParts))*1000.0);
//     fprintf(f2out," %d %d %d %f\n",THnumContrib,num_atom_typs[0],num_atom_typs[1],*energy);
//  }
//  fflush(f2out);
//  fflush(stdout);
  luatime=THinput[0];   
  luagarbage=THinput[2];
  fprintf(f4out," Lua timer:%f garbage:%f\n",luatime,luagarbage);
//  fflush(f4out);
 
 
if ( comp_force || comp_virial || comp_particleVirial ) {

  currentPartCount = 0;
  while( 1 ) {
   /* Loop over all atoms */
   if (partCounted < currentPartCount) {
       /* incremented past end of list, terminate loop */
       break; 
   }
   /* Get particle i from list */
    i = currentPartList[currentPartCount];
    /* Set number of neighbors of particle i */
    numOfPartNeigh=numOfPartNeighList[currentPartCount];
    neighCount=neighListOffset[currentPartCount];
    for (jj = 0; jj < numOfPartNeigh; ++jj) {
      j = neighListOfParts[neighCount+jj] + model_index_shift;  /* get neighbor ID */
      /* compute relative position vector and squared distance */
      Rsqij = 0.0;
      for (d = 0; d < DIM; ++d) {
        if (0 != NBC) {
          /* all methods except NEIGH_RVEC */
          Rij[d] = coords[i*DIM + d] - coords[j*DIM + d];
          //printf("Rij in coords  i:%d j:%d d:%d\n",i,j,d); 
        } else {
          /* NEIGH_RVEC_* methods */
          Rij[d] = Rij_list_all[(neighCount+jj)*DIM + d]; 
        }

        /* apply periodic boundary conditions if required */
        if (2 == NBC) {
          if (abs(Rij[d]) > 0.5*boxSideLengths[d]) {
            Rij[d] -= (Rij[d]/fabs(Rij[d]))*boxSideLengths[d]; 
          } 
        }

        Rji[d]= -Rij[d];

        /* compute squared distance */
        Rsqij += Rij[d]*Rij[d]; 
      } /* End of Rij[k] loop */

      /* compute energy and force */
      if (Rsqij < (*cutsq) && Rsqij > 0.000000001) {
        
        /* Calculate the contribution of Radial Symmetry Functions for each 
           neighbor of atom i. In calculation, the index of contributions in the 
           THinput array are shifted accordingly. */
        /* Contribution to particle i*/
       //if (j==ii && i!=ii && i!=j){
       //if (i!=j){
        radoffset=THindexShift+THPartListOffset[i]+particleSpecies[j]*num_rad;
        /*******************************************************
        * force contribution to particle j from all neighbors *
        *******************************************************/
        for (d = 0; d < DIM; d++) { ffi[d] = 0.0; 
                                    ffj[d] = 0.0; }
        for (symi=0;symi<num_rad;symi++) { 
            rad_symf_dr(Rgradi, rad_eta[symi], cutoff, Rij);
            for (d = 0; d < DIM; ++d) {
                    ff =stdeng*THinput[radoffset+symi]*Rgradi[d];
                ffi[d]-=ff;
                ffj[d]+=ff;
            }
        }
        for (d = 0; d < DIM; d++) { forces[i*DIM+d] += ffi[d]; 
                                    forces[j*DIM+d] += ffj[d]; }
        if ( comp_virial ) {
             fmm=0.5;
             stress[0][0] += fmm * ( ffj[0] * Rij[0] + ffi[0] * Rji[0] );  
             stress[0][1] += fmm * ( ffj[0] * Rij[1] + ffi[0] * Rji[1] );  
             stress[0][2] += fmm * ( ffj[0] * Rij[2] + ffi[0] * Rji[2] );  
             stress[1][1] += fmm * ( ffj[1] * Rij[1] + ffi[1] * Rji[1] );  
             stress[1][2] += fmm * ( ffj[1] * Rij[2] + ffi[1] * Rji[2] );  
             stress[2][2] += fmm * ( ffj[2] * Rij[2] + ffi[2] * Rji[2] );
        }
        if ( comp_particleVirial ) {
             fmm=0.5;
             particleVirial[i*VDIM]   += fmm * ffj[0] * Rij[0];
             particleVirial[i*VDIM+1] += fmm * ffj[1] * Rij[1];
             particleVirial[i*VDIM+2] += fmm * ffj[2] * Rij[2];
             particleVirial[i*VDIM+3] += fmm * ffj[1] * Rij[2];
             particleVirial[i*VDIM+4] += fmm * ffj[0] * Rij[2];
             particleVirial[i*VDIM+5] += fmm * ffj[0] * Rij[1];

             particleVirial[j*VDIM]   += fmm * ffi[0] * Rji[0];
             particleVirial[j*VDIM+1] += fmm * ffi[1] * Rji[1];
             particleVirial[j*VDIM+2] += fmm * ffi[2] * Rji[2];
             particleVirial[j*VDIM+3] += fmm * ffi[1] * Rji[2];
             particleVirial[j*VDIM+4] += fmm * ffi[0] * Rji[2];
             particleVirial[j*VDIM+5] += fmm * ffi[0] * Rji[1];
        }
       //} 

        /* Calculate neighbors of particle j for angular contributions */ 
        /* Set number of neighbors of particle j */
        numOfPartNeighJ=numOfPartNeighList[currentPartCount];
        neighCountJ=neighListOffset[currentPartCount];
     
        /* loop over the neighbors of particle i */
        for (kk = 0; kk < numOfPartNeighJ; ++kk) {
        //for (kk = jj+1; kk < numOfPartNeighJ; ++kk) {
           k = neighListOfParts[neighCountJ+kk] + model_index_shift;  /* get neighbor ID */
           if (jj==kk) continue;
           //if (jj==numOfPartNeigh) break;
           //if (j==k && i==k && i==j && i==ii) continue;
           if (j==k && i==k) continue;

          /* compute relative position vector and squared distance */
          Rsqik = 0.0;
          Rsqjk = 0.0;
          for (d = 0; d < DIM; ++d) {
            if (0 != NBC) {
              /* all methods except NEIGH_RVEC */
              Rik[d] = coords[i*DIM + d] - coords[k*DIM + d]; 
              Rjk[d] = coords[j*DIM + d] - coords[k*DIM + d]; 
            } else {
              /* NEIGH_RVEC_* methods */
              Rik[d] = Rij_list_all[(neighCountJ+kk)*DIM + d]; 
            }
            Rjk[d] = Rik[d] - Rij[d]; 

            /* apply periodic boundary conditions if required */
            if (2 == NBC) {
              if (abs(Rik[d]) > 0.5*boxSideLengths[d]) {
                Rik[d] -= (Rik[d]/fabs(Rik[d]))*boxSideLengths[d]; 
              } 
              if (abs(Rjk[d]) > 0.5*boxSideLengths[d]) {
                Rjk[d] -= (Rjk[d]/fabs(Rjk[d]))*boxSideLengths[d]; 
              } 
            }
   
            Rki[d] = -Rik[d]; 
            //Rkj[d] = -Rjk[d]; 

            /* compute squared distance */
            Rsqik += Rik[d]*Rik[d]; 
            Rsqjk += Rjk[d]*Rjk[d]; 
          } /* End of Rij[k] loop */
        
          /* compute energy and force */
          if (Rsqik < (*cutsq) && Rsqjk < (*cutsq) && Rsqik > 0.000000001 && Rsqjk > 0.000000001) {
            if (particleSpecies[j]==particleSpecies[k]) {
               angoffseti=THindexShift+THPartListOffset[i]+particleSpecies[j]*num_ang+num_typ*num_rad;
            } 
            else {
               angoffseti=THindexShift+THPartListOffset[i]+num_typ*(num_ang+num_rad);
               for (t1t=0;t1t<num_typ;t1t++){
                   for (t2t=t1t+1;t2t<num_typ;t2t++){
                      if ( (j==t1t && k==t2t) || (j==t2t && k==t1t) ) {
                         t1t=num_typ;
                         t2t=num_typ;
                         break;
                      }
                      //printf("i: %d  j: %d  angoffseti: %d  angoffsetj: %d  t1: %d  t2: %d \n",i,j,angoffseti,angoffsetj,t1t,t2t);
//                      angoffseti=angoffseti+num_ang;
//                      angoffsetj=angoffsetj+num_ang;
                      //printf("i: %d  j: %d  angoffseti: %d  angoffsetj: %d  t1: %d  t2: %d \n",i,j,angoffseti,angoffsetj,t1t,t2t);
                   }
               }
            }
                /*******************************************************
                * force contribution to particle j from all neighbors *
                *******************************************************/
                for (d = 0; d < DIM; d++) { ffi[d] = 0.0;
                                            ffj[d] = 0.0;
                                            ffk[d] = 0.0; }
                for (symi=0;symi<num_ang;symi++) { 
                    ang_symf_dr(Rgradi, Rgradj, Rgradk, ang_eta[symi],lambda[symi],zeta[symi],cutoff,Rij,Rik,Rjk);
                    for (d = 0; d < DIM; ++d) {
                            ff  = stdeng*THinput[angoffseti+symi];
                        ffi[d] -= ff*Rgradi[d];
                        ffj[d] -= ff*Rgradj[d];
                        ffk[d] -= ff*Rgradk[d];
                    }
                }
                for (d = 0; d < DIM; d++) { forces[i*DIM+d] += ffi[d];
                                            forces[j*DIM+d] += ffj[d]; 
                                            forces[k*DIM+d] += ffk[d]; }
                if ( comp_virial ) {
                   fmm=0.5;
                   stress[0][0] +=  fmm * ( ffj[0] * Rij[0] + ffk[0] * Rik[0] + ffk[0] * Rjk[0] );  
                   stress[0][1] +=  fmm * ( ffj[0] * Rij[1] + ffk[0] * Rik[1] + ffk[0] * Rjk[1] );  
                   stress[0][2] +=  fmm * ( ffj[0] * Rij[2] + ffk[0] * Rik[2] + ffk[0] * Rjk[2] );  
                   stress[1][1] +=  fmm * ( ffj[1] * Rij[1] + ffk[1] * Rik[1] + ffk[1] * Rjk[1] );  
                   stress[1][2] +=  fmm * ( ffj[1] * Rij[2] + ffk[1] * Rik[2] + ffk[1] * Rjk[2] );  
                   stress[2][2] +=  fmm * ( ffj[2] * Rij[2] + ffk[2] * Rik[2] + ffk[2] * Rjk[2] );  

                   fmm=0.25;
                   stress[0][0] +=  fmm * ( ffi[0] * Rji[0] + ffi[0] * Rki[0] );  
                   stress[0][1] +=  fmm * ( ffi[0] * Rji[1] + ffi[0] * Rki[1] );  
                   stress[0][2] +=  fmm * ( ffi[0] * Rji[2] + ffi[0] * Rki[2] );  
                   stress[1][1] +=  fmm * ( ffi[1] * Rji[1] + ffi[1] * Rki[1] );  
                   stress[1][2] +=  fmm * ( ffi[1] * Rji[2] + ffi[1] * Rki[2] );  
                   stress[2][2] +=  fmm * ( ffi[2] * Rji[2] + ffi[2] * Rki[2] );  
                }
                if ( comp_particleVirial ) {
                   fmm=0.5;
                   particleVirial[i*VDIM]   += fmm * ( ffj[0] * Rij[0] + ffk[0] * Rik[0] );
                   particleVirial[i*VDIM+1] += fmm * ( ffj[1] * Rij[1] + ffk[1] * Rik[1] );
                   particleVirial[i*VDIM+2] += fmm * ( ffj[2] * Rij[2] + ffk[2] * Rik[2] );
                   particleVirial[i*VDIM+3] += fmm * ( ffj[1] * Rij[2] + ffk[1] * Rik[2] );
                   particleVirial[i*VDIM+4] += fmm * ( ffj[0] * Rij[2] + ffk[0] * Rik[2] );
                   particleVirial[i*VDIM+5] += fmm * ( ffj[0] * Rij[1] + ffk[0] * Rik[1] );

                   particleVirial[j*VDIM]   += fmm * ( ffk[0] * Rjk[0] );
                   particleVirial[j*VDIM+1] += fmm * ( ffk[1] * Rjk[1] );
                   particleVirial[j*VDIM+2] += fmm * ( ffk[2] * Rjk[2] );
                   particleVirial[j*VDIM+3] += fmm * ( ffk[1] * Rjk[2] );
                   particleVirial[j*VDIM+4] += fmm * ( ffk[0] * Rjk[2] );
                   particleVirial[j*VDIM+5] += fmm * ( ffk[0] * Rjk[1] );
                
                   fmm=0.25;
                   particleVirial[j*VDIM]   += fmm * ( ffi[0] * Rji[0] );
                   particleVirial[j*VDIM+1] += fmm * ( ffi[1] * Rji[1] );
                   particleVirial[j*VDIM+2] += fmm * ( ffi[2] * Rji[2] );
                   particleVirial[j*VDIM+3] += fmm * ( ffi[1] * Rji[2] );
                   particleVirial[j*VDIM+4] += fmm * ( ffi[0] * Rji[2] );
                   particleVirial[j*VDIM+5] += fmm * ( ffi[0] * Rji[1] );

                   particleVirial[k*VDIM]   += fmm * ( ffi[0] * Rki[0] );
                   particleVirial[k*VDIM+1] += fmm * ( ffi[1] * Rki[1] );
                   particleVirial[k*VDIM+2] += fmm * ( ffi[2] * Rki[2] );
                   particleVirial[k*VDIM+3] += fmm * ( ffi[1] * Rki[2] );
                   particleVirial[k*VDIM+4] += fmm * ( ffi[0] * Rki[2] );
                   particleVirial[k*VDIM+5] += fmm * ( ffi[0] * Rki[1] );
                }
          } /* End of if inside cutoff, Rsqik < cutoff */
        }  /* loop on kk */
      } /* End of if inside cutoff, Rsqij < cutoff */
    }  /* loop on jj */
    currentPartCount++;
  }  /* infinite while loop on i (terminated by break statements above) */
 
} // end of if (comp_force || comp_virial || comp_particleVirial )

  if ( comp_virial ) {
     virial[0] = stress[0][0];
     virial[1] = stress[1][1];
     virial[2] = stress[2][2];
     virial[3] = stress[1][2];
     virial[4] = stress[0][2];
     virial[5] = stress[0][1];
  }

//  for(i=0;i<THnumContrib;i++){
//     printf("%d %g %g %g\n",i+1,force[i*DIM],force[i*DIM+1],force[i*DIM+2]);
//     fprintf(f3out,"%d %25.16E %25.16E %25.16E\n",i+1,forces[i*DIM],forces[i*DIM+1],forces[i*DIM+2]);
//  }
//  fflush(f3out);


  /* Free temporary storage */
  free(THPartListOffset);
  free(THPartBackOffset);
  free(neighListOffset);
  free(currentPartList);
  free(numOfPartNeighList);
  free(neighListOfParts);
  free(specieOffset); 
  free(specieBackOffset); 
  free(specieCount);
  free(num_atom_typs); 
  free(all_atom_typs); 
  
  if (0 == NBC) {
     free(Rij_list_all);
  }
  if (3 == NBC) {
     free(neighListOfCurrentPart);
  }
  
  THier=freeTHobject(L);
  THier=0;
  if (THier<1) {
     ier=KIM_STATUS_OK;
  } else {
     ier=KIM_STATUS_FAIL;
  }
//  if (KIM_STATUS_OK != ier) {
//    sprintf(errortext, "compTHobject Lua binding failed. Lua error code %d\n", THier);
//    KIM_API_report_error(__LINE__, __FILE__, errortext, KIM_STATUS_FAIL);
//    return ier; 
//  }
//  free(THinput);
  fflush(f4out);

  return KIM_STATUS_OK; 
}

/* Initialization function */
int model_driver_init(void *km, char *paramfile_names, int* nmstrlen,int* numparamfiles) {

  /* KIM variables */
  intptr_t *pkim = *((intptr_t **) km);
  
  /* Local variables */
  FILE* fid;
  double cutoff;
  double pcadata;
  double avgdata;
  double stddata;
  double* stddat;
  double* avgdat;
  double* pcadat;
  char* THobj;
  double* THcmp;
  double avgeng;
  int avgatom;
  double stdeng;
  double rad_eta;
  double ang_eta;
  int lambda;
  int zeta;
  int num_typ;
  int num_rad;
  int num_ang;
  int num_syms;
  int num_syms_sq;
  int pcashift;
  int pcacount;
  double *model_cutoff = NULL;
  int i,j,t,THier,Lgcer;
  int ier = KIM_STATUS_OK;
  int   kim_ntypes;
  int   kim_foo;
  model_buffer* buffer;
  const char* NBCstr;
  const double bohrtoang=(1.88*1.88)/10000.0;
  uint64_t gc_mem_usage;
  char errortext[255];
  char filename[255];
  
  /* neighbor type params */
  //int   NBC;
  //int   IterOrLoca;
  //int   HalfOrFull;
  
  char *paramfile1name = &(paramfile_names[0]);
  char *paramfile2name = &(paramfile_names[(*nmstrlen)]);
     
  //f2out = fopen("PredictionResults", "w");
  //f3out = fopen("Forces", "w");
  //srand(time(NULL));
  //myid = rand();
  THglobal++;
  FILE *urandom = fopen("/dev/urandom", "r");
  if (urandom==NULL) {
     srand(time(NULL));
     myid = rand();
  } else {
     fread(&myid, sizeof (myid), 1 ,urandom);
  }
  fclose(urandom);
  THmyid=myid;
  sprintf(filename,"INN-%lu.log",myid);
  f4out = fopen(filename, "w");
  //f4out = fopen("INN.log", "w");
 
  /* check number of files */
  if (*numparamfiles != 2) {
     ier = KIM_STATUS_FAIL;
     KIM_API_report_error(__LINE__, __FILE__, "Incorrect number of parameter files.", ier);
     return ier; 
  }

  /* get number of types */
  ier = KIM_API_get_num_model_species(pkim, &kim_ntypes, &kim_foo);
  if (KIM_STATUS_OK > ier) {
      KIM_API_report_error(__LINE__, __FILE__, "Unsupported number of particle types or KIM API error getting number of types.", KIM_STATUS_FAIL);
      return KIM_STATUS_FAIL;
  }

  /* store pointer to functions in KIM object */
  /* *INDENT-OFF* */
  KIM_API_setm_method(pkim, &ier, 3*4,
                     "compute", 1, &compute, 1,
                     "reinit",  1, &reinit,  1,
                     "destroy", 1, &destroy, 1);
  /* *INDENT-ON* */
  if (KIM_STATUS_OK > ier){
      KIM_API_report_error(__LINE__, __FILE__, "KIM_API_setm_method", ier);
      return KIM_STATUS_FAIL;
  }

  fprintf(f4out,"Allocating buffer\n");
  /* allocate buffer */
  buffer = (model_buffer *) malloc(sizeof(model_buffer));
  if (NULL == buffer) {
     ier = KIM_STATUS_FAIL;
     KIM_API_report_error(__LINE__, __FILE__, "model_buffer malloc", ier);
     return ier; 
  }

  /* Read in model parameters from parameter file */
  fprintf(f4out,"Reading file %s\n",paramfile1name);
  fid = fopen(paramfile1name, "r");
  if (fid == NULL) {
     sprintf(errortext, "Unable to open parameter file for ANN symmetry functions: %s\n", paramfile1name);
     KIM_API_report_error(__LINE__, __FILE__, errortext, KIM_STATUS_FAIL);
     exit(EXIT_FAILURE);
  }


  /* Read the cutoff of the model */ 
  ier = fscanf(fid,"%lf",&cutoff);
  /* check that we read the right number of parameters */
  if (0 == ier) {
     ier = KIM_STATUS_FAIL;
     KIM_API_report_error(__LINE__, __FILE__, "Unable to read cutoff.", KIM_STATUS_FAIL);
     return ier; 
  }

  ier = fscanf(fid,"%d",&num_typ);
  /* check that we read the right number of parameters */
  if (0 == ier) {
     ier = KIM_STATUS_FAIL;
     KIM_API_report_error(__LINE__, __FILE__, "Unable to read number of species.", KIM_STATUS_FAIL);
     return ier; 
  }
  if (kim_ntypes != num_typ) {
      KIM_API_report_error(__LINE__, __FILE__, "Number of particle types from KIM API request does not match with ANN Model.", KIM_STATUS_FAIL);
      return KIM_STATUS_FAIL;
  }
 
  /* Read number of radial symmetry functions */ 
  ier=fscanf(fid,"%d",&num_rad);
  if (0 == ier) {
     ier = KIM_STATUS_FAIL;
     KIM_API_report_error(__LINE__, __FILE__, "Unable to read number of rad. sym. func.", KIM_STATUS_FAIL);
     return ier; 
  }
  /* Allocate memory for radial sym. func. */
  buffer->rad_eta =(double*)malloc(num_rad*sizeof(double));
  /* Read parameters for radial symmetry functions */ 
  for(i=0;i<num_rad;i++){ 
     ier=fscanf(fid, "%lf",&rad_eta);
     buffer->rad_eta[i]=rad_eta*bohrtoang;
  }
  if (0 == ier) {
     ier = KIM_STATUS_FAIL;
     KIM_API_report_error(__LINE__, __FILE__, "Unable to read radial eta parameters.", KIM_STATUS_FAIL);
     return ier; 
  }
  /* Read number of angular symmetry functions */ 
  ier=fscanf(fid,"%d",&num_ang);
  if (0 == ier) {
     ier = KIM_STATUS_FAIL;
     KIM_API_report_error(__LINE__, __FILE__, "Unable to read number of ang. sym. func.", KIM_STATUS_FAIL);
     return ier; 
  }

  unsigned long comb;
  if(num_typ<2) {
    num_syms=num_rad+num_ang;
  } else {
    comb=fact((unsigned long)num_typ)/(2*fact((unsigned long)(num_typ-2)));
    num_syms=(num_typ*num_rad)+(num_typ+comb)*num_ang;
  }
  /* Allocate memory for radial sym. func. */
  buffer->ang_eta=(double*)malloc(num_ang*sizeof(double));
  buffer->lambda=(int*)malloc(num_ang*sizeof(int));
  buffer->zeta=(int*)malloc(num_ang*sizeof(int));
  avgdat=(double*)malloc((num_typ*num_syms)*sizeof(double));
  stddat=(double*)malloc((num_typ*num_syms)*sizeof(double));
  pcadat=(double*)malloc((num_typ*num_syms*num_syms)*sizeof(double));

  /* Read parameters for angular symmetry functions */ 
  for(i=0;i<num_ang;i++){ 
     ier=fscanf(fid, "%lf",&ang_eta);
     buffer->ang_eta[i]=ang_eta*bohrtoang;
  }
  if (0 == ier) {
     ier = KIM_STATUS_FAIL;
     KIM_API_report_error(__LINE__, __FILE__, "Unable to read ang. eta parameters.", KIM_STATUS_FAIL);
     return ier; 
  }
  for(i=0;i<num_ang;i++){ 
     ier=fscanf(fid, "%d",&lambda);
     buffer->lambda[i]=lambda;
  }
  if (0 == ier) {
     ier = KIM_STATUS_FAIL;
     KIM_API_report_error(__LINE__, __FILE__, "Unable to read ang. lambda parameters.", KIM_STATUS_FAIL);
     return ier; 
  }
  for(i=0;i<num_ang;i++){ 
     ier=fscanf(fid, "%d",&zeta);
     buffer->zeta[i]=zeta;
  }
  if (0 == ier) {
     ier = KIM_STATUS_FAIL;
     KIM_API_report_error(__LINE__, __FILE__, "Unable to read ang. zeta parameters.", KIM_STATUS_FAIL);
     return ier; 
  }
  /* Read the energy average */ 
  ier = fscanf(fid,"%d",&avgatom);
  /* check that we read the right number of parameters */
  if (0 == ier) {
     ier = KIM_STATUS_FAIL;
     KIM_API_report_error(__LINE__, __FILE__, "Unable to read average atom no.", KIM_STATUS_FAIL);
     return ier; 
  }
  /* Read the energy average */ 
  ier = fscanf(fid,"%lf",&avgeng);
  /* check that we read the right number of parameters */
  if (0 == ier) {
     ier = KIM_STATUS_FAIL;
     KIM_API_report_error(__LINE__, __FILE__, "Unable to read average of energy.", KIM_STATUS_FAIL);
     return ier; 
  }
  /* Read the input data averages */ 
  for(t=0;t<num_typ;t++){ 
     for(i=0;i<(num_syms);i++){ 
        ier=fscanf(fid, "%lf",&avgdata);
        avgdat[t*num_syms+i]=avgdata;
     }
  }
  if (0 == ier) {
     ier = KIM_STATUS_FAIL;
     KIM_API_report_error(__LINE__, __FILE__, "Unable to read averages for sym. funcs.", KIM_STATUS_FAIL);
     return ier; 
  }
  /* Read the energy average */ 
  ier = fscanf(fid,"%lf",&stdeng);
  /* check that we read the right number of parameters */
  if (0 == ier) {
     ier = KIM_STATUS_FAIL;
     KIM_API_report_error(__LINE__, __FILE__, "Unable to read std of energy.", KIM_STATUS_FAIL);
     return ier; 
  }
  /* Read the input data averages */ 
  for(t=0;t<num_typ;t++){ 
     for(i=0;i<(num_syms);i++){ 
        ier=fscanf(fid, "%lf",&stddata);
        stddat[t*num_syms+i]=1.0/stddata;
     }
  }
  if (0 == ier) {
     ier = KIM_STATUS_FAIL;
     KIM_API_report_error(__LINE__, __FILE__, "Unable to read std of sym. funcs.", KIM_STATUS_FAIL);
     return ier; 
  }

  /* Read the PCA components */ 
  num_syms_sq=num_syms*num_syms;
  THcmp=(double*)malloc((num_typ*((num_syms+4)*num_syms))*sizeof(double));
  for(t=0;t<num_typ;t++){
     pcashift=t*((num_syms+2)*num_syms); 
     pcacount=t*(num_syms*num_syms); 
     for(i=0;i<num_syms;i++){ 
        for(j=0;j<num_syms;j++){ 
           ier=fscanf(fid, "%lf",&pcadata);
           THcmp[pcashift+i*num_syms+j]=pcadata;
           pcadat[pcacount+i*num_syms+j]=pcadata;
        }
     }
     for(i=0;i<num_syms;i++){ 
        THcmp[pcashift+num_syms_sq+i]=avgdat[t*num_syms+i];
     }
     for(i=0;i<num_syms;i++){ 
        THcmp[pcashift+num_syms_sq+num_syms+i]=stddat[t*num_syms+i];
     }
  }
  if (0 == ier) {
     ier = KIM_STATUS_FAIL;
     KIM_API_report_error(__LINE__, __FILE__, "Unable to read PCA components.", KIM_STATUS_FAIL);
     return ier; 
  }
  fclose(fid);
  
  /* Read in model parameters from parameter file */
  fprintf(f4out,"Reading file %s\n",paramfile2name);
  fid = fopen(paramfile2name, "rb");
  if (NULL == fid) {
     sprintf(errortext, "Unable to open Torch7 ANN object file: %s\n", paramfile2name);
     KIM_API_report_error(__LINE__, __FILE__, errortext, KIM_STATUS_FAIL);
     exit(EXIT_FAILURE);
  }
  if (fid != NULL) {
      /* Go to the end of the file. */
      if (fseek(fid, 0L, SEEK_END) == 0) {
        /* Get the size of the file. */
        long bufsize = ftell(fid);
        if (bufsize < 0) { 
            ier = KIM_STATUS_FAIL;
            KIM_API_report_error(__LINE__, __FILE__, "Unable to seek the end of Torch7 ANN object file.", KIM_STATUS_FAIL);
            return ier;
        }
        fprintf(f4out," THobject size %lu\n",bufsize);

        /* Allocate our buffer to that size. */
        THobj = (char*) malloc(sizeof(char) * (bufsize + 2));

        /* Go back to the start of the file. */
        if (fseek(fid, 0L, SEEK_SET) != 0) { 
            ier = KIM_STATUS_FAIL;
            KIM_API_report_error(__LINE__, __FILE__, "Unable to rewind Torch7 ANN object file.", KIM_STATUS_FAIL);
            return ier;
        }

        /* Read the entire file into memory. */
        size_t newLen = fread(THobj, sizeof(char), bufsize+1, fid);
        
        if (newLen == 0) {
            ier = KIM_STATUS_FAIL;
            KIM_API_report_error(__LINE__, __FILE__, "Unable to read Torch7 ANN object file.", KIM_STATUS_FAIL);
            return ier;
        } else {
            newLen++;
            THobj[newLen] = '\0'; /* Ending with NULL. */
            fprintf(f4out," THobject new size %lu\n",newLen);
            fprintf(f4out," THobject good to go!\n");
            THsize=newLen;
        }
      }
      fclose(fid);
  }

 /***************************************************
  *        Initialize ANN module with THobject      *
  ***************************************************/
  /* 
    In this part THobject will be passed to Lua code
    and the ANN module will be initialized and read 
    by Torch7 routines. The serialized object includes 
    ANN modules and parameters. When the object is 
    passed to LuaJIT Torch7 code, the modules will be 
    set as ANN and 'forward' and 'gradient' methods 
    can be applied to calculate energies and derivaties 
    of ANN with respect to inputs.
  */
  void* dlh = dlopen("libluajit.so", RTLD_NOW | RTLD_GLOBAL);
  if (!dlh) { 
     sprintf(errortext,"LuaKIM Initialization dlopen failed for libluajit.so, dlopen() error: %s\n",dlerror());
     ier = KIM_STATUS_FAIL;
     KIM_API_report_error(__LINE__, __FILE__, errortext, KIM_STATUS_FAIL);
     fprintf(f4out,"%s\n",errortext);
     return ier;
  } 
  L = luaL_newstate();
  if (NULL == L) {
    KIM_API_report_error(__LINE__, __FILE__, "Lua state creation failed.", KIM_STATUS_FAIL);
    return KIM_STATUS_FAIL;
  }
  fprintf(f4out,"Register all libraries to Lua\n");
  luaL_openlibs(L);
  fprintf(f4out,"Testing garbage collector of Lua\n");
  // do full gc
  Lgcer=lua_gc(L, LUA_GCSTOP, 0);
  gc_mem_usage = ((uint64_t)lua_gc(L, LUA_GCCOUNT, 0) << 10) + lua_gc(L, LUA_GCCOUNTB, 0);
  fprintf(f4out,"Lua mem usage: [%" PRIu64 "] Bytes\n", gc_mem_usage);
  ier=luaopen_LuaKIMinit(L); 
  ier=luaopen_LuaKIMinput(L); 
  ier=luaopen_LuaKIMpca(L); 
  ier=luaopen_LuaKIMparams(L); 
  THparams = (int *) malloc((2+num_typ)*sizeof(int));
  THparams[0]=num_typ;
  THparams[1]=num_syms;
  fprintf(f4out,"Push THobject to Lua. Lua calling ...\n");
  THobject=THobj;
  THcomponent=THcmp;
  ier=initTHobject(L);
  fprintf(f4out,"Returned to C\n");
  if (ier<0 || ier>0) {
    sprintf(errortext,"LuaKIM Initialization Error code: %d \n",ier);
    ier = KIM_STATUS_FAIL;
    KIM_API_report_error(__LINE__, __FILE__, errortext, KIM_STATUS_FAIL);
    fprintf(f4out,"%s\n",errortext);
    return ier;
  }
  lua_settop(L, 0);
  // do full gc
  Lgcer=lua_gc(L, LUA_GCCOLLECT, 0);
  gc_mem_usage = ((uint64_t)lua_gc(L, LUA_GCCOUNT, 0) << 10) + lua_gc(L, LUA_GCCOUNTB, 0);
  fprintf(f4out,"Lua mem usage: [%" PRIu64 "] Bytes. %d\n", gc_mem_usage,Lgcer);
 
 /***************************************************
  *         End of THobject Initialization          *
  ***************************************************/
  
  fprintf(f4out,"TH initialized\n");
  //THobject=NULL;
  //THcomponent=NULL;
  fprintf(f4out,"freeing THobject and THcomponents\n");
  THier=freeTHobject(L);
  fprintf(f4out,"Return to C\n");
  
  if (THier>0) {
     ier=KIM_STATUS_FAIL;
     KIM_API_report_error(__LINE__, __FILE__, "freeTHobject Lua binding failed.", KIM_STATUS_FAIL);
     return ier; 
  }

  /* get model_cutoff pointer from KIM object */
  model_cutoff = (double *)KIM_API_get_data(pkim, "cutoff", &ier);
  if (KIM_STATUS_OK > ier) {
    KIM_API_report_error(__LINE__, __FILE__, "KIM_API_get_data(\"cutoff\")", ier);
    return ier;
  }
  /* store model cutoff in KIM object */
  *model_cutoff = cutoff;

  fprintf(f4out,"Setting buffer\n");
  /* setup buffer */
  buffer->Pcutoff = cutoff;
  buffer->cutsq = cutoff*cutoff;
  buffer->num_typ = num_typ;
  buffer->num_rad = num_rad;
  buffer->num_ang = num_ang;
  buffer->avgd = avgdat; 
  buffer->stdd = stddat; 
  buffer->pcad = pcadat; 
  buffer->avge = avgeng;
  buffer->avga = avgatom;
  buffer->stde = stdeng;
  //buffer->THcomponent = THcomponent;
  //buffer->THobject = THobject;
  buffer->LuaL = L;

  free(THobj);
  free(THcmp);
  //free(avgdat);
  //free(stddat);
  //free(pcadat);

  fprintf(f4out,"Updating buffer values\n");
  /* Determine neighbor list boundary condition (NBC) */
  /**************************************************************
   * NBCstr =  	0 -> cluster mode
   * 		1 -> pure mode (half or full)
   * 		2 -> rvec (half or full) 
   * 		3 -> mi_opbc
   **************************************************************/
  ier = KIM_API_get_NBC_method(pkim, &NBCstr);
  if (KIM_STATUS_OK > ier) {
    KIM_API_report_error(__LINE__, __FILE__, "KIM_API_get_NBC_method", ier);
    return ier;
  }
  if ((!strcmp("NEIGH_RVEC_H",NBCstr)) || (!strcmp("NEIGH_RVEC_F",NBCstr))) {
    buffer->NBC = 0; }
  else if ((!strcmp("NEIGH_PURE_H",NBCstr)) || (!strcmp("NEIGH_PURE_F",NBCstr))) {
    buffer->NBC = 1; }
  else if ((!strcmp("MI_OPBC_H",NBCstr)) || (!strcmp("MI_OPBC_F",NBCstr))) {
    buffer->NBC = 2; }
  else if (!strcmp("CLUSTER",NBCstr)) {
    buffer->NBC = 3; }
  else {
    ier = KIM_STATUS_FAIL;
    KIM_API_report_error(__LINE__, __FILE__, "Unknown NBC method.", ier);
    return ier; 
  }

  /* Determine if Half or Full neighbor lists are being used */
  /*****************************/
  /* HalfOrFull = 1 -- Half    */
  /*            = 2 -- Full    */
  /*****************************/
  if (KIM_API_is_half_neighbors(pkim, &ier)) {
    buffer->HalfOrFull = 1;
  } else {
    buffer->HalfOrFull = 2; 
  }

  /* determine neighbor list handling mode */
  if (buffer->NBC != 3) {
    /******************************/
    /* IterOrLoca = 1 -- Iterator */
    /*            = 2 -- Locator  */
    /******************************/
    buffer->IterOrLoca = KIM_API_get_neigh_mode(pkim, &ier);
    if (KIM_STATUS_OK > ier) {
      KIM_API_report_error(__LINE__, __FILE__, "KIM_API_get_neigh_mode", ier);
      return ier;
    }
    if ((buffer->IterOrLoca != 1) && (buffer->IterOrLoca != 2)) {
      ier = KIM_STATUS_FAIL;
      sprintf(errortext, "Unsupported IterOrLoca mode = %i\n", buffer->IterOrLoca);
      KIM_API_report_error(__LINE__, __FILE__, errortext, KIM_STATUS_FAIL);
      return ier; 
    } 
  } 
  else {
    buffer->IterOrLoca = 2;  /* for CLUSTER NBC */ 
  }

  /* get the model_index_shift from KIM to the buffer */
  buffer->model_index_shift = KIM_API_get_model_index_shift(pkim);

  /* get the indices of several data structures within the KIM API for easy access */
  /* *INDENT-OFF* */
  KIM_API_getm_index(pkim, &ier, 11 * 3,
                     "boxSideLengths",              &(buffer->boxSideLengths_ind),              1,
                     "coordinates",                 &(buffer->coordinates_ind),                 1,
                     "energy",                      &(buffer->energy_ind),                      1,
                     "forces",                      &(buffer->forces_ind),                      1,
                     "virial",                      &(buffer->virial_ind),                      1,
                     "particleVirial",              &(buffer->particleVirial_ind),              1,
                     "numberContributingParticles", &(buffer->numberContribParticles_ind),      1,
                     "numberOfParticles",           &(buffer->numberOfParticles_ind),           1,
                     "numberOfSpecies",             &(buffer->numberOfSpecies_ind), 		1,
                     "particleEnergy",              &(buffer->particleEnergy_ind),              1,
                     "particleSpecies",             &(buffer->particleSpecies_ind),             1);
  /* *INDENT-ON* */
  if (KIM_STATUS_OK > ier) {
      KIM_API_report_error(__LINE__, __FILE__, "KIM_API_getm_index", ier);
      return ier;
  }

  /* store in model buffer */
  KIM_API_set_model_buffer(pkim, (void *)buffer, &ier);
  if (KIM_STATUS_OK > ier) {
      KIM_API_report_error(__LINE__, __FILE__, "KIM_API_set_model_buffer", ier);
      return ier;
  }
  fprintf(f4out,"OK at model_driver_init\n");
  fflush(f4out);
  return KIM_STATUS_OK; 
}



