-- 8/20/14 Artificial neural network potential interface
--         to calculate total energies and forces

--[[  
  
  Artificial neural network (ANN) module of the KIM API model driver based on LuaJIT+Torch7 
  Version 0.0.1b

  All rights and copyrights of this application are governed by the policies of 
  Harvard University, including Harvard’s “Statement of Policy in Regard to 
  Intellectual Property” (the “IP Policy”) 
  http://www.otd.harvard.edu/resources/policies/IP/
   
  Contributors: Berk Onat, Ekin D. Cubuk, Brad Malone, Efthimios Kaxiras

  
  Lua driver modes are determined by variable initdriver 
     If initdriver == 1 : CUDA + binary module file
                      2 : CUDA + ascii module file
                      3 : serial code will be used with binary module file
                      4 : serial code will be used with ascii module file
                      5 : OpenMP(2 threads) + binary module file
                      6 : OpenMP(2 threads) + ascii module file
                      For n>2
                      2n-1 : OpenMP(n-1 threads) + binary module file
                      2n : OpenMP(n-1 threads) + ascii module file
]]--


  require "cutorch"
  require "cunn"
--  require "nn"

torch.setdefaulttensortype('torch.DoubleTensor')
--torch.setdefaulttensortype('torch.FloatTensor')

function load_net()
  ier=1
  torch.setdefaulttensortype('torch.DoubleTensor')
  --torch.setdefaulttensortype('torch.FloatTensor')
  print('[ Lua ANN Driver ] : Loading ANN parameters from file.')
  print('[ Lua ANN Driver ] : Get size of the memory file.')
  --file = torch.MemoryFile(LuaKIMinit(),"r") -- creates a file in memory
  --str = LuaKIMinit() -- read THobject from KIM API array
  strsize = LuaTHsize() 
  seed = LuaTHmyid() 
  print('[ Lua ANN Driver ] : Size of the file: ' .. strsize)
  --str = LuaGetString()
  print('[ Lua ANN Driver ] : Allocate string storage.')
  str=torch.CharStorage(strsize)
  print('[ Lua ANN Driver ] : Get serialized THobject from model_driver.')
  strmap = LuaKIMinit() -- read THobject from KIM API array
  print('[ Lua ANN Driver ] : Put it in a string.')
  for i=1,strsize do
      str[i] = strmap[i]
  end
  --print(LuaGetString())
  --print('[ Lua ANN Driver ] : Mem file.')
  --file = torch.MemoryFile() -- creates a file in memory
  print('[ Lua ANN Driver ] : Write Mem file.')
  file = torch.MemoryFile(str,"r") -- creates a file in memory
  --str=torch.CharStorage(LuaGetString()):string()
  --file:writeString(str)
  --print('[ Lua ANN Driver ] : Read Char.')
  --file:seek(1)
  --str = torch.CharStorage():string(file)
  --print('[ Lua ANN Driver ] : Copy string into Memory file object.')
  --file:seek(1)
  --str = file:readByte()
  --print('[ Lua ANN Driver ] : Print.')
  --print(str)
  --str:string()
  --file = str
  print('[ Lua ANN Driver ] : Deserialize THobject of KIM API to MLP.')
  strfile = file:readString("*a") -- read object from memory to Torch object
  --mlp = str:readObject() -- read object from memory to Torch object
  --mlp = torch.deserialize(strfile,'ascii')
  mlp = torch.deserialize(strfile)
  print(mlp)
  --mlp = torch.deserialize(file)
  --[[  In this part an error check may be needed to test whether the input 
        parameters match the loaded MLP structure.
  ]]--
  if mlp then
     print('[ Lua ANN Driver ] : Parameters were loaded.')
     ier=0
  else
     print('[ Lua ANN Driver ] : Could not read parameters! Lua ANN driver stopped.')
     ier=-1
  end
  return ier
end

function init_net(anyinput)
  print('[ Lua ANN Driver ] : anyinput: ' .. anyinput)
  torch.setdefaulttensortype('torch.DoubleTensor')
  --torch.setdefaulttensortype('torch.FloatTensor')
  ier=1
  -- Access the memory blocks that is shared at C model driver in KIM API to Lua
  print('[ Lua ANN Driver ] : Reading KIM API params memory blocks.')
  KIMparams = LuaKIMparams()
  -- Check if there is access to the data at each memory block and if not return error flags.
  if KIMparams then
     print('[ Lua ANN Driver ] : KIM supplied parameter set is loaded.')
     numtypes=KIMparams[1]
     print('[ Lua ANN Driver ] : numtypes: ' .. numtypes)
     numsyms=KIMparams[2]
     print('[ Lua ANN Driver ] : numsyms: ' .. numsyms)
  else
     ier=-2
     print('[ Lua ANN Driver ] : Could not read KIM parameters! Lua ANN driver stopped.')
     return ier
  end
  print('[ Lua ANN Driver ] : Reading KIM API input memory blocks.')
  KIMinput = LuaKIMinput()
  if KIMinput then
     print('[ Lua ANN Driver ] : Input data is loaded.')
  else
     ier=-3
     print('[ Lua ANN Driver ] : Could not read input data! Lua ANN driver stopped.')
     return ier
  end
  print('[ Lua ANN Driver ] : Reading KIM API PCA components memory block.')
  KIMpca = LuaKIMpca()
  if KIMpca then
     print('[ Lua ANN Driver ] : PCA components data block is loaded.')
     print('[ Lua ANN Driver ] : Copying PCA components to Tensors.')
     KIMcount=0
     for t=1,numtypes do
         if t<2 then
             PCAtensorT1=torch.zeros(numsyms,numsyms)
             STDtensor1=torch.zeros(numsyms)
             AVGtensor1=torch.zeros(numsyms)
             for i=1,numsyms do
                 for j=1,numsyms do
                     KIMcount=KIMcount+1
                     PCAtensorT1[j][i]=KIMpca[KIMcount]
                 end
             end
             for i=1,numsyms do
                 KIMcount=KIMcount+1
                 AVGtensor1[i]=KIMpca[KIMcount]
             end
             for i=1,numsyms do
                 KIMcount=KIMcount+1
                 STDtensor1[i]=KIMpca[KIMcount]
             end
         elseif t<3 then
             PCAtensorT2=torch.zeros(numsyms,numsyms)
             STDtensor2=torch.zeros(numsyms)
             AVGtensor2=torch.zeros(numsyms)
             for i=1,numsyms do
                 for j=1,numsyms do
                     KIMcount=KIMcount+1
                     PCAtensorT2[j][i]=KIMpca[KIMcount]
                 end
             end
             for i=1,numsyms do
                 KIMcount=KIMcount+1
                 AVGtensor2[i]=KIMpca[KIMcount]
             end
             for i=1,numsyms do
                 KIMcount=KIMcount+1
                 STDtensor2[i]=KIMpca[KIMcount]
             end
         end
     end
  else
     ier=-4
     print('[ Lua ANN Driver ] : Could not read PCA components data! Lua ANN driver stopped.')
     return ier
  end
  --print('[ Lua ANN Driver ] : Reading KIM API memory blocks.')
  --KIMd2eout = LuaKIMd2eout()
  --if KIMd2eout then
  --   print('[ Lua ANN Driver ] : Output data block is loaded.')
  --else
  --   ier=-4
  --   print('[ Lua ANN Driver ] : Could not read output data! Lua ANN driver stopped.')
  --   return ier
  --end
  ier=0
  return ier
end

function comp_net(anyinput)
  --require "cunn"
  print('[ Lua ANN Driver ] : anyinput: ' .. anyinput)
  torch.setdefaulttensortype('torch.DoubleTensor')
  start_time = os.clock()
  --torch.setdefaulttensortype('torch.FloatTensor')
  ier=1
  -- Access the memory blocks that is shared at C model driver in KIM API to Lua
  print('[ Lua ANN Driver ] : Reading KIM API params memory blocks.')
  KIMparams = LuaKIMparams()
  -- Check if there is access to the data at each memory block and if not return error flags.
  if KIMparams then
     print('[ Lua ANN Driver ] : KIM supplied parameter set is loaded.')
  else
     ier=-2
     print('[ Lua ANN Driver ] : Could not read KIM parameters! Lua ANN driver stopped.')
     return ier
  end
  print('[ Lua ANN Driver ] : Reading KIM API input memory blocks.')
  KIMinput = LuaKIMinput()
  if KIMinput then
     print('[ Lua ANN Driver ] : Input data is loaded.')
  else
     ier=-3
     print('[ Lua ANN Driver ] : Could not read input data! Lua ANN driver stopped.')
     return ier
  end
  --print('[ Lua ANN Driver ] : Reading KIM API output memory blocks.')
  --KIMoutput = LuaKIMoutput()
  --if KIMoutput then
  --   print('[ Lua ANN Driver ] : Output data block is loaded.')
  --else
  --   ier=-4
  --   print('[ Lua ANN Driver ] : Could not read output data! Lua ANN driver stopped.')
  --   return ier
  --end
  --print('[ Lua ANN Driver ] : Reading KIM API memory blocks.')
  --KIMd2eout = LuaKIMd2eout()
  --if KIMd2eout then
  --   print('[ Lua ANN Driver ] : Output data block is loaded.')
  --else
  --   ier=-4
  --   print('[ Lua ANN Driver ] : Could not read output data! Lua ANN driver stopped.')
  --   return ier
  --end
  
  -- The definition of dataset
  dataset={}  

  print('[ Lua ANN Driver ] : Defining Dataset memory blocks.')
  function dataset:create()
    o={}
    setmetatable(o,self)
    self.__index=self
    return o
  end

  function dataset:size() 
    return #self
  end

  print('[ Lua ANN Driver ] : Allocate Dataset.')
  -- Create our dataset that is the input of main Torch7 MLP module at ANN architecture
  --input = dataset:create()
  -- Get parameters from KIM API using KIMparam memory block
  print('[ Lua ANN Driver ] : Get KIM parameters.')
  numtypes=KIMparams[1]
  print('[ Lua ANN Driver ] : numtypes: ' .. numtypes)
  numsyms=KIMparams[2]
  print('[ Lua ANN Driver ] : numsyms: ' .. numsyms)
  numatoms=torch.zeros(numtypes)
  KIMcount=0
  -- Input dataset is passed to ANN in a table with the size of species in the system
  -- Each table element contains values of symmetry functions for each atom in 
  -- a Torch7 tensor for one element type.
  print('[ Lua ANN Driver ] : Copy input to Torch7 Tensor.')
  inputset = dataset:create()
  gradOut = dataset:create()
  --print(PCAtensorT)
  --grad = dataset:create()
  for t=1,numtypes do
      numatoms[t]=KIMparams[2+t]
      print('[ Lua ANN Driver ] : numatoms ' .. t .. ' :  ' .. numatoms[t])
--      TYPtensor=torch.zeros(numatoms[t],numsyms)
--      gOutTensor=torch.Tensor(numatoms[t],1):fill(1.0)
      if t<2 then
         --gOutTensor1=torch.Tensor(numatoms[t],1):fill(1.0)
         gOutTensor1=torch.Tensor(1):fill(1.0)
         TYPtensor1=torch.zeros(numatoms[t],numsyms)
         TYPinput1=torch.zeros(numatoms[t],numsyms)
      elseif t<3 then
         --gOutTensor2=torch.Tensor(numatoms[t],1):fill(1.0)
         gOutTensor2=torch.Tensor(1):fill(1.0)
         TYPtensor2=torch.zeros(numatoms[t],numsyms)
         TYPinput2=torch.zeros(numatoms[t],numsyms)
--      elseif t<4 then
--         TYPtensor3=torch.zeros(numatoms[t],numsyms)
--      elseif t<5 then
--         TYPtensor4=torch.zeros(numatoms[t],numsyms)
--      else
--         TYPtensor5=torch.zeros(numatoms[t],numsyms)
      end
      if t<2 then
         for i=1,numatoms[t] do
             --print('[ Lua ANN Driver ] : atom ' .. i )
             for j=1,numsyms do
                 KIMcount=KIMcount+1
  --            print("KIMinput: " .. KIMinput[KIMcount])
                 TYPinput1[i][j]=KIMinput[KIMcount]
--                 TYPtensor1[i][j]=KIMinput[KIMcount]
--                 print(TYPtensor[i][j])
             end
             TYPinput1[i]=TYPinput1[i]-AVGtensor1
         end
         TYPtensor1:mm(TYPinput1,PCAtensorT1)
         TYPinput1=TYPtensor1:clone()
         for i=1,numatoms[t] do
             TYPtensor1[i]:cmul(TYPinput1[i],STDtensor1)
         end
--         table.insert(inputset,TYPtensor)
--         table.insert(gradOut,gOutTensor)
         --table.insert(inputset,TYPtensor1)
      elseif t<3 then
         for i=1,numatoms[t] do
             for j=1,numsyms do
                 KIMcount=KIMcount+1
  --            print("KIMinput: " .. KIMinput[KIMcount])
                TYPinput2[i][j]=KIMinput[KIMcount]
                --TYPtensor2[i][j]=PCAtensor[i][j]*KIMinput[KIMcount]
             end
             TYPinput2[i]=TYPinput2[i]-AVGtensor2
         end
         TYPtensor2:mm(TYPinput2,PCAtensorT2)
         TYPinput2=TYPtensor2:clone()
         for i=1,numatoms[t] do
             TYPtensor2[i]:cmul(TYPinput2[i],STDtensor2)
         end
         --table.insert(inputset,TYPtensor2)
--      elseif t<4 then
--         for i=1,numatoms[t] do
--             for j=1,numsyms do
--                 KIMcount=KIMcount+1
--  --            print("KIMinput: " .. KIMinput[KIMcount])
--                 TYPtensor3[i][j]=KIMinput[KIMcount]
--             end
--         end
--         table.insert(inputset,TYPtensor3)
--      elseif t<5 then
--         for i=1,numatoms[t] do
--             for j=1,numsyms do
--                 KIMcount=KIMcount+1
--  --            print("KIMinput: " .. KIMinput[KIMcount])
--                 TYPtensor4[i][j]=KIMinput[KIMcount]
--             end
--         end
--         table.insert(inputset,TYPtensor4)
--      else
--         for i=1,numatoms[t] do
--             for j=1,numsyms do
--                 KIMcount=KIMcount+1
--  --            print("KIMinput: " .. KIMinput[KIMcount])
--                 TYPtensor5[i][j]=KIMinput[KIMcount]
--             end
--         end
--         table.insert(inputset,TYPtensor5)
      end
  end
  print('[ Lua ANN Driver ] : Inputset size: ' .. #inputset)
  if mlp then 
     --table.insert(input,{inputset})
     --print(input[1][1])
     drvmodel=mlp.modules[1]
     --drvmodel=mlp
     --print(drvmodel)
     -- Feed network with inputs and get total output
     --print('[ Lua ANN Driver ] : Forward MLP.')
     if numtypes<2 then
        totalout=mlp:forward(TYPtensor1)
     elseif numtypes<3 then
        totalout=mlp:forward({TYPtensor1,TYPtensor2})
     end
     --totaloutGPU=mlp:forward(inputset:cuda())
     --totalout=totaloutGPU:double()
     print('[ Lua ANN Driver ] : Output: ' .. totalout[1])
     -- Feed network with inputs and get individual outputs
     --input = {inputset[1],inputset[2]}
     --outputs=drvmodel:forward(input[1][1])
     --outputs=drvmodel:forward(inputset)
     if numtypes<2 then
        outputs=drvmodel:forward(TYPtensor1)
     elseif numtypes<3 then
        outputs=drvmodel:forward({TYPtensor1,TYPtensor2})
     end
     --outputsGPU=drvmodel:forward(inputset:cuda())
     --outputs=outputsGPU:double()
     print('[ Lua ANN Driver ] : Outputs: ')
     --print(outputs[1])
     --print(outputs[2])
     --table.insert(gradOut,{grad})
     --print(gradOut)
     --print(gradOut[2])
     print('[ Lua ANN Driver ] : Outputs printed. ')
     count=3
     totalenergy=0.0
     print('[ Lua ANN Driver ] : Packing data to send to C.')
     for t=1,numtypes do
        drvnn=drvmodel.modules[t]
         --for i=1,numatoms[t] do
            if t<2 then
             outdrv1=drvnn:forward(TYPtensor1)
         for i=1,numatoms[t] do
             count=count+1
             --print('[ Lua ANN Driver ] : Pack typ ' .. t .. ' no ' .. i)
             --KIMinput[count]=outputs[t][i][1]
             KIMinput[count]=outdrv1[i][1]
             --totalenergy=totalenergy+outputs[t][i][1] 
             totalenergy=totalenergy+outdrv1[i][1] 
         end
            elseif t<3 then
             outdrv2=drvnn:forward(TYPtensor2)
         for i=1,numatoms[t] do
             count=count+1
             --print('[ Lua ANN Driver ] : Pack typ ' .. t .. ' no ' .. i)
             KIMinput[count]=outdrv2[i][1]
             totalenergy=totalenergy+outdrv2[i][1]
         end
            end
     end
     print('[ Lua ANN Driver ] : Total energy:' .. totalenergy)
     KIMinput[2]=totalenergy
     -- Compute the gradient d_output/d_input using automatic derivation of Torch7 for MLP
     -- See page 543 at
     -- Ronan Collobert, Koray Kavukcuoglu, and Clement Farabet, "Implementing Neural Networks Efficiently" in G. Montavon et al. (Eds.): Neural Networks: Tricks of the Trade, 2nd Edition, LNCS 7700, (Springer-Verlag, Berlin Heidelberg, 2012) Chapter 21, Pages 537-557.
     for t=1,numtypes do
         drvnn=drvmodel.modules[t]
         for i=1,numatoms[t] do
            --print('Atom:' .. i)
            if t<2 then
               outdrv1=drvnn:forward(TYPtensor1[i])
               dOutdIn1=drvnn:updateGradInput(TYPtensor1[i],gOutTensor1)
               DdOutdIn1=dOutdIn1:double()
               dOutdIn1=torch.cmul(DdOutdIn1,STDtensor1)
               DdOutdIn1:mv(PCAtensorT1,dOutdIn1)
               for j=1,numsyms do
                   count=count+1
                   KIMinput[count]=DdOutdIn1[j]
               end
            elseif t<3 then
               outdrv2=drvnn:forward(TYPtensor2[i])
               dOutdIn2=drvnn:updateGradInput(TYPtensor2[i],gOutTensor2)
               DdOutdIn2=dOutdIn2:double()
               dOutdIn2=torch.cmul(DdOutdIn2,STDtensor2)
               DdOutdIn2:mv(PCAtensorT2,dOutdIn2)
               for j=1,numsyms do
                   count=count+1
                   KIMinput[count]=DdOutdIn2[j]
               end
            end
         end
     end
     --print(dOutdIn1)
     --print(dOutdIn2)
     --print(dOutdIn[1])
     --print(dOutdIn[2])
     --print(dOutdIn[2][16][82])
--     for t=1,numtypes do
--        if t<2 then
--         for i=1,numatoms[t] do
--             for j=1,numsyms do
--                 count=count+1
--                 KIMinput[count]=dOutdIn1[i][j]
--             end
--         end
--        elseif t<3 then
--         for i=1,numatoms[t] do
--             for j=1,numsyms do
--                 count=count+1
--                 KIMinput[count]=dOutdIn2[i][j]
--             end
--         end
--        end
--     end
     --print(KIMinput[count])
     print('[ Lua ANN Driver ] : Everything is Perfect! Returning to C.')
     end_time = os.clock()
     KIMinput[1] = end_time - start_time
     memsize = collectgarbage('count')
     KIMinput[3] = memsize
     print(string.format("[ Lua ANN Driver ] : Compute time at Lua Driver:  %.4f\n",end_time - start_time))
     --mlp2=torch.load('NNparams.mlp','ascii')
     --mlp2=torch.load('NNparams.mlp')
     --totalout=mlp2:forward(inputset)
     --print('[ Lua ANN Driver ] : Output: ' .. totalout[1])
     print('[ Lua ANN Driver ] : Garbage Size: ' .. memsize)
     ier=0
     return ier
  else
     ier=-5
     return ier
  end
end

function trash_collect(anyinput)
  print('[ Lua ANN Driver ] : trash collecting... ')
  torch.setdefaulttensortype('torch.DoubleTensor')
  if mlp then
     strmap=nil
     str=nil
     strfile=nil
  end 
  if KIMpca then
     KIMpca=nil
  end 
  collectgarbage()
  memsize = collectgarbage('count')
  print('[ Lua ANN Driver ] : trash collected: ' .. memsize)
end

ier=load_net()
ier=init_net(0)
return ier

