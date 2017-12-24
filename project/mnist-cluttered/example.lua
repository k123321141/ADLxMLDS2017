--[[
Copyright 2014 Google Inc. All Rights Reserved.

Use of this source code is governed by a BSD-style
license that can be found in the LICENSE file or at
https://developers.google.com/open-source/licenses/bsd
]]

local mnist_cluttered = require 'mnist_cluttered'
require 'image'

local dataConfig = {megapatch_w=60, num_dist=8}
local dataInfo = mnist_cluttered.createData(dataConfig)
local n = 10000 
for i=1,n do
    local observation, target = unpack(dataInfo.nextExample())
    
    --print("observation size:", table.concat(observation:size():totable(), 'x'))
    --
    _,idx = torch.max(target,1)
    idx = idx:max()
    idx = idx - 1
    --print("targets:", target)
    --print("targets:", idx)

    --print("Saving example.png")
    local formatted = image.toDisplayTensor({input=observation})
    image.save("./data/example_" .. i .. "_" .. idx ..".png", formatted)
    print(i .. "/" .. n)
end

