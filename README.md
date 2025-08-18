# vulkanComputeLinearRegression
A repository to demonstrate an exercise in Linear Regression using Vulkan processing

# INSTRUCTIONS

- Create directory "bin" in root folder
- Change working directory to "bin"
- Run command "cmake .."
- Run command "make"
- Run commmand "./main"

You should see the weight and bias terms output for the linear regression on the command line. Data can be changed in the "main.cpp" file

# GENERAL VULKAN WORKFLOW:

## Initialize Vulkan Instance:
- Input the required metadata and create the vulkan instance

## Connect to Physical Device:
- Enumerate available devices/gpu's, and select based on index, or other factors based on device properties

## Find Queue Family on Physical Device:
- Find queue family from physical device properties. The Queue family will be used for logical device instance creation, command pool creation for command buffer recording, and final queue submission

## Create Logical Instance of Device:
- Use queue family from physical device properties to create the logical instance of device

## Allocating memory, binding and submitting:
- Create the required buffers on the host for the vulkan context 
- Allocate the memory buffers to gpu memory using required memory type from host version of buffer  
- Bind the buffers to the gpu/context
- Create shader modules by reading shader code
- Create descriptor set layout
- Create Pipeline 
- Create descriptor sets along with descriptor set update layouts
- Finally, record command buffer to bind pipeline > bind descriptor sets > dispatch gpu resources > submit queue

More information regarding documentation and definitions can be found in https://docs.vulkan.org/spec/latest/appendices/glossary.html

A good resource for learning graphics language concepts is also at https://paroj.github.io/gltut/

# FUTURE WORK

- Code is kind of bloated, but straightforward for learning purposes. Can be simplified a lot by using functions to do repetitive tasks
- Adding a way for user to input data from a file
- Introducing staging buffers and other optimization methods

