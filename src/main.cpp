#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <chrono>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <array>
#include <optional>
#include <set>
#include <random>

const std::vector<const char*> validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

const std::vector<const char*> deviceExtensions = {
    VK_KHR_SWAPCHAIN_EXTENSION_NAME
};

#ifdef NDEBUG
const bool enableValidationLayers = false;
#else
const bool enableValidationLayers = true;
#endif

const int layerDim = 1;
const int N = 12;

struct inputParams{
	float weights[layerDim] = {0.5};
	float biases[layerDim] = {0.5};
	float x[N] = {0.03, 0.19, 0.34, 0.46, 0.78, 0.81, 1.08, 1.18, 1.39, 1.60, 1.65, 1.90};	
	float y[N] = {0.67, 0.85, 1.05, 1.0, 1.40, 1.5, 1.3, 1.54, 1.55, 1.68, 1.73, 1.6 };		
//	float y[N] = {0.09, 0.57, 1.02, 1.38, 2.34, 2.43, 3.24, 3.54, 4.17, 4.8, 4.95, 5.7 };		
	float lr = 0.01;
};


void submitQueue(VkDevice device, uint32_t ComputeQueueFamilyIndex, VkPipeline ComputePipeline, VkPipelineLayout PipelineLayout, VkDescriptorSet DescriptorSet){

	VkQueue Queue;	
	VkFence Fence;
	VkCommandPool CommandPool;
	VkCommandBuffer CmdBuffer;

	VkCommandPoolCreateInfo CommandPoolCreateInfo{};
	CommandPoolCreateInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
	CommandPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
	CommandPoolCreateInfo.queueFamilyIndex = ComputeQueueFamilyIndex;	

	vkCreateCommandPool(device, &CommandPoolCreateInfo, nullptr, &CommandPool);
	
	//initialize command buffers to record. The number of command buffers will determine allocation size
	std::vector<VkCommandBuffer> CmdBuffers; 
	CmdBuffers.resize(1);
	VkCommandBufferAllocateInfo allocInfo{};
	allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
	allocInfo.commandPool = CommandPool;
	allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
	allocInfo.commandBufferCount = (uint32_t)CmdBuffers.size();

	vkAllocateCommandBuffers(device, &allocInfo, CmdBuffers.data());
	CmdBuffer = CmdBuffers[0];   

	VkFenceCreateInfo fenceInfo{};//we use a fence to know from the host when the gpu has finished processing
	fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
	fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;	

	vkCreateFence(device, &fenceInfo, nullptr, &Fence);
	vkGetDeviceQueue(device, ComputeQueueFamilyIndex, 0, &Queue);

	VkCommandBufferBeginInfo beginInfo{};
	beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
	// vkCmd* functions will take the selected command buffer as input and record the step
	vkBeginCommandBuffer(CmdBuffer, &beginInfo);

	// has descriptor set layout and shader stage info (bind shader code etc. to command buffer, "use this pipeline to do the next dispatch")
	vkCmdBindPipeline(CmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, ComputePipeline); 
	
	// has descriptor set layout information (bind descriptor set to pipeline layout)
	vkCmdBindDescriptorSets(CmdBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, PipelineLayout, 0, 1, &DescriptorSet, 0, nullptr);

	vkCmdDispatch(CmdBuffer, 32, 1, 1); 

	vkEndCommandBuffer(CmdBuffer);
	
	VkSubmitInfo submitInfo{};
	submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
	submitInfo.commandBufferCount = 1;
	submitInfo.pCommandBuffers = &CmdBuffer;

	vkQueueSubmit(Queue, 1, &submitInfo, Fence);
	
	vkWaitForFences(device, 1, &Fence, VK_TRUE, UINT64_MAX);
	
	vkDestroyFence(device, Fence, nullptr);
	vkDestroyCommandPool(device, CommandPool, nullptr);

}

void allocateBindAndSubmit(VkPhysicalDevice PhysicalDevice, VkDevice device, uint32_t ComputeQueueFamilyIndex, inputParams inputs, int epochs)
{
	// SUMMARY: 
	// Create the required buffers for the vulkan application 
    	// Allocate the memory buffers to gpu memory using required memory type from host version of buffer
    	// Bind the buffers to to gpu/context 
	const uint32_t BufferSizeWeight = sizeof(inputs.weights) * sizeof(float);
	const uint32_t BufferSizeBatch = sizeof(float) * N;
	const uint32_t BufferSizeOthers = sizeof(float);

	VkBufferCreateInfo BufferCreateInfoWeight{};
	BufferCreateInfoWeight.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	BufferCreateInfoWeight.size = BufferSizeWeight;
	BufferCreateInfoWeight.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	BufferCreateInfoWeight.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	
	VkBufferCreateInfo BufferCreateInfoBiases{};
	BufferCreateInfoBiases.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	BufferCreateInfoBiases.size = BufferSizeOthers;
	BufferCreateInfoBiases.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	BufferCreateInfoBiases.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	VkBufferCreateInfo BufferCreateInfoBatch{};
	BufferCreateInfoBatch.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	BufferCreateInfoBatch.size = BufferSizeBatch;
	BufferCreateInfoBatch.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	BufferCreateInfoBatch.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	
	VkBufferCreateInfo BufferCreateInfoOthers{};
	BufferCreateInfoOthers.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	BufferCreateInfoOthers.size = BufferSizeOthers;
	BufferCreateInfoOthers.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
	BufferCreateInfoOthers.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

	VkBuffer weightBuffer;
	VkBuffer biasBuffer;
	VkBuffer lrBuffer;
	VkBuffer yBuffer;
	VkBuffer xBuffer;
	VkBuffer dEdWBuffer;
	VkBuffer dEdBBuffer;

	vkCreateBuffer(device, &BufferCreateInfoWeight, nullptr, &weightBuffer);
	vkCreateBuffer(device, &BufferCreateInfoBiases, nullptr, &biasBuffer);
	vkCreateBuffer(device, &BufferCreateInfoBatch, nullptr, &yBuffer); 
	vkCreateBuffer(device, &BufferCreateInfoBatch, nullptr, &xBuffer); 
	vkCreateBuffer(device, &BufferCreateInfoBatch, nullptr, &dEdWBuffer); 
	vkCreateBuffer(device, &BufferCreateInfoBatch, nullptr, &dEdBBuffer); 
	vkCreateBuffer(device, &BufferCreateInfoOthers, nullptr, &lrBuffer); 
 
	
	// Memory req
	VkMemoryRequirements WeightBufferMemoryRequirements;
	VkMemoryRequirements OtherBufferMemoryRequirements;	
	VkMemoryRequirements BatchBufferMemoryRequirements;		

	vkGetBufferMemoryRequirements(device, weightBuffer, &WeightBufferMemoryRequirements);
	vkGetBufferMemoryRequirements(device, lrBuffer, &OtherBufferMemoryRequirements);
	vkGetBufferMemoryRequirements(device, yBuffer, &BatchBufferMemoryRequirements);

	// query
	VkPhysicalDeviceMemoryProperties MemoryProperties;
	vkGetPhysicalDeviceMemoryProperties(PhysicalDevice, &MemoryProperties);        
	VkMemoryPropertyFlags properties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

	uint32_t MemoryTypeIndex = uint32_t(~0);
	VkDeviceSize MemoryHeapSize = uint32_t(~0);
	VkMemoryType MemoryType;

	for (uint32_t i = 0; i < MemoryProperties.memoryTypeCount; i++) {
		//check first condition below
		if ((WeightBufferMemoryRequirements.memoryTypeBits & (1 << i)) && (MemoryProperties.memoryTypes[i].propertyFlags & properties) == properties) {
			MemoryType = MemoryProperties.memoryTypes[i];
			MemoryHeapSize = MemoryProperties.memoryHeaps[MemoryType.heapIndex].size;
		    MemoryTypeIndex = i;
		    break;            
		}
	}

	VkMemoryAllocateInfo WeightBufferMemoryAllocateInfo{};
	WeightBufferMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	WeightBufferMemoryAllocateInfo.allocationSize = WeightBufferMemoryRequirements.size;
	WeightBufferMemoryAllocateInfo.memoryTypeIndex = MemoryTypeIndex;
	
	VkMemoryAllocateInfo BiasBufferMemoryAllocateInfo{};
	BiasBufferMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	BiasBufferMemoryAllocateInfo.allocationSize = OtherBufferMemoryRequirements.size;
	BiasBufferMemoryAllocateInfo.memoryTypeIndex = MemoryTypeIndex;

	VkMemoryAllocateInfo OtherBufferMemoryAllocateInfo{};
	OtherBufferMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	OtherBufferMemoryAllocateInfo.allocationSize = OtherBufferMemoryRequirements.size;
	OtherBufferMemoryAllocateInfo.memoryTypeIndex = MemoryTypeIndex;
	
	VkMemoryAllocateInfo BatchBufferMemoryAllocateInfo{};
	BatchBufferMemoryAllocateInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	BatchBufferMemoryAllocateInfo.allocationSize = BatchBufferMemoryRequirements.size;
	BatchBufferMemoryAllocateInfo.memoryTypeIndex = MemoryTypeIndex;
	
	VkDeviceMemory weightBufferMemory;
	VkDeviceMemory biasBufferMemory;
	VkDeviceMemory yBufferMemory;
	VkDeviceMemory lrBufferMemory;
	VkDeviceMemory xBufferMemory;
	VkDeviceMemory dEdWBufferMemory;
	VkDeviceMemory dEdBBufferMemory;
	
	vkAllocateMemory(device, &WeightBufferMemoryAllocateInfo, nullptr, &weightBufferMemory);
	vkAllocateMemory(device, &BiasBufferMemoryAllocateInfo, nullptr, &biasBufferMemory);
	vkAllocateMemory(device, &BatchBufferMemoryAllocateInfo, nullptr, &yBufferMemory);
	vkAllocateMemory(device, &BatchBufferMemoryAllocateInfo, nullptr, &xBufferMemory);
	vkAllocateMemory(device, &BatchBufferMemoryAllocateInfo, nullptr, &dEdWBufferMemory);
	vkAllocateMemory(device, &BatchBufferMemoryAllocateInfo, nullptr, &dEdBBufferMemory);
	vkAllocateMemory(device, &OtherBufferMemoryAllocateInfo, nullptr, &lrBufferMemory);	

	std::cout << "Memory Type Index: " << MemoryTypeIndex << std::endl;
	std::cout << "Memory Heap Size : " << MemoryHeapSize / 1024 / 1024 / 1024  << " GB" << std::endl;
	
	void* InBufferPtrAddr; // will be used to map gpu memory back to host, so we can print results
	float* InBufferPtr; //for casting the pointer to the proper datatype for printing information
	VkDescriptorSetLayout DescriptorSetLayout;	
	VkPipelineLayout PipelineLayout;
	VkShaderModule ShaderModule;
	VkPipeline ComputePipeline;
	VkDescriptorPool DescriptorPool;
	std::vector<VkDescriptorSet> DescriptorSets;
	DescriptorSets.resize(1);
		
	vkMapMemory(device, weightBufferMemory, 0, BufferSizeWeight, 0, &InBufferPtrAddr);
	memcpy(InBufferPtrAddr, inputs.weights, (size_t)BufferSizeWeight);
	vkUnmapMemory(device, weightBufferMemory); 

	vkMapMemory(device, biasBufferMemory, 0, BufferSizeOthers, 0, &InBufferPtrAddr);
	memcpy(InBufferPtrAddr, inputs.biases, (size_t)BufferSizeOthers);
	vkUnmapMemory(device, biasBufferMemory);
		
	vkMapMemory(device, lrBufferMemory, 0, BufferSizeOthers, 0, &InBufferPtrAddr);
	memcpy(InBufferPtrAddr, &inputs.lr, (size_t)BufferSizeOthers);
	vkUnmapMemory(device, lrBufferMemory); 
	
	vkMapMemory(device, yBufferMemory, 0, BufferSizeBatch, 0, &InBufferPtrAddr);
	memcpy(InBufferPtrAddr, inputs.y, (size_t)BufferSizeBatch);
	vkUnmapMemory(device, yBufferMemory); 
	
	vkMapMemory(device, xBufferMemory, 0, BufferSizeBatch, 0, &InBufferPtrAddr);
	memcpy(InBufferPtrAddr, inputs.x, (size_t)BufferSizeBatch);
	vkUnmapMemory(device, xBufferMemory);  
	// "Binding" memory means making the buffer visible to the vulkan context and GPU
	vkBindBufferMemory(device, weightBuffer, weightBufferMemory, 0);
	vkBindBufferMemory(device, biasBuffer, biasBufferMemory, 0);
	vkBindBufferMemory(device, lrBuffer, lrBufferMemory, 0);
	vkBindBufferMemory(device, yBuffer, yBufferMemory, 0);
	vkBindBufferMemory(device, xBuffer, xBufferMemory, 0);
	vkBindBufferMemory(device, dEdWBuffer, dEdWBufferMemory, 0);
	vkBindBufferMemory(device, dEdBBuffer, dEdBBufferMemory, 0);

	for(int i = 0; i < epochs; i++){	


		////////////////////////////////////////////////////////////////////////
		// CONFIGURE PIPELINE (NEED DESCRIPTOR LAYOUTS AND SHADER MODULES)    //
		////////////////////////////////////////////////////////////////////////

		// Shader module
		std::vector<char> ShaderContents;
		if (std::ifstream ShaderFile{ "shaders/shader.comp.spv", std::ios::binary | std::ios::ate }) {
			const size_t FileSize = ShaderFile.tellg();
			ShaderFile.seekg(0);
			ShaderContents.resize(FileSize, '\0');
			ShaderFile.read(ShaderContents.data(), FileSize);
		}

		VkShaderModuleCreateInfo ShaderModuleCreateInfo{};
		ShaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
		ShaderModuleCreateInfo.codeSize = ShaderContents.size();
		ShaderModuleCreateInfo.pCode = reinterpret_cast<const uint32_t*>(ShaderContents.data());


		vkCreateShaderModule(device, &ShaderModuleCreateInfo, nullptr, &ShaderModule);        

		// Descriptor Set Layout
		// The layout of data to be passed to pipeline

		std::array<VkDescriptorSetLayoutBinding, 7> DescriptorSetLayoutBinding{};
		DescriptorSetLayoutBinding[0].binding = 0;
		DescriptorSetLayoutBinding[0].descriptorCount = 1;
		DescriptorSetLayoutBinding[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		DescriptorSetLayoutBinding[0].pImmutableSamplers = nullptr;
		DescriptorSetLayoutBinding[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		DescriptorSetLayoutBinding[1].binding = 1;
		DescriptorSetLayoutBinding[1].descriptorCount = 1;
		DescriptorSetLayoutBinding[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		DescriptorSetLayoutBinding[1].pImmutableSamplers = nullptr;
		DescriptorSetLayoutBinding[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		
		DescriptorSetLayoutBinding[2].binding = 2;
		DescriptorSetLayoutBinding[2].descriptorCount = 1;
		DescriptorSetLayoutBinding[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		DescriptorSetLayoutBinding[2].pImmutableSamplers = nullptr;
		DescriptorSetLayoutBinding[2].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		DescriptorSetLayoutBinding[3].binding = 3;
		DescriptorSetLayoutBinding[3].descriptorCount = 1;
		DescriptorSetLayoutBinding[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		DescriptorSetLayoutBinding[3].pImmutableSamplers = nullptr;
		DescriptorSetLayoutBinding[3].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		
		DescriptorSetLayoutBinding[4].binding = 4;
		DescriptorSetLayoutBinding[4].descriptorCount = 1;
		DescriptorSetLayoutBinding[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		DescriptorSetLayoutBinding[4].pImmutableSamplers = nullptr;
		DescriptorSetLayoutBinding[4].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		
		DescriptorSetLayoutBinding[5].binding = 5;
		DescriptorSetLayoutBinding[5].descriptorCount = 1;
		DescriptorSetLayoutBinding[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		DescriptorSetLayoutBinding[5].pImmutableSamplers = nullptr;
		DescriptorSetLayoutBinding[5].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
		
		DescriptorSetLayoutBinding[6].binding = 6;
		DescriptorSetLayoutBinding[6].descriptorCount = 1;
		DescriptorSetLayoutBinding[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		DescriptorSetLayoutBinding[6].pImmutableSamplers = nullptr;
		DescriptorSetLayoutBinding[6].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

		VkDescriptorSetLayoutCreateInfo DescriptorSetLayoutCreateInfo{};
		DescriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
		DescriptorSetLayoutCreateInfo.bindingCount = 7;
		DescriptorSetLayoutCreateInfo.pBindings = DescriptorSetLayoutBinding.data();

		vkCreateDescriptorSetLayout(device, &DescriptorSetLayoutCreateInfo, nullptr, &DescriptorSetLayout);

		VkPipelineShaderStageCreateInfo PipelineShaderCreateInfo{};
		PipelineShaderCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
		PipelineShaderCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		PipelineShaderCreateInfo.module = ShaderModule;
		PipelineShaderCreateInfo.pName = "main";

		VkPipelineLayoutCreateInfo PipelineLayoutCreateInfo{};
		PipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
		PipelineLayoutCreateInfo.setLayoutCount = 1;
		PipelineLayoutCreateInfo.pSetLayouts = &DescriptorSetLayout;		
		
		vkCreatePipelineLayout(device, &PipelineLayoutCreateInfo, nullptr, &PipelineLayout);
		// Pipeline Layout
		VkComputePipelineCreateInfo ComputePipelineCreateInfo{};
		ComputePipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
		ComputePipelineCreateInfo.layout = PipelineLayout;
		ComputePipelineCreateInfo.stage = PipelineShaderCreateInfo;
			
		vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &ComputePipelineCreateInfo, nullptr, &ComputePipeline);	

		////////////////////////////////////////////////////////////////////////
		//                CONFIGURE DESCRIPTOR SETS                           //
		////////////////////////////////////////////////////////////////////////		

		VkDescriptorPoolSize DescriptorPoolSize;
		DescriptorPoolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		DescriptorPoolSize.descriptorCount = static_cast<uint32_t>(7);

		VkDescriptorPoolCreateInfo DescriptorPoolCreateInfo{};
		DescriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
		DescriptorPoolCreateInfo.poolSizeCount = 1;
		DescriptorPoolCreateInfo.maxSets = 1;
		DescriptorPoolCreateInfo.pPoolSizes = &DescriptorPoolSize;		
		
		vkCreateDescriptorPool(device, &DescriptorPoolCreateInfo, nullptr, &DescriptorPool);        

		VkDescriptorSetAllocateInfo DescriptorSetAllocInfo{};
		DescriptorSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
		DescriptorSetAllocInfo.descriptorPool = DescriptorPool;
		DescriptorSetAllocInfo.descriptorSetCount = static_cast<uint32_t>(1);
		DescriptorSetAllocInfo.pSetLayouts = &DescriptorSetLayout;

		vkAllocateDescriptorSets(device, &DescriptorSetAllocInfo, DescriptorSets.data());
		
		VkDescriptorBufferInfo WeightBufferInfo{};
		WeightBufferInfo.buffer = weightBuffer;
		WeightBufferInfo.offset = 0;
		WeightBufferInfo.range = sizeof(inputs.weights) * sizeof(float);
		
		VkDescriptorBufferInfo BiasBufferInfo{};
		BiasBufferInfo.buffer = biasBuffer;
		BiasBufferInfo.offset = 0;
		BiasBufferInfo.range = sizeof(inputs.biases) * sizeof(float);
		
		VkDescriptorBufferInfo LrBufferInfo{};
		LrBufferInfo.buffer = lrBuffer;
		LrBufferInfo.offset = 0;
		LrBufferInfo.range = sizeof(float);
		
		VkDescriptorBufferInfo yBufferInfo{};
		yBufferInfo.buffer = yBuffer;
		yBufferInfo.offset = 0;
		yBufferInfo.range = sizeof(float) * N;
		
		VkDescriptorBufferInfo xBufferInfo{};
		xBufferInfo.buffer = xBuffer;
		xBufferInfo.offset = 0;
		xBufferInfo.range = sizeof(float) * N;
		
		VkDescriptorBufferInfo dEdWBufferInfo{};
		dEdWBufferInfo.buffer = dEdWBuffer;
		dEdWBufferInfo.offset = 0;
		dEdWBufferInfo.range = sizeof(float) * N;
		
		VkDescriptorBufferInfo dEdBBufferInfo{};
		dEdBBufferInfo.buffer = dEdBBuffer;
		dEdBBufferInfo.offset = 0;
		dEdBBufferInfo.range = sizeof(float) * N;
			
		std::array<VkWriteDescriptorSet, 7> WriteDescriptorSets{};

		WriteDescriptorSets[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		WriteDescriptorSets[0].dstSet = DescriptorSets[0];
		WriteDescriptorSets[0].dstBinding = 0; // corresponds to shader binding number
		WriteDescriptorSets[0].dstArrayElement = 0;
		WriteDescriptorSets[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		WriteDescriptorSets[0].descriptorCount = 1;
		WriteDescriptorSets[0].pBufferInfo = &WeightBufferInfo; // was binded to the GPU memory "VkDeviceMemory" object after creating the buffers and allocating to device
		
		WriteDescriptorSets[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		WriteDescriptorSets[1].dstSet = DescriptorSets[0];
		WriteDescriptorSets[1].dstBinding = 1;
		WriteDescriptorSets[1].dstArrayElement = 0;
		WriteDescriptorSets[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		WriteDescriptorSets[1].descriptorCount = 1;
		WriteDescriptorSets[1].pBufferInfo = &BiasBufferInfo; 
		
		WriteDescriptorSets[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		WriteDescriptorSets[2].dstSet = DescriptorSets[0];
		WriteDescriptorSets[2].dstBinding = 2;
		WriteDescriptorSets[2].dstArrayElement = 0;
		WriteDescriptorSets[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		WriteDescriptorSets[2].descriptorCount = 1;
		WriteDescriptorSets[2].pBufferInfo = &yBufferInfo; 


		
		WriteDescriptorSets[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		WriteDescriptorSets[3].dstSet = DescriptorSets[0];
		WriteDescriptorSets[3].dstBinding = 3; // corresponds to shader binding number
		WriteDescriptorSets[3].dstArrayElement = 0;
		WriteDescriptorSets[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		WriteDescriptorSets[3].descriptorCount = 1;
		WriteDescriptorSets[3].pBufferInfo = &LrBufferInfo; 


		WriteDescriptorSets[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		WriteDescriptorSets[4].dstSet = DescriptorSets[0];
		WriteDescriptorSets[4].dstBinding = 4;
		WriteDescriptorSets[4].dstArrayElement = 0;
		WriteDescriptorSets[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		WriteDescriptorSets[4].descriptorCount = 1;
		WriteDescriptorSets[4].pBufferInfo = &xBufferInfo; 
		
		WriteDescriptorSets[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		WriteDescriptorSets[5].dstSet = DescriptorSets[0];
		WriteDescriptorSets[5].dstBinding = 5;
		WriteDescriptorSets[5].dstArrayElement = 0;
		WriteDescriptorSets[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		WriteDescriptorSets[5].descriptorCount = 1;
		WriteDescriptorSets[5].pBufferInfo = &dEdWBufferInfo; 
		
		WriteDescriptorSets[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		WriteDescriptorSets[6].dstSet = DescriptorSets[0];
		WriteDescriptorSets[6].dstBinding = 6;
		WriteDescriptorSets[6].dstArrayElement = 0;
		WriteDescriptorSets[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		WriteDescriptorSets[6].descriptorCount = 1;
		WriteDescriptorSets[6].pBufferInfo = &dEdBBufferInfo; 
		
	  	vkUpdateDescriptorSets(device, 7, WriteDescriptorSets.data(), 0, nullptr); 

		//////////////////////////////////////////////////////////////////////////////////////////
		//            	        	SUBMIT QUEUE TO GPU                                     //
		//(create and record command buffer > bind pipeline > bind descriptor sets > dispatch)//
		///////////////////////////////////////////////////////////////////////////////////////

	    	float gradientSum = 0;
	    	// record command buffer, bind pipeline, bind descriptor sets, dispatch gpu resources, and submit queue
		submitQueue(device, ComputeQueueFamilyIndex, ComputePipeline, PipelineLayout, DescriptorSets[0]);

		//////////////////////////////////////////////////////////////////////////////////////////
		//            	        	PRINTING RESULTS                                        //		
		///////////////////////////////////////////////////////////////////////////////////////  
		  
		std::cout << std::endl;		

		vkMapMemory(device, weightBufferMemory, 0, BufferSizeWeight, 0, &InBufferPtrAddr);
		InBufferPtr = static_cast<float*> (InBufferPtrAddr); // ++ ss	
		std::cout << "WEIGHT: ";	
		std::cout << InBufferPtr[0] << " ";
		std::cout << std::endl;
		inputs.weights[0] = InBufferPtr[0]; //for memcpy at the start of the loop
		vkUnmapMemory(device, weightBufferMemory);		
		
		vkMapMemory(device, dEdWBufferMemory, 0, BufferSizeBatch, 0, &InBufferPtrAddr);
		InBufferPtr = static_cast<float*> (InBufferPtrAddr); // ++ ss	
		std::cout << "Total error gradient (weights): ";	
		for(int i = 0; i < N; i++){
			//std::cout << InBufferPtr[i] << " ";	//dEdW per point
			gradientSum += InBufferPtr[i];	
			}
		std::cout << gradientSum << " ";
		std::cout << std::endl;
		vkUnmapMemory(device, dEdWBufferMemory);
		inputs.weights[0] -= inputs.lr * gradientSum; //update weights for next batch iteration
		
		vkMapMemory(device, weightBufferMemory, 0, BufferSizeWeight, 0, &InBufferPtrAddr);
		memcpy(InBufferPtrAddr, inputs.weights, (size_t)BufferSizeWeight); //copy updated weights back to shader
		vkUnmapMemory(device, weightBufferMemory); 
		
		gradientSum = 0;
		
		vkMapMemory(device, biasBufferMemory, 0, BufferSizeOthers, 0, &InBufferPtrAddr);
		InBufferPtr = static_cast<float*> (InBufferPtrAddr); // ++ ss	
		std::cout << "BIAS: ";	
		std::cout << InBufferPtr[0] << " ";
		std::cout << std::endl;
		inputs.biases[0] = InBufferPtr[0]; //for memcpy at the start of the loop
		vkUnmapMemory(device, biasBufferMemory);
		
		vkMapMemory(device, dEdBBufferMemory, 0, BufferSizeBatch, 0, &InBufferPtrAddr);
		InBufferPtr = static_cast<float*> (InBufferPtrAddr); // ++ ss	
		std::cout << "Total error gradient (biases): ";	
		for(int i = 0; i < N; i++){
			//std::cout << InBufferPtr[i] << " ";	//dEdW per point
			gradientSum += InBufferPtr[i];	
			}
		std::cout << gradientSum << " ";
		std::cout << std::endl;
		vkUnmapMemory(device, dEdBBufferMemory);
		inputs.biases[0] -= inputs.lr * gradientSum; //update weights for next batch iteration		
		
		vkMapMemory(device, biasBufferMemory, 0, BufferSizeOthers, 0, &InBufferPtrAddr);
		memcpy(InBufferPtrAddr, inputs.biases, (size_t)BufferSizeOthers); //copy updated biases back to shader
		vkUnmapMemory(device, biasBufferMemory);			
	}
		
	
	////////////////////////////////////////////////////////////////////////
	//                              CLEANUP                               //
	////////////////////////////////////////////////////////////////////////

	vkDestroyPipelineLayout(device, PipelineLayout, nullptr);
	vkDestroyShaderModule(device, ShaderModule, nullptr);
	vkDestroyPipeline(device, ComputePipeline, nullptr);

	vkDestroyDescriptorPool(device, DescriptorPool, nullptr);
	vkDestroyDescriptorSetLayout(device, DescriptorSetLayout, nullptr);        
	vkDestroyBuffer(device, weightBuffer, nullptr);
	vkDestroyBuffer(device, lrBuffer, nullptr);
	vkDestroyBuffer(device, yBuffer, nullptr);
	vkDestroyBuffer(device, xBuffer, nullptr);
	
	vkFreeMemory(device, weightBufferMemory, nullptr);     
	vkFreeMemory(device, lrBufferMemory, nullptr); 
	vkFreeMemory(device, yBufferMemory, nullptr);     
	vkFreeMemory(device, xBufferMemory, nullptr);     
	
//	vkDestroyFence(device, Fence, nullptr);
//	vkDestroyCommandPool(device, CommandPool, nullptr);
	vkDestroyDevice(device, nullptr);

}
	
void vulkanLinearRegression(inputParams inputs, int epochs){
	////////////////////////////////////////////////////////////////////////
	//                          VULKAN INSTANCE                           //
	////////////////////////////////////////////////////////////////////////
	VkInstance instance;

        VkApplicationInfo appInfo{};
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "VulkanCompute";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_1;

        VkInstanceCreateInfo InstanceCreateInfo{};
        InstanceCreateInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        InstanceCreateInfo.pApplicationInfo = &appInfo;        

	vkCreateInstance(&InstanceCreateInfo, nullptr, &instance);

	////////////////////////////////////////////////////////////////////////
	//                     PHYSICAL DEVICE (GPU)                          //
	////////////////////////////////////////////////////////////////////////
	VkPhysicalDevice PhysicalDevice;
	u_int32_t deviceCount = 1;
	vkEnumeratePhysicalDevices(instance, &deviceCount, &PhysicalDevice);
	
	VkPhysicalDeviceProperties DeviceProps;
	vkGetPhysicalDeviceProperties(PhysicalDevice, &DeviceProps);
	std::cout << "Device Name    : " << DeviceProps.deviceName << std::endl;
	const uint32_t ApiVersion = DeviceProps.apiVersion;
	std::cout << "Vulkan Version : " << VK_VERSION_MAJOR(ApiVersion) << "." << VK_VERSION_MINOR(ApiVersion) << "." << VK_VERSION_PATCH(ApiVersion) << std::endl;



	////////////////////////////////////////////////////////////////////////
	//                       FIND QUEUE FAMILY                            //
	////////////////////////////////////////////////////////////////////////
	std::vector<VkQueueFamilyProperties> QueueFamilyProps;
	uint32_t queueFamilyCount = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(PhysicalDevice, &queueFamilyCount, QueueFamilyProps.data());
	auto PropIt = std::find_if(QueueFamilyProps.begin(), QueueFamilyProps.end(), [](const VkQueueFamilyProperties& Prop) {
	return Prop.queueFlags & VK_QUEUE_COMPUTE_BIT;
	});
	const uint32_t ComputeQueueFamilyIndex = std::distance(QueueFamilyProps.begin(), PropIt);
	std::cout << "Compute Queue Family Index: " << ComputeQueueFamilyIndex << std::endl;


	///////////////////////////////////////////////////////////////////////////////////////
	// CREATE LOGICAL VERSION OF DEVICE USING QUEUE FAMILY FOUND FROM PHYSICAL DEVICE    //
	///////////////////////////////////////////////////////////////////////////////////////
	float queuePriorities = 1.0f;

	VkDeviceQueueCreateInfo DeviceQueueCreateInfo{};
	DeviceQueueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
	DeviceQueueCreateInfo.queueFamilyIndex = ComputeQueueFamilyIndex;
	DeviceQueueCreateInfo.queueCount = 1;
	DeviceQueueCreateInfo.pQueuePriorities = &queuePriorities;
	    
	VkDeviceCreateInfo DeviceCreateInfo{};
	DeviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;

	DeviceCreateInfo.queueCreateInfoCount = static_cast<uint32_t>(1);
	DeviceCreateInfo.pQueueCreateInfos = &DeviceQueueCreateInfo;

	VkDevice device;
	vkCreateDevice(PhysicalDevice, &DeviceCreateInfo, nullptr, &device);
	////////////////////////////////////////////////////////////////////////
	// 		Allocating Memory, binding and submitting              //
	////////////////////////////////////////////////////////////////////////
	allocateBindAndSubmit(PhysicalDevice, device, ComputeQueueFamilyIndex, inputs, epochs);
	
}


int main(int argc, char *argv[]) {

	std::cout<<"running vulkan process"<<std::endl;

	inputParams inputs;	
	int epochs = 50;
	vulkanLinearRegression(inputs, epochs);
	
	return 0;
}
