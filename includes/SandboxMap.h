#ifndef __SANDBOX_MAP__
#define __SANDBOX_MAP__

#pragma once

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif


#include <GL/glew.h>

#include <GLM/glm.hpp>
#include <GLM/gtc/matrix_transform.hpp>
#include <GLM/gtc/type_ptr.hpp>
#include <GLM/gtx/quaternion.hpp>
#include <GLM/gtx/transform.hpp>

#include <sstream>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <stdarg.h>
#include <time.h>
#include <math.h>

// AO COMPUTE SETTINGS								original values
static const float CUSTOM_AO_RADIUS = 50.0f;		//50
static const int CUSTOM_AO_SAMPLES = 256;			//1024
static const int CUSTOM_AO_STEPS = 3;				//5

enum EMapFormat
{
	EF_TXT,
	EF_XYZ,
	EG_RAW
};


/**
 * load height map w/AO data (compute)
 */
class SandboxMap
{
public:

	SandboxMap();
	~SandboxMap();

	float sample(glm::vec2 Position, int w, int h);
	float computeOffsetCPU(std::vector<std::vector<float>> map_data, int w, int h);

	glm::vec3* computeScaledNormalPosCPU(int w, int h);
	glm::vec3* computeNormalsCPU(const glm::vec3* posns, int w, int h);
	float* computeAmbienOcclusionCPU(const glm::vec3* norms, const glm::vec3* posns, int w, int h, FILE *ao_file, bool bGenerate);

	float* computeVBOCPU(const glm::vec3* norms, const glm::vec3* posns, const float *aos, int w, int h);
	void computeTBOCPU(int w, int h, uint32_t *out_tbo_data);

	void load(const char* filename, float multiplier, EMapFormat map_format, int xyz_map_size);

	float horizontal_scale;
	float vertical_scale;
	float height_offset;

	std::vector<std::vector<float>> MapData;
	GLuint vertex_buffer_object;
	GLuint texture_buffer_object;
};


#endif