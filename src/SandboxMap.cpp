#include "SandboxMap.h"

extern "C" bool GetCUDAInfo();
extern "C" float computeOffsetCUDA(std::vector<float> fdata, int map_size);


SandboxMap::SandboxMap() 
	: horizontal_scale(3.937007874f)
	, vertical_scale(3.0f)
	, height_offset(0.0f)
	, vertex_buffer_object(0)
	, texture_buffer_object(0)
{
	//
}

SandboxMap::~SandboxMap()
{
	if (vertex_buffer_object != 0)
	{
		glDeleteBuffers(1, &vertex_buffer_object);
		vertex_buffer_object = 0;
	}

	if (texture_buffer_object != 0)
	{
		glDeleteBuffers(1, &texture_buffer_object);
		texture_buffer_object = 0;
	}
}


float SandboxMap::sample(glm::vec2 Position, int w, int h)
{
	Position.x = (Position.x / horizontal_scale) + w / 2;
	Position.y = (Position.y / horizontal_scale) + h / 2;

	float a0 = fmod(Position.x, 1.0f);
	float a1 = fmod(Position.y, 1.0f);

	int x0 = (int)std::floor(Position.x);
	int x1 = (int)std::ceil(Position.x);
	int y0 = (int)std::floor(Position.y);
	int y1 = (int)std::ceil(Position.y);

	x0 = x0 < 0 ? 0 : x0; x0 = x0 >= w ? w - 1 : x0;
	x1 = x1 < 0 ? 0 : x1; x1 = x1 >= w ? w - 1 : x1;
	y0 = y0 < 0 ? 0 : y0; y0 = y0 >= h ? h - 1 : y0;
	y1 = y1 < 0 ? 0 : y1; y1 = y1 >= h ? h - 1 : y1;

	float s0 = vertical_scale * (MapData[x0][y0] - height_offset);
	float s1 = vertical_scale * (MapData[x1][y0] - height_offset);
	float s2 = vertical_scale * (MapData[x0][y1] - height_offset);
	float s3 = vertical_scale * (MapData[x1][y1] - height_offset);

	return (s0 * (1 - a0) + s1 * a0) * (1 - a1) + (s2 * (1 - a0) + s3 * a0) * a1;
}

float SandboxMap::computeOffsetCPU(std::vector<std::vector<float>> map_data, int w, int h)
{
	float h_offset = 0.f;
	for (int x = 0; x < w; x++)
	{
		for (int y = 0; y < h; y++)
		{
			h_offset += map_data[x][y];
		}	
	}
	printf("CPU offset SUM: %.4f\n", h_offset);
	int size = w * h;
	h_offset /= size;
	printf("CPU offset AVG: %.4f\n", h_offset);
	return h_offset;
}

glm::vec3* SandboxMap::computeScaledNormalPosCPU(int w, int h)
{
	glm::vec3 *posns = (glm::vec3 *)malloc(sizeof(glm::vec3) * w * h);
	for (int x = 0; x < w; x++)
	{
		for (int y = 0; y < h; y++)
		{
			float cx = horizontal_scale * x;
			float cy = horizontal_scale * y;
			float cw = horizontal_scale * w;
			float ch = horizontal_scale * h;
			float posX = cx - cw / 2;
			float posY = sample(glm::vec2(cx - cw / 2, cy - ch / 2), w, h);
			float posZ = cy - ch / 2;
			posns[x + y * w] = glm::vec3(posX, posY, posZ);
		}
	}
	return posns;
}

glm::vec3* SandboxMap::computeNormalsCPU(const glm::vec3* posns, int w, int h)
{
	glm::vec3* norms = (glm::vec3 *)malloc(sizeof(glm::vec3) * w * h);
	for (int x = 0; x < w; x++)
	{
		for (int y = 0; y < h; y++)
		{
			const bool bNormalizeData = (x > 0 && x < w - 1 && y > 0 && y < h - 1);
			norms[x + y * w] = bNormalizeData
				? glm::normalize(glm::mix(glm::cross(posns[(x + 0) + (y + 1)*w] - posns[x + y * w], posns[(x + 1) + (y + 0)*w] - posns[x + y * w]), glm::cross(posns[(x + 0) + (y - 1)*w] - posns[x + y * w], posns[(x - 1) + (y + 0)*w] - posns[x + y * w]), 0.5))
				: glm::vec3(0, 1, 0);
		}
	}
	return norms;
}

float* SandboxMap::computeAmbienOcclusionCPU(const glm::vec3* norms, const glm::vec3* posns, int w, int h, FILE *ao_file, bool bGenerate)
{
	float *aos = (float*)malloc(sizeof(float) * w * h);
	for (int x = 0; x < w; x++)
	{
		for (int y = 0; y < h; y++)
		{
			if (bGenerate)
			{
				float ao_amount = 0.f;
				for (int i = 0; i < CUSTOM_AO_SAMPLES; i++)
				{
					glm::vec3 rnd_Offset = glm::vec3(rand() % 10000 - 5000, rand() % 10000 - 5000, rand() % 10000 - 5000);
					glm::vec3 Offset = glm::normalize(rnd_Offset);
					if (glm::dot(Offset, norms[x + y * w]) < 0.0f)
					{
						Offset = -Offset;
					}
					for (int j = 1; j <= CUSTOM_AO_STEPS; j++)
					{
						glm::vec3 next = posns[x + y * w] + (((float)j) / CUSTOM_AO_STEPS) * CUSTOM_AO_RADIUS * Offset;
						float currentSample = sample(glm::vec2(next.x, next.z), w, h);
						if (currentSample > next.y)
						{
							ao_amount += 1.0;
							break;
						}
					}
				}
				aos[x + y * w] = 1.0f - (ao_amount / CUSTOM_AO_SAMPLES);
				fprintf(ao_file, y == h - 1 ? "%f\n" : "%f ", aos[x + y * w]);
			}
			else
			{
				fscanf(ao_file, y == h - 1 ? "%f\n" : "%f ", &aos[x + y * w]);
			}
		}
		printf("Generating %d of %d\r", x, w);
	}
	return aos;
}

float* SandboxMap::computeVBOCPU(const glm::vec3* norms, const glm::vec3* posns, const float *aos, int w, int h)
{
	float *vbo_data = (float *)malloc(sizeof(float) * 7 * w * h);
	for (int x = 0; x < w; x++)
	{
		for (int y = 0; y < h; y++)
		{
			vbo_data[x * 7 + y * 7 * w + 0] = posns[x + y * w].x;
			vbo_data[x * 7 + y * 7 * w + 1] = posns[x + y * w].y;
			vbo_data[x * 7 + y * 7 * w + 2] = posns[x + y * w].z;
			vbo_data[x * 7 + y * 7 * w + 3] = norms[x + y * w].x;
			vbo_data[x * 7 + y * 7 * w + 4] = norms[x + y * w].y;
			vbo_data[x * 7 + y * 7 * w + 5] = norms[x + y * w].z;
			vbo_data[x * 7 + y * 7 * w + 6] = aos[x + y * w];
		}
	}
	return vbo_data;
}

void SandboxMap::computeTBOCPU(int w, int h, uint32_t *out_tbo_data)
{
	for (int x = 0; x < (w - 1); x++)
	{
		for (int y = 0; y < (h - 1); y++)
		{
			out_tbo_data[x * 3 * 2 + y * 3 * 2 * (w - 1) + 0] = (x + 0) + (y + 0) * w;
			out_tbo_data[x * 3 * 2 + y * 3 * 2 * (w - 1) + 1] = (x + 0) + (y + 1) * w;
			out_tbo_data[x * 3 * 2 + y * 3 * 2 * (w - 1) + 2] = (x + 1) + (y + 0) * w;
			out_tbo_data[x * 3 * 2 + y * 3 * 2 * (w - 1) + 3] = (x + 1) + (y + 1) * w;
			out_tbo_data[x * 3 * 2 + y * 3 * 2 * (w - 1) + 4] = (x + 1) + (y + 0) * w;
			out_tbo_data[x * 3 * 2 + y * 3 * 2 * (w - 1) + 5] = (x + 0) + (y + 1) * w;
		}
	}
}

void SandboxMap::load(const char* filename, float multiplier, EMapFormat map_format, int xyz_map_size)
{
	// LEVEL3 --> 523 rows x 532
	const bool bValidCuda = GetCUDAInfo();
	vertical_scale = multiplier * vertical_scale;
	if (vertex_buffer_object != 0)
	{
		glDeleteBuffers(1, &vertex_buffer_object);
		vertex_buffer_object = 0;
	}
	if (texture_buffer_object != 0)
	{
		glDeleteBuffers(1, &texture_buffer_object);
		texture_buffer_object = 0;
	}

	glGenBuffers(1, &vertex_buffer_object);
	glGenBuffers(1, &texture_buffer_object);
	MapData.clear();

	printf("loading map: '%s' ...\n", filename);
	std::vector<float> flatten_data;
	if (map_format == EMapFormat::EF_XYZ)
	{
		std::ifstream XYZfile(filename);
		char xyz2txt_filename[512];
		memcpy(xyz2txt_filename, filename, strlen(filename) - 4);
		xyz2txt_filename[strlen(filename) - 4] = '\0';
		strcat(xyz2txt_filename, "_new.txt");
		srand(0);
		FILE* txt_file = fopen(xyz2txt_filename, "r");
		if (txt_file)
		{
			printf("Transposed XYZ map file already created (%s), skip conversion.\n", xyz2txt_filename);
		}
		else
		{
			printf("Transpose XYZ map to ROW/COL data: %s.\n", xyz2txt_filename);
			txt_file = fopen(xyz2txt_filename, "w");
			
			for (int ypos = 0; ypos < xyz_map_size; ypos++)
			{
				for (int xpos = 0; xpos < xyz_map_size; xpos++)
				{
					std::string XYZline;
					if (std::getline(XYZfile, XYZline))
					{
						std::istringstream iss(XYZline);
						float x, y, z;
						if (iss >> std::skipws >> x >> y >> z || !iss.eof())
						{
							fprintf(txt_file, xpos == xyz_map_size - 1 ? "%f\n" : "%f ", z);
						}
					}
				}
			}
			fclose(txt_file);
		}
		// re-open new TXT file
		filename = xyz2txt_filename;
	}

	/**
	std::ifstream TXTfile(filename);
	std::string TXTline;
	printf("Processing: '%s' ...\n", filename);
	while (std::getline(TXTfile, TXTline))
	{
		std::vector<float> row;
		std::istringstream input_stream_string(TXTline);
		while (input_stream_string)
		{
			float depth_data = 0.f;
			while (input_stream_string >> depth_data || !input_stream_string.eof())
			{
				flatten_data.push_back(depth_data);
				row.push_back(depth_data);
			}	
			input_stream_string >> depth_data;
			flatten_data.push_back(depth_data);
			row.push_back(depth_data);
		}
		MapData.push_back(row);
	}
	**/

	std::ifstream file(filename);
	std::string row_line;
	while (std::getline(file, row_line))
	{
		std::vector<float> row;
		std::istringstream iss(row_line);
		while (iss)
		{
			float f;
			iss >> f;
			flatten_data.push_back(f);
			row.push_back(f); // row size 533
		}
		MapData.push_back(row);
	}

	printf("Finished\n");
	printf("Computing Samples-Offsets...\n");
	int w = (int)MapData.size();
	int h = (int)MapData[0].size(); // 533
	int map_size = w * h;

	printf("Map Size: (cols) %d (rows) %d, flatten_data: %d\n", w, h, (int)flatten_data.size());

    float tmp = computeOffsetCPU(MapData, w, h);

	height_offset = 0.f;
	height_offset = computeOffsetCUDA(flatten_data, map_size);

	printf("Computing Normals\n");
	glm::vec3 *posns = computeScaledNormalPosCPU(w, h);
	glm::vec3 *norms = computeNormalsCPU(posns, w, h);


	char ao_filename[512];
	memcpy(ao_filename, filename, strlen(filename) - 4);
	ao_filename[strlen(filename) - 4] = '\0';
	strcat(ao_filename, "_ao.txt");
	srand(0);
	FILE *ao_file = fopen(ao_filename, "r");
	bool ao_generate = false;
	if (ao_file == NULL || ao_generate)
	{
		ao_file = fopen(ao_filename, "w");
		ao_generate = true;
		printf("Computing AO\n");
	}
	else
	{
		printf("Loading AO\n");
	}

	float *aos = computeAmbienOcclusionCPU(norms, posns, w, h, ao_file, ao_generate);
	fclose(ao_file);

	float *vbo_data = computeVBOCPU(norms, posns, aos, w, h);

	free(posns);
	free(norms);
	free(aos);

	uint32_t *tbo_data = (uint32_t *)malloc(sizeof(uint32_t) * 3 * 2 * (w - 1) * (h - 1));
	computeTBOCPU(w, h, tbo_data);

	glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_object);
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 7 * w * h, vbo_data, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, texture_buffer_object);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(uint32_t) * 3 * 2 * (w - 1) * (h - 1), tbo_data, GL_STATIC_DRAW);

	free(vbo_data);
	free(tbo_data);
}


