#ifndef __SANDBOX_SHADERS__
#define __SANDBOX_SHADERS__

#pragma once

#include <GL/glew.h>

#include <SDL2/SDL.h>
#include <SDL2/SDL_platform.h>
#include <SDL2/SDL_opengl.h>
#include <SDL2/SDL_video.h>
#include <SDL2/SDL_events.h>
#include <SDL2/SDL_joystick.h>

#include <sstream>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <stdarg.h>
#include <time.h>

#define		BUFFER_CHAR				2048


/**
 * shaders
 */
class SandboxShaders
{
public:

	SandboxShaders();
	~SandboxShaders();

	void reset_shaders();
	bool validate_shaders();
	void internal_load_shader(const char* filename, GLenum type, GLuint *shader);
	void load(const char* vertex_file, const char* fragment_file);

	GLuint program_shader;
	GLuint vertex_shader;
	GLuint fragment_shader;
};



#endif
