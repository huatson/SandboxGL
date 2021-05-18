#include "SandboxShaders.h"

SandboxShaders::SandboxShaders() 
	: program_shader(0)
	, vertex_shader(0)
	, fragment_shader(0)
{
	//
}

SandboxShaders::~SandboxShaders()
{
	if (vertex_shader || fragment_shader || program_shader)
	{
		reset_shaders();
	}
}

void SandboxShaders::reset_shaders()
{
	glDeleteShader(vertex_shader);
	vertex_shader = 0;
	glDeleteShader(fragment_shader);
	fragment_shader = 0;
	glDeleteShader(program_shader);
	program_shader = 0;
}

bool SandboxShaders::validate_shaders()
{
	if (vertex_shader && fragment_shader && program_shader)
	{
		return true;
	}
	return  false;
}

void SandboxShaders::internal_load_shader(const char* filename, GLenum type, GLuint *shader)
{
	SDL_RWops* file = SDL_RWFromFile(filename, "r");
	if (!file)
	{
		fprintf(stderr, "Cannot load file %s\n", filename);
		exit(1);
	}
	else
	{
		fprintf(stdout, "Loaded file %s\n", filename);
	}

	long size = (long)SDL_RWseek(file, 0, SEEK_END);
	char* contents = (char*)malloc(size + 1);
	contents[size] = '\0';

	SDL_RWseek(file, 0, SEEK_SET);
	SDL_RWread(file, contents, size, 1);
	SDL_RWclose(file);

	*shader = glCreateShader(type);

	glShaderSource(*shader, 1, (const char**)&contents, NULL);
	glCompileShader(*shader);

	free(contents);

	char log[BUFFER_CHAR];
	int i;
	glGetShaderInfoLog(*shader, BUFFER_CHAR, &i, log);
	log[i] = '\0';
	if (strcmp(log, "") != 0) { printf("%s\n", log); }

	int compile_error = 0;
	glGetShaderiv(*shader, GL_COMPILE_STATUS, &compile_error);
	if (compile_error == GL_FALSE)
	{
		fprintf(stderr, "Compiler Error on Shader %s.\n", filename);
		exit(1);
	}
}

void SandboxShaders::load(const char* vertex_file, const char* fragment_file)
{
	reset_shaders();
	program_shader = glCreateProgram();
	internal_load_shader(vertex_file, GL_VERTEX_SHADER, &vertex_shader);
	internal_load_shader(fragment_file, GL_FRAGMENT_SHADER, &fragment_shader);
	glAttachShader(program_shader, vertex_shader);
	glAttachShader(program_shader, fragment_shader);
	glLinkProgram(program_shader);

	char log[BUFFER_CHAR];
	int i;
	glGetProgramInfoLog(program_shader, BUFFER_CHAR, &i, log);
	log[i] = '\0';
	if (strcmp(log, "") != 0)
	{
		printf("%s\n", log);
	}
}
