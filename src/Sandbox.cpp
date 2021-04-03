#include "Sandbox.h"


static const char* FILE_LEVEL_1 = "./levels/level1.txt";	// 500*501
static const char* FILE_LEVEL_2 = "./levels/level2.txt";
static const char* FILE_LEVEL_3 = "./levels/level3.txt";
static const char* FILE_LEVEL_S10 = "./levels/level_S10.txt";

static const char* FILE_LEVEL_XYZ = "./levels/heightmap.xyz";
static const int XYZ_MAP_SIZE = 512;



// fwd declaration
extern "C" bool AddVector(int N);

static bool runAddCudaExample()
{
	/**
	 * CUDA implementation steps:
	 * 1. Create a .cu file: 'SandboxCUDA.cu', and set an entry "C" function: 'Add_Cuda': this will handle all the required kernels from main
	 * 2. At 'SandboxCUDA.cu', create all the desired kernels, by assigning '__global__' macro: 'Add_Kernel' g.e.
	 * 3. From 'Add_Cuda', call the kernel 'Add_Kernel' using triple brackets <<<param1,param2, etc>>>.
	 */

	 /** at main:
	 if (runAddCudaExample())
	 {
		 return EXIT_SUCCESS;
	 }
	 **/

	 // N-Million elements
	int N = 5 << 20;
	std::cout << "CUDA Inter-ops: " << N << "\n";
	return AddVector(N);
}

static void load_level(SandboxMap* sandboxmap)
{
	sandboxmap->load(FILE_LEVEL_XYZ, 1.0, EF_XYZ, XYZ_MAP_SIZE);
}

static void pre_render(SandboxCamera* camera, SDL_Joystick* input_controller)
{
	int xinput = SDL_JoystickGetAxis(input_controller, EGamePadButton::GAMEPAD_STICK_R_HORIZONTAL);
	int yinput = SDL_JoystickGetAxis(input_controller, EGamePadButton::GAMEPAD_STICK_R_VERTICAL);
	float zoom_in = SDL_JoystickGetButton(input_controller, EGamePadButton::GAMEPAD_SHOULDER_L) * 20.0;
	float zoom_out = SDL_JoystickGetButton(input_controller, EGamePadButton::GAMEPAD_SHOULDER_R) * 20.0;


	camera->UpdateCamera(xinput, yinput, 16, 5, zoom_in, zoom_out);

	glm::vec3 trajectory_target_direction_new = glm::normalize(glm::vec3(camera->direction().x, 0.0, camera->direction().z));
	glm::mat3 trajectory_target_rotation = glm::mat3(glm::rotate(atan2f(trajectory_target_direction_new.x, trajectory_target_direction_new.z), glm::vec3(0, 1, 0)));

	camera->target = glm::vec3(0, 0, 0);
}

static void post_render(SandboxCamera* camera)
{
	camera->target = glm::vec3(0, 0, 0);
}

void render(SandboxCamera* camera, 
	FLightDirectional* light, 
	SandboxShaders* groundShader, 
	SandboxShaders* groundShaderShadow,
	SandboxMap* sandboxmap, 
	SDL_Joystick* input_controller)
{
	glm::mat4 light_view = glm::lookAt(camera->target + light->position, camera->target, glm::vec3(0, 1, 0));
	glm::mat4 light_proj = glm::ortho(-2000.0f, 2000.0f, -2000.0f, 2000.0f, 10.0f, 10000.0f);
	glBindFramebuffer(GL_FRAMEBUFFER, light->frame_buffer_object);

	glViewport(0, 0, 2048, 2048);

	glClearDepth(1.0f);
	glClear(GL_DEPTH_BUFFER_BIT);

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);

	glCullFace(GL_FRONT);

	glUseProgram(0);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glUseProgram(groundShaderShadow->program_shader);

	glUniformMatrix4fv(glGetUniformLocation(groundShaderShadow->program_shader, "light_view"), 1, GL_FALSE, glm::value_ptr(light_view));
	glUniformMatrix4fv(glGetUniformLocation(groundShaderShadow->program_shader, "light_proj"), 1, GL_FALSE, glm::value_ptr(light_proj));

	glBindBuffer(GL_ARRAY_BUFFER, sandboxmap->vertex_buffer_object);
	glEnableVertexAttribArray(glGetAttribLocation(groundShaderShadow->program_shader, "vPosition"));
	glVertexAttribPointer(glGetAttribLocation(groundShaderShadow->program_shader, "vPosition"), 3, GL_FLOAT, GL_FALSE, sizeof(float) * 7, (void*)(sizeof(float) * 0));

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sandboxmap->texture_buffer_object);
	glDrawElements(GL_TRIANGLES, (sandboxmap->MapData.size() - 1) * (sandboxmap->MapData[0].size() - 1) * 2 * 3, GL_UNSIGNED_INT, (void*)0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glDisableVertexAttribArray(glGetAttribLocation(groundShaderShadow->program_shader, "vPosition"));
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glUseProgram(0);

	glCullFace(GL_BACK);
	glDisable(GL_CULL_FACE);
	glDisable(GL_DEPTH_TEST);
	glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
	glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);

	glEnable(GL_LINE_SMOOTH);
	glEnable(GL_POLYGON_SMOOTH);
	glEnable(GL_POINT_SMOOTH);
	glEnable(GL_BLEND);

	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glEnable(GL_MULTISAMPLE);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);

	glClearDepth(1.0);
	glClearColor(1.0, 1.0, 1.0, 1.0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glm::vec3 lightDirection = glm::normalize(light->target - light->position);

	glUseProgram(groundShader->program_shader);

	glUniformMatrix4fv(glGetUniformLocation(groundShader->program_shader, "view"), 1, GL_FALSE, glm::value_ptr(camera->view_matrix()));
	glUniformMatrix4fv(glGetUniformLocation(groundShader->program_shader, "proj"), 1, GL_FALSE, glm::value_ptr(camera->proj_matrix()));
	glUniform3f(glGetUniformLocation(groundShader->program_shader, "light_dir"), lightDirection.x, lightDirection.y, lightDirection.z);

	glUniformMatrix4fv(glGetUniformLocation(groundShader->program_shader, "light_view"), 1, GL_FALSE, glm::value_ptr(light_view));
	glUniformMatrix4fv(glGetUniformLocation(groundShader->program_shader, "light_proj"), 1, GL_FALSE, glm::value_ptr(light_proj));

	glActiveTexture(GL_TEXTURE0 + 0);
	glBindTexture(GL_TEXTURE_2D, light->texture);
	glUniform1i(glGetUniformLocation(groundShader->program_shader, "shadows"), 0);

	//
	glBindBuffer(GL_ARRAY_BUFFER, sandboxmap->vertex_buffer_object);

	glEnableVertexAttribArray(glGetAttribLocation(groundShader->program_shader, "vPosition"));
	glEnableVertexAttribArray(glGetAttribLocation(groundShader->program_shader, "vNormal"));
	glEnableVertexAttribArray(glGetAttribLocation(groundShader->program_shader, "vAO"));

	glVertexAttribPointer(glGetAttribLocation(groundShader->program_shader, "vPosition"), 3, GL_FLOAT, GL_FALSE, sizeof(float) * 7, (void*)(sizeof(float) * 0));
	glVertexAttribPointer(glGetAttribLocation(groundShader->program_shader, "vNormal"), 3, GL_FLOAT, GL_FALSE, sizeof(float) * 7, (void*)(sizeof(float) * 3));
	glVertexAttribPointer(glGetAttribLocation(groundShader->program_shader, "vAO"), 1, GL_FLOAT, GL_FALSE, sizeof(float) * 7, (void*)(sizeof(float) * 6));

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, sandboxmap->texture_buffer_object);
	glDrawElements(GL_TRIANGLES, (sandboxmap->MapData.size() - 1) * (sandboxmap->MapData[0].size() - 1) * 2 * 3, GL_UNSIGNED_INT, (void*)0);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	glDisableVertexAttribArray(glGetAttribLocation(groundShader->program_shader, "vPosition"));
	glDisableVertexAttribArray(glGetAttribLocation(groundShader->program_shader, "vNormal"));
	glDisableVertexAttribArray(glGetAttribLocation(groundShader->program_shader, "vAO"));

	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glUseProgram(0);

	// camera
	glMatrixMode(GL_MODELVIEW);
	glLoadMatrixf(glm::value_ptr(camera->view_matrix()));

	glMatrixMode(GL_PROJECTION);
	glLoadMatrixf(glm::value_ptr(camera->proj_matrix()));

	
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);
	glDisable(GL_MULTISAMPLE);
	glDisable(GL_LINE_SMOOTH);
	glDisable(GL_POLYGON_SMOOTH);
	glDisable(GL_POINT_SMOOTH);
	glDisable(GL_BLEND);
}



int main(int argc, char **argv)
{
	//	SDL
	SDL_Init(SDL_INIT_VIDEO | SDL_INIT_JOYSTICK);
	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
	SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 1);
	SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 8);

	SDL_Window *window = SDL_CreateWindow("Sandbox",
		SDL_WINDOWPOS_CENTERED,
		SDL_WINDOWPOS_CENTERED,
		WINDOW_WIDTH,
		WINDOW_HEIGHT,
		SDL_WINDOW_OPENGL);
	if (!window)
	{
		fprintf(stderr, "SDL Error: %s\n", SDL_GetError());
		return EXIT_FAILURE;
	}

	SDL_GLContext context = SDL_GL_CreateContext(window);
	SDL_GL_SetSwapInterval(1);


	//GLEW
	GLenum err = glewInit();
	if (GLEW_OK != err)
	{
		fprintf(stderr, "GLEW Error: %s\n", glewGetErrorString(err));
		return EXIT_FAILURE;
	}


	//	setup
	SandboxShaders* groundShader = new SandboxShaders();
	SandboxShaders* groundShaderShadow = new SandboxShaders();
	groundShader->load("./shaders/ground.vs", "./shaders/ground.fs");
	groundShaderShadow->load("./shaders/ground_shadow.vs", "./shaders/ground_shadow.fs");

	if (!groundShader->validate_shaders() || !groundShaderShadow->validate_shaders())
	{
		fprintf(stderr, "Missing shaders...\n");

		// DESTROY
		delete groundShader;
		delete groundShaderShadow;
		SDL_GL_DeleteContext(context);
		SDL_DestroyWindow(window);
		SDL_Quit();

		return EXIT_FAILURE;
	}
	FLightDirectional* light = new FLightDirectional();
	SandboxCamera* camera = new SandboxCamera();
	SandboxMap* sandboxmap = new SandboxMap();
	load_level(sandboxmap);

	static bool bLoopingGame = true;

	SDL_Joystick* input_controller = nullptr;
	input_controller = SDL_JoystickOpen(0);

	if (!input_controller)
	{
		printf("No controller available\n");
	}

	while (bLoopingGame)
	{
		SDL_Event sandbox_event;
		while (SDL_PollEvent(&sandbox_event))
		{
			if (sandbox_event.type == SDL_QUIT)
			{
				bLoopingGame = false;
				break;
			}

			if (sandbox_event.type == SDL_KEYDOWN)
			{
				switch (sandbox_event.key.keysym.sym)
				{
				case SDLK_ESCAPE:
					bLoopingGame = false;
					break;
				}
			}
		}

		pre_render(camera, input_controller);
		render(camera, light, groundShader, groundShaderShadow, sandboxmap, input_controller);
		post_render(camera);

		glFlush();
		glFinish();
		SDL_GL_SwapWindow(window);
	}

	// DESTROY
	delete camera;
	delete groundShader;
	delete groundShaderShadow;
	delete sandboxmap;
	SDL_GL_DeleteContext(context);
	SDL_DestroyWindow(window);
	SDL_Quit();

	return EXIT_SUCCESS;
}