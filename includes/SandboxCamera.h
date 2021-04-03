#ifndef __SANDBOX_CAMERA__
#define __SANDBOX_CAMERA__

#pragma once

#include <GLM/glm.hpp>
#include <GLM/gtc/matrix_transform.hpp>
#include <GLM/gtc/type_ptr.hpp>
#include <GLM/gtx/quaternion.hpp>
#include <GLM/gtx/transform.hpp>


#include <SDL2/SDL.h>
#include <SDL2/SDL_platform.h>
#include <SDL2/SDL_opengl.h>
#include <SDL2/SDL_video.h>
#include <SDL2/SDL_events.h>
#include <SDL2/SDL_joystick.h>



#define		WINDOW_WIDTH			1280
#define		WINDOW_HEIGHT			720

/**
 * simple camera
 */
class SandboxCamera
{
public:

	SandboxCamera();
	~SandboxCamera();

	glm::vec3 position();
	glm::vec3 direction();
	glm::mat4 view_matrix();
	glm::mat4 proj_matrix();

	void UpdateCamera(int xInput, int yInput, int TargetMinPitch, int TargetMaxPitch, float zIn, float zOut);


	glm::vec3 target;
	float CameraPitch;
	float CameraYaw;
	float CameraOffsetDistance;
};
#endif
