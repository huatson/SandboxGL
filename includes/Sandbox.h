#ifndef __SANDBOX_H__
#define __SANDBOX_H__

#pragma once

#include "SandboxMap.h"
#include "SandboxCamera.h"
#include "SandboxShaders.h"

#include <Eigen/Dense>


// CUDA STUFF
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


// debug performance
#define		TRAJECTORY_LEN			120
#define		LIGHT_BUFFER_SIZE		2048
#define		DEBUG_INPUT				1
#define		DEBUG_SCALE				3	//low scale 2



/**
 * from SDL Joystick input States
 */
enum EGamePadButton
{
	GAMEPAD_BACK = 5,
	GAMEPAD_START = 4,
	GAMEPAD_A = 10,
	GAMEPAD_B = 11,
	GAMEPAD_X = 12,
	GAMEPAD_Y = 13,
	GAMEPAD_TRIGGER_L = 4,
	GAMEPAD_TRIGGER_R = 5,
	GAMEPAD_SHOULDER_L = 8,
	GAMEPAD_SHOULDER_R = 9,
	GAMEPAD_STICK_L_HORIZONTAL = 0,
	GAMEPAD_STICK_L_VERTICAL = 1,
	GAMEPAD_STICK_R_HORIZONTAL = 2,
	GAMEPAD_STICK_R_VERTICAL = 3
};


struct FTrajectory 
{
	float width;

	glm::vec3 positions[TRAJECTORY_LEN];
	glm::vec3 directions[TRAJECTORY_LEN];
	glm::mat3 rotations[TRAJECTORY_LEN];
	float heights[TRAJECTORY_LEN];

	float gait_stand[TRAJECTORY_LEN];
	float gait_walk[TRAJECTORY_LEN];
	float gait_jog[TRAJECTORY_LEN];
	float gait_crouch[TRAJECTORY_LEN];
	float gait_jump[TRAJECTORY_LEN];
	float gait_bump[TRAJECTORY_LEN];

	glm::vec3 target_dir;
	glm::vec3 target_vel;

	FTrajectory()
		: width(25.0f)
		, target_dir(glm::vec3(0.f, 0.f, 1.0f))
		, target_vel(glm::vec3(0.f, 0.f, 0.f))
	{
		//
	}

};


struct FLightDirectional 
{
	glm::vec3 target;
	glm::vec3 position;

	GLuint frame_buffer_object;
	GLuint buffer;
	GLuint texture;

	FLightDirectional() 
		: target(glm::vec3(0.f))
		, position(glm::vec3(3000.f, 3700.f, 1500.f))
		, frame_buffer_object(0)
		, buffer(0)
		, texture(0) 
	{

		glGenFramebuffers(1, &frame_buffer_object);
		glBindFramebuffer(GL_FRAMEBUFFER, frame_buffer_object);
		glDrawBuffer(GL_NONE);
		glReadBuffer(GL_NONE);

		glGenRenderbuffers(1, &buffer);
		glBindRenderbuffer(GL_RENDERBUFFER, buffer);
		glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, LIGHT_BUFFER_SIZE, LIGHT_BUFFER_SIZE);
		glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, buffer);

		glGenTextures(1, &texture);
		glBindTexture(GL_TEXTURE_2D, texture);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, LIGHT_BUFFER_SIZE, LIGHT_BUFFER_SIZE, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);


		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
		glTexParameteri(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, GL_INTENSITY);
		glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, texture, 0);

		glBindTexture(GL_TEXTURE_2D, 0);
		glBindRenderbuffer(GL_RENDERBUFFER, 0);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);

	}

	~FLightDirectional() 
	{
		glDeleteBuffers(1, &frame_buffer_object);
		glDeleteBuffers(1, &buffer);
		glDeleteTextures(1, &texture);
	}
};

#endif