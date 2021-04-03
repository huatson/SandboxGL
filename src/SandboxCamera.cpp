#include "SandboxCamera.h"

SandboxCamera::SandboxCamera() 
{
	target = glm::vec3(0.f, 0.f, 0.f);
	CameraPitch = M_PI / 6.f;
	CameraYaw = 0.f;
	CameraOffsetDistance = 300.f;
}

SandboxCamera::~SandboxCamera()
{
	//
}

glm::vec3 SandboxCamera::position()
{
	glm::vec3 CameraPosition = glm::mat3(glm::rotate(CameraYaw, glm::vec3(0, 1, 0))) * glm::vec3(CameraOffsetDistance, 0, 0);
	glm::vec3 CameraAxis = glm::normalize(glm::cross(CameraPosition, glm::vec3(0, 1, 0)));
	return glm::mat3(glm::rotate(CameraPitch, CameraAxis)) * CameraPosition + target;
}

glm::vec3 SandboxCamera::direction()
{
	return glm::normalize(target - position());
}

glm::mat4 SandboxCamera::view_matrix()
{
	return glm::lookAt(position(), target, glm::vec3(0, 1, 0));
}

glm::mat4 SandboxCamera::proj_matrix()
{
	return glm::perspective(45.0f, (float)WINDOW_WIDTH / (float)WINDOW_HEIGHT, 10.0f, 10000.0f);
}

void SandboxCamera::UpdateCamera(int xInput, int yInput, int TargetMinPitch, int TargetMaxPitch, float zIn, float zOut)
{
	if (abs(xInput) + abs(yInput) < 10000)
	{
		xInput = 0;
		yInput = 0;
	};
	CameraPitch = (float)glm::clamp(CameraPitch + (yInput / 32768.0) * 0.03, M_PI / TargetMinPitch, 2 * M_PI / TargetMaxPitch);
	CameraYaw += (xInput / 32768.0) * 0.03;
	CameraOffsetDistance = zOut > 1e-5 ? (float)glm::clamp(CameraOffsetDistance - zOut, 10.0f, 10000.0f) : CameraOffsetDistance;
	CameraOffsetDistance = zIn > 1e-5 ? (float)glm::clamp(CameraOffsetDistance + zIn, 10.0f, 10000.0f) : CameraOffsetDistance;
}
