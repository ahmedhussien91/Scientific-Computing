
#include "stdafx.h"
#include <GL/gl.h>
#include <GL/glu.h>
#include <assert.h>
#include <math.h>
#include <iostream>
#include <fstream>

#include "Clothy.h"
#include "PhysEnv.h"
#include "SimProps.h"
#include "SetVert.h"
#include "AddSpher.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#undef THIS_FILE
static char THIS_FILE[] = __FILE__;
#endif

#pragma warning (disable:4244)      // I NEED TO CONVERT FROM DOUBLE TO FLOAT

/////////////////////////////////////////////////////////////////////////////
// CPhysEnv

// INITIALIZE THE SIMULATION WORLD
CPhysEnv::CPhysEnv()
{
	m_IntegratorType = EULER_INTEGRATOR;

	m_Pick[0] = -1;
	m_Pick[1] = -1;
	m_ParticleSys[0] = NULL;
	m_ParticleSys[1] = NULL;
	m_ParticleSys[2] = NULL;	// RESET BUFFER
	// THESE TEMP PARTICLE BUFFERS ARE NEEDED FOR THE MIDPOINT AND RK4 INTEGRATOR
	for (int i = 0; i < 5; i++)
		m_TempSys[i] = NULL;
	m_ParticleCnt = 0;
	m_Contact = NULL;
	m_Spring = NULL;
	m_SpringCnt = 0;		
	m_MouseForceActive = FALSE;

	m_UseGravity = TRUE;
	m_DrawSprings = TRUE;
	m_DrawStructural = TRUE;	// By default only draw structural springs
	m_DrawBend = FALSE;
	m_DrawShear = FALSE;
	m_DrawVertices	= TRUE;
	m_CollisionActive = TRUE;
	m_CollisionRootFinding = FALSE;		// I AM NOT LOOKING FOR A COLLISION RIGHT AWAY

	MAKEVECTOR(m_Gravity, 0.0f, -0.2f, 0.0f)
	m_UserForceMag = 100.0;
	m_UserForceActive = FALSE;
	m_MouseForceKs = 2.0f;	// MOUSE SPRING CONSTANT
	m_Kd	= 0.04f;	// DAMPING FACTOR
	m_Kr	= 0.1f;		// 1.0 = SUPERBALL BOUNCE 0.0 = DEAD WEIGHT
	m_Ksh	= 5.0f;		// HOOK'S SPRING CONSTANT
	m_Ksd	= 0.1f;		// SPRING DAMPING CONSTANT

	// CREATE THE SIZE FOR THE SIMULATION WORLD
	m_WorldSizeX = 15.0f;
	m_WorldSizeY = 15.0f;
	m_WorldSizeZ = 15.0f;

	m_CollisionPlane = (tCollisionPlane	*)malloc(sizeof(tCollisionPlane) * 6);
	m_CollisionPlaneCnt = 6;

	// MAKE THE TOP PLANE (CEILING)
	MAKEVECTOR(m_CollisionPlane[0].normal,0.0f, -1.0f, 0.0f)
	m_CollisionPlane[0].d = m_WorldSizeY / 2.0f;

	// MAKE THE BOTTOM PLANE (FLOOR)
	MAKEVECTOR(m_CollisionPlane[1].normal,0.0f, 1.0f, 0.0f)
	m_CollisionPlane[1].d = m_WorldSizeY / 2.0f;

	// MAKE THE LEFT PLANE
	MAKEVECTOR(m_CollisionPlane[2].normal,-1.0f, 0.0f, 0.0f)
	m_CollisionPlane[2].d = m_WorldSizeX / 2.0f;

	// MAKE THE RIGHT PLANE
	MAKEVECTOR(m_CollisionPlane[3].normal,1.0f, 0.0f, 0.0f)
	m_CollisionPlane[3].d = m_WorldSizeX / 2.0f;

	// MAKE THE FRONT PLANE
	MAKEVECTOR(m_CollisionPlane[4].normal,0.0f, 0.0f, -1.0f)
	m_CollisionPlane[4].d = m_WorldSizeZ / 2.0f;

	// MAKE THE BACK PLANE
	MAKEVECTOR(m_CollisionPlane[5].normal,0.0f, 0.0f, 1.0f)
	m_CollisionPlane[5].d = m_WorldSizeZ / 2.0f;

	m_SphereCnt = 0;

}

CPhysEnv::~CPhysEnv()
{
	if (m_ParticleSys[0])
		free(m_ParticleSys[0]);
	if (m_ParticleSys[1])
		free(m_ParticleSys[1]);
	if (m_ParticleSys[2])
		free(m_ParticleSys[2]);
	for (int i = 0; i < 5; i++)
	{
		if (m_TempSys[i])
			free(m_TempSys[i]);
	}
	if (m_Contact)
		free(m_Contact);
	free(m_CollisionPlane);
	free(m_Spring);

	free(m_Sphere);
}

void CPhysEnv::RenderWorld()
{
	tParticle	*tempParticle;
	tSpring		*tempSpring;

	// FIRST DRAW THE WORLD CONTAINER  
	glColor3f(1.0f,1.0f,1.0f);
    // do a big linestrip to get most of edges
    glBegin(GL_LINE_STRIP);
        glVertex3f(-m_WorldSizeX/2.0f, m_WorldSizeY/2.0f,-m_WorldSizeZ/2.0f);
        glVertex3f( m_WorldSizeX/2.0f, m_WorldSizeY/2.0f,-m_WorldSizeZ/2.0f);
        glVertex3f( m_WorldSizeX/2.0f, m_WorldSizeY/2.0f, m_WorldSizeZ/2.0f);
        glVertex3f(-m_WorldSizeX/2.0f, m_WorldSizeY/2.0f, m_WorldSizeZ/2.0f);
        glVertex3f(-m_WorldSizeX/2.0f, m_WorldSizeY/2.0f,-m_WorldSizeZ/2.0f);
        glVertex3f(-m_WorldSizeX/2.0f,-m_WorldSizeY/2.0f,-m_WorldSizeZ/2.0f);
    glEnd();
    // fill in the stragglers
    glBegin(GL_LINES);
        glVertex3f( m_WorldSizeX/2.0f, m_WorldSizeY/2.0f,-m_WorldSizeZ/2.0f);
        glVertex3f( m_WorldSizeX/2.0f,-m_WorldSizeY/2.0f,-m_WorldSizeZ/2.0f);

        glVertex3f( m_WorldSizeX/2.0f, m_WorldSizeY/2.0f, m_WorldSizeZ/2.0f);
        glVertex3f( m_WorldSizeX/2.0f,-m_WorldSizeY/2.0f, m_WorldSizeZ/2.0f);

        glVertex3f(-m_WorldSizeX/2.0f, m_WorldSizeY/2.0f, m_WorldSizeZ/2.0f);
        glVertex3f(-m_WorldSizeX/2.0f,-m_WorldSizeY/2.0f, m_WorldSizeZ/2.0f);
    glEnd();
    
    // draw floor
    glDisable(GL_CULL_FACE);
    glBegin(GL_QUADS);
        glColor3f(0.0f,0.0f,0.5f);
        glVertex3f(-m_WorldSizeX/2.0f,-m_WorldSizeY/2.0f,-m_WorldSizeZ/2.0f);
        glVertex3f( m_WorldSizeX/2.0f,-m_WorldSizeY/2.0f,-m_WorldSizeZ/2.0f);
        glVertex3f( m_WorldSizeX/2.0f,-m_WorldSizeY/2.0f, m_WorldSizeZ/2.0f);
        glVertex3f(-m_WorldSizeX/2.0f,-m_WorldSizeY/2.0f, m_WorldSizeZ/2.0f);
    glEnd();
    glEnable(GL_CULL_FACE);


	if (m_ParticleSys)
	{
		if (m_Spring && m_DrawSprings)
		{
			glBegin(GL_LINES);
			glColor3f(0.0f,0.8f,0.8f);
			tempSpring = m_Spring;
			for (int loop = 0; loop < m_SpringCnt; loop++)
			{
				// Only draw normal springs or the cloth "structural" ones
				if ((tempSpring->type == MANUAL_SPRING) ||
					(tempSpring->type == STRUCTURAL_SPRING && m_DrawStructural) ||
					(tempSpring->type == SHEAR_SPRING && m_DrawShear) ||
					(tempSpring->type == BEND_SPRING && m_DrawBend))
				{
					glVertex3fv((float *)&m_CurrentSys[tempSpring->p1].pos);
					glVertex3fv((float *)&m_CurrentSys[tempSpring->p2].pos);
				}
				tempSpring++;
			}
			if (m_MouseForceActive)	// DRAW MOUSESPRING FORCE
			{
				if (m_Pick[0] > -1)
				{
					glColor3f(0.8f,0.0f,0.8f);
					glVertex3fv((float *)&m_CurrentSys[m_Pick[0]].pos);
					glVertex3fv((float *)&m_MouseDragPos[0]);
				}
				if (m_Pick[1] > -1)
				{
					glColor3f(0.8f,0.0f,0.8f);
					glVertex3fv((float *)&m_CurrentSys[m_Pick[1]].pos);
					glVertex3fv((float *)&m_MouseDragPos[1]);
				}
			}
			glEnd();
		}
		if (m_DrawVertices)
		{
			glBegin(GL_POINTS);
			tempParticle = m_CurrentSys;
			for (int loop = 0; loop < m_ParticleCnt; loop++)
			{
				if (loop == m_Pick[0])
					glColor3f(0.0f,0.8f,0.0f);
				else if (loop == m_Pick[1])
					glColor3f(0.8f,0.0f,0.0f);
				else
					glColor3f(0.8f,0.8f,0.0f);
				glVertex3fv((float *)&tempParticle->pos);
				tempParticle++;
			}
			glEnd();
		}
	}

	if (m_SphereCnt > 0 && m_CollisionActive)
	{
		glColor3f(0.5f,0.0f,0.0f);
		for (int loop = 0; loop < m_SphereCnt; loop++)
		{
			glPushMatrix();
			glTranslatef(m_Sphere[loop].pos.x, m_Sphere[loop].pos.y, m_Sphere[loop].pos.z);
			glScalef(m_Sphere[loop].radius,m_Sphere[loop].radius,m_Sphere[loop].radius);
			glCallList(OGL_AXIS_DLIST);
			glPopMatrix();
		}
	}
}

void CPhysEnv::GetNearestPoint(int x, int y)
{
/// Local Variables ///////////////////////////////////////////////////////////
	float *feedBuffer;
	int hitCount;
	tParticle *tempParticle;
	int loop;
///////////////////////////////////////////////////////////////////////////////
	// INITIALIZE A PLACE TO PUT ALL THE FEEDBACK INFO (3 DATA, 1 TAG, 2 TOKENS)
	feedBuffer = (float *)malloc(sizeof(GLfloat) * m_ParticleCnt * 6);
	// TELL OPENGL ABOUT THE BUFFER
	glFeedbackBuffer(m_ParticleCnt * 6,GL_3D,feedBuffer);
	(void)glRenderMode(GL_FEEDBACK);	// SET IT IN FEEDBACK MODE

	tempParticle = m_CurrentSys;
	for (loop = 0; loop < m_ParticleCnt; loop++)
	{
		// PASS THROUGH A MARKET LETTING ME KNOW WHAT VERTEX IT WAS
		glPassThrough((float)loop);
		// SEND THE VERTEX
		glBegin(GL_POINTS);
		glVertex3fv((float *)&tempParticle->pos);
		glEnd();
		tempParticle++;
	}
	hitCount = glRenderMode(GL_RENDER); // HOW MANY HITS DID I GET
	//fout<<"hit count "<<hitCount<,endl;

	CompareBuffer(hitCount,feedBuffer,(float)x,(float)y);		// CHECK THE HIT 
	free(feedBuffer);		// GET RID OF THE MEMORY
}

///////////////////////////////////////////////////////////////////////////////
// Function:	CompareBuffer
// Purpose:		Check the feedback buffer to see if anything is hit
// Arguments:	Number of hits, pointer to buffer, point to test
///////////////////////////////////////////////////////////////////////////////
void CPhysEnv::CompareBuffer(int size, float *buffer,float x, float y)
{
/// Local Variables ///////////////////////////////////////////////////////////
	GLint count;
	GLfloat token,point[3];
	int loop,currentVertex,result = -1;
	long nearest = -1, dist;
///////////////////////////////////////////////////////////////////////////////
	count = size;
	while (count)
	{
		token = buffer[size - count];	// CHECK THE TOKEN
		count--;
		if (token == GL_PASS_THROUGH_TOKEN)	// VERTEX MARKER
		{
			currentVertex = (int)buffer[size - count]; // WHAT VERTEX
			count--;
		}
		else if (token == GL_POINT_TOKEN)
		{
			// THERE ARE THREE ELEMENTS TO A POINT TOKEN
			for (loop = 0; loop < 3; loop++)
			{
				point[loop] = buffer[size - count];
				count--;
			}
			dist = ((x - point[0]) * (x - point[0])) + ((y - point[1]) * (y - point[1]));
			if (result == -1 || dist < nearest)
			{
				nearest = dist;
				result = currentVertex;
			}
		}
	}

	if (nearest < 50.0f)
	{
		if (m_Pick[0] == -1)
			m_Pick[0] = result;
		else if (m_Pick[1] == -1)
			m_Pick[1] = result;
		else
		{
			m_Pick[0] = result;
			m_Pick[1] = -1;
		}
	}
}
////// CompareBuffer //////////////////////////////////////////////////////////

void CPhysEnv::SetWorldParticles(tTexturedVertex *coords,int particleCnt)
{
	tParticle *tempParticle;

	if (m_ParticleSys[0])
		free(m_ParticleSys[0]);
	if (m_ParticleSys[1])
		free(m_ParticleSys[1]);
	if (m_ParticleSys[2])
		free(m_ParticleSys[2]);
	for (int i = 0; i < 5; i++)
	{
		if (m_TempSys[i])
			free(m_TempSys[i]);
	}
	if (m_Contact)
		free(m_Contact);
	// THE SYSTEM IS DOUBLE BUFFERED TO MAKE THINGS EASIER
	m_CurrentSys = (tParticle *)malloc(sizeof(tParticle) * particleCnt);
	m_TargetSys = (tParticle *)malloc(sizeof(tParticle) * particleCnt);
	m_ParticleSys[2] = (tParticle *)malloc(sizeof(tParticle) * particleCnt);
	for (int i = 0; i < 5; i++)
	{
		m_TempSys[i] = (tParticle *)malloc(sizeof(tParticle) * particleCnt);
	}
	m_ParticleCnt = particleCnt;

	// MULTIPLIED PARTICLE COUNT * 2 SINCE THEY CAN COLLIDE WITH MULTIPLE WALLS
	m_Contact = (tContact *)malloc(sizeof(tContact) * particleCnt * 2);
	m_ContactCnt = 0;

	tempParticle = m_CurrentSys;
	for (int loop = 0; loop < particleCnt; loop++)
	{
		MAKEVECTOR(tempParticle->pos, coords->x, coords->y, coords->z)
		MAKEVECTOR(tempParticle->v, 0.0f, 0.0f, 0.0f)
		MAKEVECTOR(tempParticle->f, 0.0f, 0.0f, 0.0f)
		tempParticle->oneOverM = 1.0f;							// MASS OF 1
		tempParticle++;
		coords++;
	}

	// COPY THE SYSTEM TO THE SECOND ONE ALSO
	memcpy(m_TargetSys,m_CurrentSys,sizeof(tParticle) * particleCnt);
	// COPY THE SYSTEM TO THE RESET BUFFER ALSO
	memcpy(m_ParticleSys[2],m_CurrentSys,sizeof(tParticle) * particleCnt);

	m_ParticleSys[0] = m_CurrentSys;
	m_ParticleSys[1] = m_TargetSys;
}

///////////////////////////////////////////////////////////////////////////////
// Function:	FreeSystem
// Purpose:		Remove all particles and clear it out
///////////////////////////////////////////////////////////////////////////////
void CPhysEnv::FreeSystem()
{
	m_Pick[0] = -1;
	m_Pick[1] = -1;
	if (m_ParticleSys[0])
	{
		m_ParticleSys[0] = NULL;
		free(m_ParticleSys[0]);
	}
	if (m_ParticleSys[1])
	{
		free(m_ParticleSys[1]);
		m_ParticleSys[1] = NULL;
	}
	if (m_ParticleSys[2])
	{
		free(m_ParticleSys[2]);
		m_ParticleSys[2] = NULL;	// RESET BUFFER
	}
	for (int i = 0; i < 5; i++)
	{
		if (m_TempSys[i])
		{
			free(m_TempSys[i]);
			m_TempSys[i] = NULL;	// RESET BUFFER
		}
	}
	if (m_Contact)
	{
		free(m_Contact);
		m_Contact = NULL;
	}
	if (m_Spring)
	{
		free(m_Spring);
		m_Spring = NULL;
	}
	m_SpringCnt = 0;	
	m_ParticleCnt = 0;
}
////// FreeSystem //////////////////////////////////////////////////////////////

void CPhysEnv::LoadData(FILE *fp)
{
	fread(&m_UseGravity,sizeof(BOOL),1,fp);
	fread(&m_UseDamping,sizeof(BOOL),1,fp);
	fread(&m_UserForceActive,sizeof(BOOL),1,fp);
	fread(&m_Gravity,sizeof(tVector),1,fp);
	fread(&m_UserForce,sizeof(tVector),1,fp);
	fread(&m_UserForceMag,sizeof(float),1,fp);
	fread(&m_Kd,sizeof(float),1,fp);
	fread(&m_Kr,sizeof(float),1,fp);
	fread(&m_Ksh,sizeof(float),1,fp);
	fread(&m_Ksd,sizeof(float),1,fp);
	fread(&m_ParticleCnt,sizeof(int),1,fp);
	m_CurrentSys = (tParticle *)malloc(sizeof(tParticle) * m_ParticleCnt);
	m_TargetSys = (tParticle *)malloc(sizeof(tParticle) * m_ParticleCnt);
	m_ParticleSys[2] = (tParticle *)malloc(sizeof(tParticle) * m_ParticleCnt);
	for (int i = 0; i < 5; i++)
	{
		m_TempSys[i] = (tParticle *)malloc(sizeof(tParticle) * m_ParticleCnt);
	}
	m_ParticleSys[0] = m_CurrentSys;
	m_ParticleSys[1] = m_TargetSys;
	m_Contact = (tContact *)malloc(sizeof(tContact) * m_ParticleCnt);
	fread(m_ParticleSys[0],sizeof(tParticle),m_ParticleCnt,fp);
	fread(m_ParticleSys[1],sizeof(tParticle),m_ParticleCnt,fp);
	fread(m_ParticleSys[2],sizeof(tParticle),m_ParticleCnt,fp);
	fread(&m_SpringCnt,sizeof(int),1,fp);
	m_Spring = (tSpring *)malloc(sizeof(tSpring) * (m_SpringCnt));
	fread(m_Spring,sizeof(tSpring),m_SpringCnt,fp);
	fread(m_Pick,sizeof(int),2,fp);
	fread(&m_SphereCnt,sizeof(int),1,fp);
	m_Sphere = (tCollisionSphere *)malloc(sizeof(tCollisionSphere) * (m_SphereCnt));
	fread(m_Sphere,sizeof(tCollisionSphere),m_SphereCnt,fp);
}

void CPhysEnv::SaveData(FILE *fp)
{
	fwrite(&m_UseGravity,sizeof(BOOL),1,fp);
	fwrite(&m_UseDamping,sizeof(BOOL),1,fp);
	fwrite(&m_UserForceActive,sizeof(BOOL),1,fp);
	fwrite(&m_Gravity,sizeof(tVector),1,fp);
	fwrite(&m_UserForce,sizeof(tVector),1,fp);
	fwrite(&m_UserForceMag,sizeof(float),1,fp);
	fwrite(&m_Kd,sizeof(float),1,fp);
	fwrite(&m_Kr,sizeof(float),1,fp);
	fwrite(&m_Ksh,sizeof(float),1,fp);
	fwrite(&m_Ksd,sizeof(float),1,fp);
	fwrite(&m_ParticleCnt,sizeof(int),1,fp);
	fwrite(m_ParticleSys[0],sizeof(tParticle),m_ParticleCnt,fp);
	fwrite(m_ParticleSys[1],sizeof(tParticle),m_ParticleCnt,fp);
	fwrite(m_ParticleSys[2],sizeof(tParticle),m_ParticleCnt,fp);
	fwrite(&m_SpringCnt,sizeof(int),1,fp);
	fwrite(m_Spring,sizeof(tSpring),m_SpringCnt,fp);
	fwrite(m_Pick,sizeof(int),2,fp);
	fwrite(&m_SphereCnt,sizeof(int),1,fp);
	fwrite(m_Sphere,sizeof(tCollisionSphere),m_SphereCnt,fp);
}

// RESET THE SIM TO INITIAL VALUES
void CPhysEnv::ResetWorld()
{
	memcpy(m_CurrentSys,m_ParticleSys[2],sizeof(tParticle) * m_ParticleCnt);
	memcpy(m_TargetSys,m_ParticleSys[2],sizeof(tParticle) * m_ParticleCnt);
}

void CPhysEnv::SetWorldProperties()
{
	CSimProps	dialog;
	dialog.m_CoefRest = m_Kr;
	dialog.m_Damping = m_Kd;
	dialog.m_GravX = m_Gravity.x;
	dialog.m_GravY = m_Gravity.y;
	dialog.m_GravZ = m_Gravity.z;
	dialog.m_SpringConst = m_Ksh;
	dialog.m_SpringDamp = m_Ksd;
	dialog.m_UserForceMag = m_UserForceMag;
	if (dialog.DoModal() == IDOK)
	{
		m_Kr = dialog.m_CoefRest;
		m_Kd = dialog.m_Damping;
		m_Gravity.x = dialog.m_GravX;
		m_Gravity.y = dialog.m_GravY;
		m_Gravity.z = dialog.m_GravZ;
		m_UserForceMag = dialog.m_UserForceMag;
		m_Ksh = dialog.m_SpringConst;
		m_Ksd = dialog.m_SpringDamp;
		for (int loop = 0; loop < m_SpringCnt; loop++)
		{
			m_Spring[loop].Ks = m_Ksh;
			m_Spring[loop].Kd = m_Ksd;
		}
	}
}

void CPhysEnv::SetVertexProperties()
{
	CSetVert	dialog;
	dialog.m_VertexMass = m_CurrentSys[m_Pick[0]].oneOverM;
	if (dialog.DoModal() == IDOK)
	{
		m_ParticleSys[0][m_Pick[0]].oneOverM = dialog.m_VertexMass;
		m_ParticleSys[0][m_Pick[1]].oneOverM = dialog.m_VertexMass;
		m_ParticleSys[1][m_Pick[0]].oneOverM = dialog.m_VertexMass;
		m_ParticleSys[1][m_Pick[1]].oneOverM = dialog.m_VertexMass;
		m_ParticleSys[2][m_Pick[0]].oneOverM = dialog.m_VertexMass;
		m_ParticleSys[2][m_Pick[1]].oneOverM = dialog.m_VertexMass;
	}
}

void CPhysEnv::ApplyUserForce(tVector *force)
{
	ScaleVector(force,  m_UserForceMag, &m_UserForce);
	m_UserForceActive = TRUE;
}

///////////////////////////////////////////////////////////////////////////////
// Function:	SetMouseForce 
// Purpose:		Allows the user to interact with selected points by dragging
// Arguments:	Delta distance from clicked point, local x and y axes
///////////////////////////////////////////////////////////////////////////////
void CPhysEnv::SetMouseForce(int deltaX,int deltaY, tVector *localX, tVector *localY)
{
/// Local Variables ///////////////////////////////////////////////////////////
	tVector tempX,tempY;
///////////////////////////////////////////////////////////////////////////////
	ScaleVector(localX,  (float)deltaX * 0.03f, &tempX);
	ScaleVector(localY,  -(float)deltaY * 0.03f, &tempY);
	if (m_Pick[0] > -1)
	{
		VectorSum(&m_CurrentSys[m_Pick[0]].pos,&tempX,&m_MouseDragPos[0]);
		VectorSum(&m_MouseDragPos[0],&tempY,&m_MouseDragPos[0]);
	}
	if (m_Pick[1] > -1)
	{
		VectorSum(&m_CurrentSys[m_Pick[1]].pos,&tempX,&m_MouseDragPos[1]);
		VectorSum(&m_MouseDragPos[1],&tempY,&m_MouseDragPos[1]);
	}
}
/// SetMouseForce /////////////////////////////////////////////////////////////


void CPhysEnv::AddSpring()
{
	tSpring	*spring;
	// MAKE SURE TWO PARTICLES ARE PICKED
	if (m_Pick[0] > -1 && m_Pick[1] > -1)
	{
		spring = (tSpring *)malloc(sizeof(tSpring) * (m_SpringCnt + 1));
		if (m_Spring != NULL)
			memcpy(spring,m_Spring,sizeof(tSpring) * m_SpringCnt);
		m_Spring = spring;
		spring = &m_Spring[m_SpringCnt++];
		spring->Ks = m_Ksh;
		spring->Kd = m_Ksd;
		spring->p1 = m_Pick[0];
		spring->p2 = m_Pick[1];
		spring->restLen = 
			sqrt(VectorSquaredDistance(&m_CurrentSys[m_Pick[0]].pos, 
									   &m_CurrentSys[m_Pick[1]].pos));
		spring->type = 	MANUAL_SPRING;
	}
}

void CPhysEnv::AddSpring(int v1, int v2,float Ksh,float Ksd, int type)
{
	tSpring	*spring;
	// MAKE SURE TWO PARTICLES ARE PICKED
	if (v1 > -1 && v2 > -1)
	{
		spring = (tSpring *)malloc(sizeof(tSpring) * (m_SpringCnt + 1));
		if (m_Spring != NULL)
		{
			memcpy(spring,m_Spring,sizeof(tSpring) * m_SpringCnt);
			free(m_Spring);
		}
		m_Spring = spring;
		spring = &m_Spring[m_SpringCnt++];
		spring->type = type;
		spring->Ks = Ksh;
		spring->Kd = Ksd;
		spring->p1 = v1;
		spring->p2 = v2;
		spring->restLen = 
			sqrt(VectorSquaredDistance(&m_CurrentSys[v1].pos, 
									   &m_CurrentSys[v2].pos));
	}
}

void CPhysEnv::ComputeForces( tParticle	*system )
{
	int loop;
	tParticle	*curParticle,*p1, *p2;
	tSpring		*spring;
	float		dist, Hterm, Dterm;
	tVector		springForce,deltaV,deltaP;

	curParticle = system;
	for (loop = 0; loop < m_ParticleCnt; loop++)
	{
		MAKEVECTOR(curParticle->f,0.0f,0.0f,0.0f)		// CLEAR FORCE VECTOR

		if (m_UseGravity && curParticle->oneOverM != 0)
		{
			curParticle->f.x += (m_Gravity.x / curParticle->oneOverM);
			curParticle->f.y += (m_Gravity.y / curParticle->oneOverM);
			curParticle->f.z += (m_Gravity.z / curParticle->oneOverM);
		}

		if (m_UseDamping)
		{
			curParticle->f.x += (-m_Kd * curParticle->v.x);
			curParticle->f.y += (-m_Kd * curParticle->v.y);
			curParticle->f.z += (-m_Kd * curParticle->v.z);
		}
		else
		{
			curParticle->f.x += (-DEFAULT_DAMPING * curParticle->v.x);
			curParticle->f.y += (-DEFAULT_DAMPING * curParticle->v.y);
			curParticle->f.z += (-DEFAULT_DAMPING * curParticle->v.z);
		}
		curParticle++;
	}

	// CHECK IF THERE IS A USER FORCE BEING APPLIED
	if (m_UserForceActive)
	{
		if (m_Pick[0] != -1)
		{
			VectorSum(&system[m_Pick[0]].f,&m_UserForce,&system[m_Pick[0]].f);
		}
		if (m_Pick[1] != -1)
		{
			VectorSum(&system[m_Pick[1]].f,&m_UserForce,&system[m_Pick[1]].f);
		}
		MAKEVECTOR(m_UserForce,0.0f,0.0f,0.0f);	// CLEAR USER FORCE
	}

	// NOW DO ALL THE SPRINGS
	spring = m_Spring;
	for (loop = 0; loop < m_SpringCnt; loop++)
	{
		p1 = &system[spring->p1];
		p2 = &system[spring->p2];
		VectorDifference(&p1->pos,&p2->pos,&deltaP);	// Vector distance 
		dist = VectorLength(&deltaP);					// Magnitude of deltaP

		Hterm = (dist - spring->restLen) * spring->Ks;	// Ks * (dist - rest)
		
		VectorDifference(&p1->v,&p2->v,&deltaV);		// Delta Velocity Vector
		Dterm = (DotProduct(&deltaV,&deltaP) * spring->Kd) / dist; // Damping Term
		
		ScaleVector(&deltaP,1.0f / dist, &springForce);	// Normalize Distance Vector
		ScaleVector(&springForce,-(Hterm + Dterm),&springForce);	// Calc Force
		VectorSum(&p1->f,&springForce,&p1->f);			// Apply to Particle 1
		VectorDifference(&p2->f,&springForce,&p2->f);	// - Force on Particle 2
		spring++;					// DO THE NEXT SPRING
	}

	// APPLY THE MOUSE DRAG FORCES IF THEY ARE ACTIVE
	if (m_MouseForceActive)
	{
		// APPLY TO EACH PICKED PARTICLE
		if (m_Pick[0] > -1)
		{
			p1 = &system[m_Pick[0]];
			VectorDifference(&p1->pos,&m_MouseDragPos[0],&deltaP);	// Vector distance 
			dist = VectorLength(&deltaP);					// Magnitude of deltaP

			if (dist != 0.0f)
			{
				Hterm = (dist) * m_MouseForceKs;					// Ks * dist

				ScaleVector(&deltaP,1.0f / dist, &springForce);	// Normalize Distance Vector
				ScaleVector(&springForce,-(Hterm),&springForce);	// Calc Force
				VectorSum(&p1->f,&springForce,&p1->f);			// Apply to Particle 1
			}
		}
		if (m_Pick[1] > -1)
		{
			p1 = &system[m_Pick[1]];
			VectorDifference(&p1->pos,&m_MouseDragPos[1],&deltaP);	// Vector distance 
			dist = VectorLength(&deltaP);					// Magnitude of deltaP

			if (dist != 0.0f)
			{
				Hterm = (dist) * m_MouseForceKs;					// Ks * dist

				ScaleVector(&deltaP,1.0f / dist, &springForce);	// Normalize Distance Vector
				ScaleVector(&springForce,-(Hterm),&springForce);	// Calc Force
				VectorSum(&p1->f,&springForce,&p1->f);			// Apply to Particle 1
			}
		}
	}
}   

///////////////////////////////////////////////////////////////////////////////
// Function:	IntegrateSysOverTime 
// Purpose:		Does the Integration for all the points in a system
// Arguments:	Initial Position, Source and Target Particle Systems and Time
// Notes:		Computes a single integration step
///////////////////////////////////////////////////////////////////////////////
void CPhysEnv::IntegrateSysOverTime(tParticle *initial,tParticle *source, tParticle *target, float deltaTime)
{
/// Local Variables ///////////////////////////////////////////////////////////
	int loop;
	float deltaTimeMass;
///////////////////////////////////////////////////////////////////////////////
	for (loop = 0; loop < m_ParticleCnt; loop++)
	{
		deltaTimeMass = deltaTime * initial->oneOverM;
		// DETERMINE THE NEW VELOCITY FOR THE PARTICLE
		target->v.x = initial->v.x + (source->f.x * deltaTimeMass);
		target->v.y = initial->v.y + (source->f.y * deltaTimeMass);
		target->v.z = initial->v.z + (source->f.z * deltaTimeMass);

		target->oneOverM = initial->oneOverM;

		// SET THE NEW POSITION
		target->pos.x = initial->pos.x + (deltaTime * source->v.x);
		target->pos.y = initial->pos.y + (deltaTime * source->v.y);
		target->pos.z = initial->pos.z + (deltaTime * source->v.z);

		initial++;
		source++;
		target++;
	}
}

float CPhysEnv::CalculateErrorForHeun(tParticle * currentIteration, tParticle * previousIteration)
{
	float accumulativeError = 0.0f;

	for (int i = 0; i < m_ParticleCnt; i++)
	{
		accumulativeError += fabsf((currentIteration->pos.x - previousIteration->pos.x) / currentIteration->pos.x) * 100;
		accumulativeError += fabsf((currentIteration->pos.y - previousIteration->pos.y) / currentIteration->pos.y) * 100;
		accumulativeError += fabsf((currentIteration->pos.z - previousIteration->pos.z) / currentIteration->pos.z) * 100;
	}

	return accumulativeError / (3 * m_ParticleCnt);
}

///////////////////////////////////////////////////////////////////////////////
// Function:	EulerIntegrate 
// Purpose:		Calculate new Positions and Velocities given a deltatime
// Arguments:	DeltaTime that has passed since last iteration
// Notes:		This integrator uses Euler's method
///////////////////////////////////////////////////////////////////////////////
void CPhysEnv::EulerIntegrate( float DeltaTime)
{
	// JUST TAKE A SINGLE STEP
	IntegrateSysOverTime(m_CurrentSys,m_CurrentSys, m_TargetSys,DeltaTime);
}

///////////////////////////////////////////////////////////////////////////////
// Function:	MidPointIntegrate 
// Purpose:		Calculate new Positions and Velocities given a deltatime
// Arguments:	DeltaTime that has passed since last iteration
// Notes:		This integrator uses the Midpoint method
///////////////////////////////////////////////////////////////////////////////
void CPhysEnv::MidPointIntegrate( float DeltaTime)
{
/// Local Variables ///////////////////////////////////////////////////////////
	float		halfDeltaT;
///////////////////////////////////////////////////////////////////////////////
	halfDeltaT = DeltaTime / 2.0f;

	// TAKE A HALF STEP AND UPDATE VELOCITY AND POSITION - Find y at half-interval
	IntegrateSysOverTime(m_CurrentSys,m_CurrentSys,m_TempSys[0],halfDeltaT);

	// COMPUTE FORCES USING THESE NEW POSITIONS AND VELOCITIES - Compute slope at half-interval 
	ComputeForces(m_TempSys[0]);

	// TAKE THE FULL STEP WITH THIS NEW INFORMATION - Now use above slope for full-interval
	IntegrateSysOverTime(m_CurrentSys,m_TempSys[0],m_TargetSys,DeltaTime);
}

///////////////////////////////////////////////////////////////////////////////
// Function:	HeunIntegrate 
// Purpose:		Calculate new Positions and Velocities given a deltatime
// Arguments:	DeltaTime that has passed since last iteration
// Notes:		This integrator uses the Heun's method
///////////////////////////////////////////////////////////////////////////////
void CPhysEnv::HeunIntegrate(float DeltaTime)
{
	int maxNumberOfIterations = 15, i;
	float errors[2];

	IntegrateSysOverTime(m_CurrentSys, m_CurrentSys, m_TempSys[0], DeltaTime);

	for (i = 0; i < maxNumberOfIterations; ++i)
	{
		ComputeForces(m_TempSys[i % 2]);

		IntegrateSysOverTime(m_CurrentSys, m_TempSys[i % 2], m_TempSys[(i + 1) % 2], DeltaTime);

		errors[(i + 1) % 2] = CalculateErrorForHeun(m_TempSys[(i + 1) % 2], m_TempSys[i % 2]);

		if (errors[(i + 1) % 2] > errors[i % 2])
		{
			break;
		}
	}

	ComputeForces(m_TempSys[i % 2]);

	tParticle *source, *target;
	target = m_TargetSys;
	source = m_TempSys[i % 2];

	for (int loop = 0; loop < m_ParticleCnt; loop++)
	{
		target->v.x = source->v.x;
		target->v.y = source->v.y;
		target->v.z = source->v.z;

		target->oneOverM = source->oneOverM;

		target->pos.x = source->pos.x;
		target->pos.y = source->pos.y;
		target->pos.z = source->pos.z;

		source++;
		target++;
	}
}


///////////////////////////////////////////////////////////////////////////////
// Function:	RK4Integrate 
// Purpose:		Calculate new Positions and Velocities given a deltatime
// Arguments:	DeltaTime that has passed since last iteration
// Notes:		This integrator uses the fourth-order Runge-Kutta method 
///////////////////////////////////////////////////////////////////////////////
void CPhysEnv::RK4Integrate(float DeltaTime, tParticle *initial, tParticle *target)
{
	/// Local Variables ///////////////////////////////////////////////////////////
	float		halfDeltaT;
	///////////////////////////////////////////////////////////////////////////////
	halfDeltaT = DeltaTime / 2.0f;


	// TAKE A HALF STEP AND UPDATE VELOCITY AND POSITION - Find y at half-interval - using forces from K2 (K2)
	IntegrateSysOverTime(initial, initial, m_TempSys[0], halfDeltaT);
	
	// COMPUTE FORCES USING THESE NEW POSITIONS AND VELOCITIES - Compute slope at half-interval 
	ComputeForces(m_TempSys[0]);

	// TAKE A HALF STEP AND UPDATE VELOCITY AND POSITION - Find y at half-interval - using forces from K2 (K3)
	IntegrateSysOverTime(initial, m_TempSys[0], m_TempSys[1], halfDeltaT);

	// COMPUTE FORCES USING THESE NEW POSITIONS AND VELOCITIES - Compute slope at half-interval 
	ComputeForces(m_TempSys[1]);

	// TAKE A HALF STEP AND UPDATE VELOCITY AND POSITION - Find y at interval - using forces from K3 (K4)
	IntegrateSysOverTime(initial, m_TempSys[1], m_TempSys[2], DeltaTime);

	// COMPUTE FORCES USING THESE NEW POSITIONS AND VELOCITIES - Compute slope at full-interval 
	ComputeForces(m_TempSys[2]);

	tParticle *k1, *k2, *k3, *k4, *goal_t;
	k1 = initial;
	k2 = m_TempSys[0];
	k3 = m_TempSys[1];
	k4 = m_TempSys[2];
	goal_t = m_TempSys[3];// keeping the address that m_TempSys[3] carries away from the ++!

	for (int loop = 0; loop < m_ParticleCnt; loop++)
	{
		goal_t->f.x = (k1->f.x + 2 * (k2->f.x + k3->f.x) + k4->f.x) / 6;
		goal_t->f.y = (k1->f.y + 2 * (k2->f.y + k3->f.y) + k4->f.y) / 6;
		goal_t->f.z = (k1->f.z + 2 * (k2->f.z + k3->f.z) + k4->f.z) / 6;

		goal_t->v.x = (k1->v.x + 2 * (k2->v.x + k3->v.x) + k4->v.x) / 6;
		goal_t->v.y = (k1->v.y + 2 * (k2->v.y + k3->v.y) + k4->v.y) / 6;
		goal_t->v.z = (k1->v.z + 2 * (k2->v.z + k3->v.z) + k4->v.z) / 6;
		
		k1++;
		k2++;
		k3++;
		k4++;
		goal_t++;
	}

	IntegrateSysOverTime(initial, m_TempSys[3], target, DeltaTime);
}

///////////////////////////////////////////////////////////////////////////////
// Function:	RK4AdaptiveIntegrate 
// Purpose:		Calculate new Positions and Velocities given a deltatime
// Arguments:	DeltaTime that has passed since last iteration
// Notes:		This integrator uses the fourth-order Runge-Kutta method based on step-halving technique
///////////////////////////////////////////////////////////////////////////////
void CPhysEnv::RK4AdaptiveIntegrate( float DeltaTime)  
{
	//Try the full-step
	tParticle * particle_sys_full_stp  = (tParticle *)malloc(sizeof(tParticle) * m_ParticleCnt);
	RK4Integrate(DeltaTime, m_CurrentSys, particle_sys_full_stp);

	//Try the half-step
	tParticle * particle_sys_half_stp  = (tParticle *)malloc(sizeof(tParticle) * m_ParticleCnt);
	float		halfDeltaT;
	halfDeltaT = DeltaTime / 2.0f;

	RK4Integrate(halfDeltaT, m_CurrentSys, m_TempSys[4]);
	ComputeForces(m_TempSys[4]);
	RK4Integrate(halfDeltaT, m_TempSys[4], m_TargetSys);
	
	//Calculate error over all the particles
	tParticle *full_stp, *half_stp;
	/* float abs_dist_err = 0; */

	full_stp = particle_sys_full_stp;
	half_stp = m_TargetSys;

	for (int loop = 0; loop < m_ParticleCnt; loop++)
	{

		half_stp->pos.x += (half_stp->pos.x - full_stp->pos.x)/15;
		half_stp->pos.y += (half_stp->pos.y - full_stp->pos.y)/15;
		half_stp->pos.z += (half_stp->pos.z - full_stp->pos.z)/15;

		full_stp++;
		half_stp++;
	}

	/* if (m_ParticleCnt !=0)
		abs_dist_err = abs_dist_err / m_ParticleCnt;


	// Push the adapted particle system into the target
	tParticle *target_m, *adap_part_sys;
	
	adap_part_sys = particle_sys_full_stp;


	if (abs_dist_err > 0.001)
	{
		adap_part_sys = particle_sys_half_stp;
	}

	target_m = m_TargetSys;

	for (int loop = 0; loop < m_ParticleCnt; loop++)
	{
		*target_m = *adap_part_sys;

		target_m++;
		adap_part_sys++;
	}
 */
	free(particle_sys_full_stp);
	/* free(particle_sys_half_stp); */
}
///////////////////////////////////////////////////////////////////////////////

int CPhysEnv::CheckForCollisions( tParticle	*system )
{
    // be optimistic!
    int collisionState = NOT_COLLIDING;
    float const depthEpsilon = 0.001f;

	int loop;
	tParticle *curParticle;

	m_ContactCnt = 0;		// THERE ARE CURRENTLY NO CONTACTS

	curParticle = system;
	for (loop = 0; (loop < m_ParticleCnt) && (collisionState != PENETRATING); 
			loop++,curParticle++)
	{
		// CHECK THE MAIN BOUNDARY PLANES FIRST
        for(int planeIndex = 0;(planeIndex < m_CollisionPlaneCnt) &&
            (collisionState != PENETRATING);planeIndex++)
        {
			tCollisionPlane *plane = &m_CollisionPlane[planeIndex];

            float axbyczd = DotProduct(&curParticle->pos,&plane->normal) + plane->d;

            if(axbyczd < -depthEpsilon)
            {
				// ONCE ANY PARTICLE PENETRATES, QUIT THE LOOP
				collisionState = PENETRATING;
            }
            else
            if(axbyczd < depthEpsilon)
            {
                float relativeVelocity = DotProduct(&plane->normal,&curParticle->v);

                if(relativeVelocity < 0.0f)
                {
                    collisionState = COLLIDING;
					m_Contact[m_ContactCnt].particle = loop; 
					memcpy(&m_Contact[m_ContactCnt].normal,&plane->normal,sizeof(tVector));
					m_ContactCnt++;
                }
            }
        }
		if (m_CollisionActive)
		{
			// THIS IS NEW FROM MARCH - ADDED SPHERE BOUNDARIES
			// CHECK ANY SPHERE BOUNDARIES
			for(int sphereIndex = 0;(sphereIndex < m_SphereCnt) &&
				(collisionState != PENETRATING);sphereIndex++)
			{
				tCollisionSphere *sphere = &m_Sphere[sphereIndex];

				tVector	distVect;

				VectorDifference(&curParticle->pos, &sphere->pos, &distVect);

				float radius = VectorSquaredLength(&distVect);
				// SINCE IT IS TESTING THE SQUARED DISTANCE, SQUARE THE RADIUS ALSO
				float dist = radius - (sphere->radius * sphere->radius);

				if(dist < -depthEpsilon)
				{
					// ONCE ANY PARTICLE PENETRATES, QUIT THE LOOP
					collisionState = PENETRATING;
				}
				else
				if(dist < depthEpsilon)
				{
					// NORMALIZE THE VECTOR
					NormalizeVector(&distVect);

					float relativeVelocity = DotProduct(&distVect,&curParticle->v);

					if(relativeVelocity < 0.0f)
					{
						collisionState = COLLIDING;
						m_Contact[m_ContactCnt].particle = loop; 
						memcpy(&m_Contact[m_ContactCnt].normal,&distVect,sizeof(tVector));
						m_ContactCnt++;
					}
				}
			}
		}

	}

    return collisionState;
}

void CPhysEnv::ResolveCollisions( tParticle	*system )
{
	tContact	*contact;
	tParticle	*particle;		// THE PARTICLE COLLIDING
	float		VdotN;		
	tVector		Vn,Vt;				// CONTACT RESOLUTION IMPULSE
	contact = m_Contact;
	for (int loop = 0; loop < m_ContactCnt; loop++,contact++)
	{
		particle = &system[contact->particle];
		// CALCULATE Vn
		VdotN = DotProduct(&contact->normal,&particle->v);
		ScaleVector(&contact->normal, VdotN, &Vn);
		// CALCULATE Vt
		VectorDifference(&particle->v, &Vn, &Vt);
		// SCALE Vn BY COEFFICIENT OF RESTITUTION
		ScaleVector(&Vn, m_Kr, &Vn);
		// SET THE VELOCITY TO BE THE NEW IMPULSE
		VectorDifference(&Vt, &Vn, &particle->v);
	}
}

void CPhysEnv::Simulate(float DeltaTime, BOOL running)
{
    float		CurrentTime = 0.0f;
    float		TargetTime = DeltaTime;
	tParticle	*tempSys;
	int			collisionState;
	std::ofstream myfile;
	static bool frist_time=true;
	static float time = 0;
	myfile.open("test_currentSys.txt", std::ios::app);
	if (frist_time)
	{
		myfile << "Pcount" << m_ParticleCnt << "\n";
		frist_time = false;
	}

    while(CurrentTime < DeltaTime)
    {
		if (m_ParticleCnt != 0) {
			time += DeltaTime;
			myfile << "t=" << time << "\n";
		}
		for (int loop = 0; loop < m_ParticleCnt; loop++)
		{
			myfile << "P" << loop << "=" << m_CurrentSys->pos.y << "\n";
			m_CurrentSys++;
		}
		m_CurrentSys -= m_ParticleCnt;
		myfile.close();

		if (running)
		{
			ComputeForces(m_CurrentSys);

			// IN ORDER TO MAKE THINGS RUN FASTER, I HAVE THIS LITTLE TRICK
			// IF THE SYSTEM IS DOING A BINARY SEARCH FOR THE COLLISION POINT,
			// I FORCE EULER'S METHOD ON IT. OTHERWISE, LET THE USER CHOOSE.
			// THIS DOESN'T SEEM TO EFFECT STABILITY EITHER WAY
			if (m_CollisionRootFinding)
			{
				EulerIntegrate(TargetTime-CurrentTime);
			}
			else
			{
				
				switch (m_IntegratorType)
				{
				case EULER_INTEGRATOR:
					EulerIntegrate(TargetTime-CurrentTime);
					break;
				case MIDPOINT_INTEGRATOR:
					MidPointIntegrate(TargetTime-CurrentTime);
					break;
				case HEUN_INTEGRATOR:
					HeunIntegrate(TargetTime-CurrentTime);
					break;
				case RK4_INTEGRATOR:
					RK4Integrate(TargetTime - CurrentTime, m_CurrentSys, m_TargetSys);
					break;
				case RK4_ADAPTIVE_INTEGRATOR:
					RK4AdaptiveIntegrate(TargetTime-CurrentTime);
					break;
				}
			}
		}

		collisionState = CheckForCollisions(m_TargetSys);

        if(collisionState == PENETRATING)
        {
			// TELL THE SYSTEM I AM LOOKING FOR A COLLISION SO IT WILL USE EULER
			m_CollisionRootFinding = TRUE;
            // we simulated too far, so subdivide time and try again
            TargetTime = (CurrentTime + TargetTime) / 2.0f;

            // blow up if we aren't moving forward each step, which is
            // probably caused by interpenetration at the frame start
            assert(fabs(TargetTime - CurrentTime) > EPSILON);
        }
        else
        {
            // either colliding or clear
            if(collisionState == COLLIDING)
            {
                int Counter = 0;
                do
                {
                    ResolveCollisions(m_TargetSys);
                    Counter++;
                } while((CheckForCollisions(m_TargetSys) ==
                            COLLIDING) && (Counter < 100));

                assert(Counter < 100);
				m_CollisionRootFinding = FALSE;  // FOUND THE COLLISION POINT
            }

            // we made a successful step, so swap configurations
            // to "save" the data for the next step
            
			CurrentTime = TargetTime;
			TargetTime = DeltaTime;

			// SWAP MY TWO PARTICLE SYSTEM BUFFERS SO I CAN DO IT AGAIN
			tempSys = m_CurrentSys;
			m_CurrentSys = m_TargetSys;
			m_TargetSys = tempSys;
        }
    }
}

///////////////////////////////////////////////////////////////////////////////
// Function:	AddCollisionSphere 
// Purpose:		Add a collision sphere to the system
///////////////////////////////////////////////////////////////////////////////
void CPhysEnv::AddCollisionSphere()
{
/// Local Variables ///////////////////////////////////////////////////////////
	tCollisionSphere *temparray;
	CAddSpher		dialog;
///////////////////////////////////////////////////////////////////////////////
	dialog.m_Radius = 2.0f;
	dialog.m_XPos = 0.0f;
	dialog.m_YPos = -3.0f;
	dialog.m_ZPos = 0.0f;
	if (dialog.DoModal())
	{
		temparray = (tCollisionSphere *)malloc(sizeof(tCollisionSphere) * (m_SphereCnt+1)); 	
		if (m_SphereCnt > 0)
		{
			memcpy(temparray,m_Sphere,sizeof(tCollisionSphere) * m_SphereCnt);
			free(m_Sphere);
		}
		
		MAKEVECTOR(temparray[m_SphereCnt].pos,dialog.m_XPos,dialog.m_YPos, dialog.m_ZPos)
		temparray[m_SphereCnt].radius = dialog.m_Radius;
	
		m_Sphere = temparray;
		m_SphereCnt++;
	}
}